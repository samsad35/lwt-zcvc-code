import os, sys, time, torch, torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
import jiwer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.inference.speaker import EncoderClassifier
import torchaudio.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.core.models import compute_trajectory_jitter

device = 'cuda:0'

print("================================================================================")
print("=== EVALUATING LOCAL ORTHOGONAL PROCRUSTES ON UNIFIED 200 BENCHMARK ===")
print("================================================================================")

# 1. Models
wavlm = torch.hub.load('bshall/knn-vc', 'wavlm_large', trust_repo=True, device=device).eval()
hifigan, _ = torch.hub.load('bshall/knn-vc', 'hifigan_wavlm', trust_repo=True, prematched=True, device=device)
hifigan.eval()
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
asr = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h').to(device).eval()
spk_model = EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb', 
    run_opts={'device': 'cuda:0'}, 
    savedir='/tmp/speechbrain'
)

def ext(p):
    w, sr = torchaudio.load(str(p))
    w = w.to(device)
    if sr != 16000: w = F.resample(w, sr, 16000)
    if w.dim() == 1: w = w.unsqueeze(0)
    with torch.no_grad(): feat, _ = wavlm.extract_features(w, output_layer=6)
    return feat.squeeze(0).cpu().numpy(), w.squeeze().cpu()

def voc(f):
    with torch.inference_mode():
        if isinstance(f, np.ndarray): f = torch.tensor(f, dtype=torch.float32, device=device)
        return hifigan(f.unsqueeze(0)).squeeze(0).cpu()

def tr(w):
    inp = processor(w.squeeze().numpy(), sampling_rate=16000, return_tensors='pt', padding=True).to(device)
    with torch.no_grad(): log = asr(inp.input_values).logits
    pred_ids = torch.argmax(log, dim=-1)
    return processor.batch_decode(pred_ids)[0].strip()

def emb(w):
    with torch.no_grad():
        return torch.nn.functional.normalize(
            spk_model.encode_batch(w.squeeze().float().unsqueeze(0).to(device)).squeeze().cpu(), 
            dim=0
        )

root = Path('/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean')

# 10 Target Speakers
target_speakers = [
    '121', '1089', '1221', '3570', '3729', # Female
    '237', '260', '1188', '4446', '4507'   # Male
]

# 20 Source Utterances from 10 distinct speakers
src_spks = ['1284', '1320', '1580', '1995', '2094', '2300', '2830', '2961', '3575', '4077']
src_files = []
for s in src_spks:
    f_list = sorted(list((root / s).rglob('*.flac')))
    if len(f_list) >= 2:
        src_files.extend(f_list[:2])
    else:
        src_files.extend(f_list)
src_files = src_files[:20]

print("Pre-extracting source features and references...")
src_data = []
for p in src_files:
    feat, wav = ext(p)
    ref = tr(wav)
    src_data.append({
        'path': p,
        'spk': p.parent.parent.name,
        'feat': feat,
        'ref': ref,
        'duration': len(feat) * 0.02
    })

# Background shared clusters K=10
bg_spks = ['4970', '4992', '5142', '5639', '5683', '61']
bg_feats = []
for b_spk in bg_spks:
    b_files = sorted(list((root / b_spk).rglob('*.flac')))[:6]
    for bf in b_files:
        x, _ = ext(bf)
        bg_feats.append(x)
bg_Y = np.concatenate(bg_feats, axis=0)

K_clusters = 10
beta_val = 20.0
km_shared = KMeans(n_clusters=K_clusters, random_state=42, n_init=1).fit(bg_Y)
cents_shared = torch.tensor(km_shared.cluster_centers_, dtype=torch.float32, device=device)
cents_shared_norm = torch.nn.functional.normalize(cents_shared, dim=1)

# Target models (T=20 utts)
print("Building target representations (T=20)...")
target_models = {}
for spk in target_speakers:
    files = sorted(list((root / spk).rglob('*.flac')))[:20]
    X_list, wavs = [], []
    for p in files:
        x, w = ext(p); X_list.append(x); wavs.append(w)
    Y = np.concatenate(X_list, axis=0)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    Y_norm = torch.nn.functional.normalize(Y_t, dim=1)
    prof = torch.stack([emb(w) for w in wavs[:10]]).mean(dim=0)
    prof = torch.nn.functional.normalize(prof, dim=0)
    
    sim_Y = Y_norm @ cents_shared_norm.t()
    w_Y = torch.softmax(sim_Y * beta_val, dim=1)
    
    cl_mu = []
    cl_cov = []
    for k in range(K_clusters):
        wk = w_Y[:, k:k+1]
        mass_k = wk.sum()
        muk = (Y_t * wk).sum(dim=0) / (mass_k + 1e-8)
        yc = Y_t - muk.unsqueeze(0)
        covk = torch.mm(yc.t(), yc * wk) / (mass_k + 1e-8)
        cl_mu.append(muk)
        cl_cov.append(covk)
        
    target_models[spk] = {
        'Y_t': Y_t,
        'Y_norm': Y_norm,
        'prof': prof,
        'cl_mu': cl_mu,
        'cl_cov': cl_cov
    }

procrustes_variants = [
    ('Local Stat-Procrustes (Ours)', 'Piecewise Orthogonal / Covariance'),
    ('Local Point-Procrustes (Ours)', 'Piecewise Orthogonal / Instance')
]

all_procrustes_records = []

for method_name, approach_type in procrustes_variants:
    print(f"\nEvaluating {method_name} across 200 conversions...")
    t0_m = time.time()
    count = 0
    for tgt_spk, tm in target_models.items():
        for s_item in src_data:
            xs = s_item['feat']
            ref = s_item['ref']
            duration = s_item['duration']
            
            xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
            xs_n = torch.nn.functional.normalize(xs_t, dim=1)
            sim_xs = xs_n @ cents_shared_norm.t()
            w_dyn = torch.softmax(sim_xs * beta_val, dim=1)
            
            t0 = time.perf_counter()
            
            if method_name == 'Local Stat-Procrustes (Ours)':
                # Statistical Procrustes on Covariance Square Roots
                x_hat = torch.zeros_like(xs_t)
                for k in range(K_clusters):
                    wk = w_dyn[:, k:k+1]
                    mass_k = wk.sum()
                    if mass_k < 1e-4: continue
                    mu_X_k = (xs_t * wk).sum(dim=0) / mass_k
                    xc = xs_t - mu_X_k.unsqueeze(0)
                    cov_X_k = torch.mm(xc.t(), xc * wk) / mass_k
                    
                    evals_X, evecs_X = torch.linalg.eigh(cov_X_k + 1e-3 * torch.eye(1024, device=device))
                    cX_half = evecs_X @ torch.diag(torch.sqrt(evals_X.clamp(min=1e-3))) @ evecs_X.t()
                    
                    evals_Y, evecs_Y = torch.linalg.eigh(tm['cl_cov'][k] + 1e-3 * torch.eye(1024, device=device))
                    cY_half = evecs_Y @ torch.diag(torch.sqrt(evals_Y.clamp(min=1e-3))) @ evecs_Y.t()
                    
                    M = cX_half @ cY_half
                    U, _, Vh = torch.linalg.svd(M)
                    R_k = U @ Vh
                    
                    T_k = torch.mm(xc, R_k) + tm['cl_mu'][k].unsqueeze(0)
                    x_hat += wk * T_k
                x_hat_np = x_hat.cpu().numpy()
                
            elif method_name == 'Local Point-Procrustes (Ours)':
                # Matched Instance Procrustes
                sim_nn = torch.mm(xs_n, tm['Y_norm'].t())
                best_idx = torch.argmax(sim_nn, dim=1)
                Y_matched = tm['Y_t'][best_idx]
                
                x_hat = torch.zeros_like(xs_t)
                for k in range(K_clusters):
                    wk = w_dyn[:, k:k+1]
                    mass_k = wk.sum()
                    if mass_k < 1e-4: continue
                    mu_X_k = (xs_t * wk).sum(dim=0) / mass_k
                    xc = xs_t - mu_X_k.unsqueeze(0)
                    
                    mu_Y_k = tm['cl_mu'][k]
                    yc = Y_matched - mu_Y_k.unsqueeze(0)
                    
                    S_k = torch.mm(xc.t(), yc * wk)
                    U, _, Vh = torch.linalg.svd(S_k)
                    R_k = U @ Vh
                    
                    T_k = torch.mm(xc, R_k) + mu_Y_k.unsqueeze(0)
                    x_hat += wk * T_k
                x_hat_np = x_hat.cpu().numpy()
                
            t1 = time.perf_counter()
            rtf = (t1 - t0) / duration
            
            _, rjit = compute_trajectory_jitter(x_hat_np, xs)
            wc = voc(x_hat_np)
            hyp = tr(wc)
            cer = jiwer.cer(ref, hyp) * 100.0
            wer = jiwer.wer(ref, hyp) * 100.0
            sim = torch.dot(tm['prof'], emb(wc)).item()
            
            all_procrustes_records.append({
                'Method': method_name,
                'Approach_Type': approach_type,
                'Target_Speaker': tgt_spk,
                'Source_Speaker': s_item['spk'],
                'CER': cer,
                'WER': wer,
                'Sim': sim,
                'Jitter': rjit,
                'RTF': rtf
            })
            count += 1
            if count % 50 == 0:
                print(f"  Processed {count}/200 conversions...")
                
    print(f"  --> {method_name} completed in {time.time()-t0_m:.1f}s.")

df_proc = pd.DataFrame(all_procrustes_records)
summary_proc = df_proc.groupby(['Method', 'Approach_Type']).agg({
    'CER': ['mean', 'std'],
    'WER': ['mean', 'std'],
    'Sim': ['mean', 'std'],
    'Jitter': ['mean', 'std'],
    'RTF': ['mean', 'std']
}).reset_index()

summary_proc.columns = [
    'Method', 'Approach_Type',
    'CER_mean', 'CER_std',
    'WER_mean', 'WER_std',
    'Sim_mean', 'Sim_std',
    'Jitter_mean', 'Jitter_std',
    'RTF_mean', 'RTF_std'
]

print("\n================================================================================")
print("=== PROCRUSTES VARIANTS SUMMARY (N=200 CONVERSIONS) ===")
print("================================================================================")
print(summary_proc.to_string())

out_proc_csv = Path('/local_scratch/ssadok/un_projet_audio/output/tables/procrustes_200_summary.csv')
summary_proc.to_csv(out_proc_csv, index=False)
df_proc.to_csv('/local_scratch/ssadok/un_projet_audio/output/tables/procrustes_200_conversions_detail.csv', index=False)
print(f"Saved Procrustes evaluation to {out_proc_csv}")
