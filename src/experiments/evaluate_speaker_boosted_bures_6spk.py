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

print("=== Evaluating Speaker-Covariance-Boosted Bures-Wasserstein on 6 Speakers ===")
wavlm = torch.hub.load('bshall/knn-vc', 'wavlm_large', trust_repo=True, device=device).eval()
hifigan, _ = torch.hub.load('bshall/knn-vc', 'hifigan_wavlm', trust_repo=True, prematched=True, device=device)
hifigan.eval()

processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
asr = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h').to(device).eval()
spk_model = EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', run_opts={'device': 'cuda:0'}, savedir='/tmp/speechbrain')

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
    return processor.batch_decode(torch.argmax(log, dim=-1))[0]

def emb(w):
    with torch.no_grad():
        return torch.nn.functional.normalize(spk_model.encode_batch(w.squeeze().float().unsqueeze(0).to(device)).squeeze().cpu(), dim=0)

def project_psd(M, eps=1e-3):
    evals, evecs = torch.linalg.eigh(M)
    evals = torch.clamp(evals, min=eps)
    return evecs @ torch.diag(evals) @ evecs.t()

def compute_bures_map_torch(cov_X, cov_Y, eps=1e-2):
    D = cov_X.shape[0]
    I_D = torch.eye(D, device=cov_X.device, dtype=cov_X.dtype)
    cX = cov_X + eps * I_D
    cY = cov_Y + eps * I_D
    
    evals_X, evecs_X = torch.linalg.eigh(cX)
    evals_X = evals_X.clamp(min=eps)
    cX_half = evecs_X @ torch.diag(torch.sqrt(evals_X)) @ evecs_X.t()
    cX_inv_half = evecs_X @ torch.diag(1.0 / torch.sqrt(evals_X)) @ evecs_X.t()
    
    M = cX_half @ cY @ cX_half
    evals_M, evecs_M = torch.linalg.eigh(M)
    evals_M = evals_M.clamp(min=eps**2)
    M_half = evecs_M @ torch.diag(torch.sqrt(evals_M)) @ evecs_M.t()
    
    A = cX_inv_half @ M_half @ cX_inv_half
    return A

root = Path('/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean')

# 6 balanced target speakers (3F, 3M)
target_speakers = ['121', '237', '1221', '260', '1089', '1188']
# 6 diverse source test sentences
src_spks = ['1284', '1320', '1580', '1995', '2300', '2830']
src_files = [list((root / s).rglob('*.flac'))[0] for s in src_spks]

print(f"Pre-extracting {len(src_files)} source test utterances...")
src_data = [(ext(p)[0], tr(ext(p)[1])) for p in src_files]

T_target_utts = 20
K_clusters = 10
beta_val = 20.0

# Build Population Shared Phonetic Covariance from background speech (4 diverse speakers)
print("Building Population Shared Phonetic Covariance from background speech...")
bg_spks = ['3570', '3575', '4446', '4507', '5105', '5142']
bg_feats = []
for b_spk in bg_spks:
    b_files = sorted(list((root / b_spk).rglob('*.flac')))[:10]
    for bf in b_files:
        x, _ = ext(bf)
        bg_feats.append(x)
bg_Y = np.concatenate(bg_feats, axis=0)

# Multi-speaker KMeans to establish the shared canonical acoustic clusters
km_shared = KMeans(n_clusters=K_clusters, random_state=42, n_init=1).fit(bg_Y)
cents_shared = torch.tensor(km_shared.cluster_centers_, dtype=torch.float32, device=device)
cents_shared_norm = torch.nn.functional.normalize(cents_shared, dim=1)

cov_shared_k = []
for k in range(K_clusters):
    idx_k = np.where(km_shared.labels_ == k)[0]
    Yk = bg_Y[idx_k]
    muk = np.mean(Yk, axis=0)
    covk = np.cov(Yk - muk, rowvar=False) if len(Yk) > 1 else np.zeros((1024, 1024))
    cov_shared_k.append(torch.tensor(covk, dtype=torch.float32, device=device))

print("Universal shared phonetic covariances computed for K=10.")

alphas = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
records = []

for spk_idx, spk in enumerate(target_speakers):
    print(f"\n=======================================================")
    print(f"[{spk_idx+1}/{len(target_speakers)}] Target Speaker: {spk} (T={T_target_utts} utterances)")
    print(f"=======================================================")
    
    files = sorted(list((root / spk).rglob('*.flac')))[:T_target_utts]
    X_list, wavs = [], []
    for p in files:
        x, w = ext(p); X_list.append(x); wavs.append(w)
    Y = np.concatenate(X_list, axis=0)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    Y_n = torch.nn.functional.normalize(Y_t, dim=1)
    
    prof = torch.stack([emb(w) for w in wavs[:10]]).mean(dim=0)
    prof = torch.nn.functional.normalize(prof, dim=0)
    
    # Soft assignment of target Y to shared clusters
    sim_Y = Y_n @ cents_shared_norm.t()
    w_Y = torch.softmax(sim_Y * beta_val, dim=1)
    
    cl_mu_Y = []
    cl_cov_Y = []
    delta_cov_speaker = []
    for k in range(K_clusters):
        wk = w_Y[:, k:k+1]
        mass_k = wk.sum()
        muk = (Y_t * wk).sum(dim=0) / (mass_k + 1e-8)
        yc = Y_t - muk.unsqueeze(0)
        cov_Y_k = torch.mm(yc.t(), yc * wk) / (mass_k + 1e-8)
        cl_mu_Y.append(muk)
        cl_cov_Y.append(cov_Y_k)
        
        # Speaker Delta: Sigma_{Y, k} - Sigma_{shared, k}
        d_cov = cov_Y_k - cov_shared_k[k]
        delta_cov_speaker.append(d_cov)

    for alpha in alphas:
        cers, sims, rel_jitters = [], [], []
        t0 = time.perf_counter()
        
        for xs, ref in src_data:
            xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
            xs_n = torch.nn.functional.normalize(xs_t, dim=1)
            
            # Shared Routing w_k(x)
            sim_mat = xs_n @ cents_shared_norm.t()
            w_dyn = torch.softmax(sim_mat * beta_val, dim=1)
            
            x_hat = torch.zeros_like(xs_t)
            for k in range(K_clusters):
                wk = w_dyn[:, k:k+1]
                mass_k = wk.sum()
                if mass_k < 1e-4:
                    continue
                mu_X_k = (xs_t * wk).sum(dim=0) / mass_k
                xc = xs_t - mu_X_k.unsqueeze(0)
                cov_X_k = torch.mm(xc.t(), xc * wk) / mass_k
                
                # --- Speaker Boosted Covariance ---
                cov_Y_boosted = cov_shared_k[k] + alpha * delta_cov_speaker[k]
                cov_Y_boosted = project_psd(cov_Y_boosted, eps=1e-3)
                
                # Compute Bures-Wasserstein Map
                A_k = compute_bures_map_torch(cov_X_k, cov_Y_boosted, eps=1e-2)
                T_k = torch.mm(xc, A_k) + cl_mu_Y[k].unsqueeze(0)
                
                x_hat += wk * T_k
                
            x_np = x_hat.cpu().numpy()
            _, rjit = compute_trajectory_jitter(x_np, xs)
            rel_jitters.append(rjit)
            
            wc = voc(x_np)
            cers.append(jiwer.cer(ref, tr(wc)))
            sims.append(torch.dot(prof, emb(wc)).item())
            
        mean_cer = float(np.mean(cers) * 100)
        mean_sim = float(np.mean(sims))
        mean_jit = float(np.mean(rel_jitters))
        
        records.append({
            'Speaker': spk,
            'Alpha': alpha,
            'CER': mean_cer,
            'Sim': mean_sim,
            'Jitter': mean_jit
        })
        print(f"  alpha = {alpha:4.1f} | CER: {mean_cer:5.2f}% | Sim: {mean_sim:.4f} | Jitter: {mean_jit:.3f}")

df = pd.DataFrame(records)
out_csv = Path('/local_scratch/ssadok/un_projet_audio/output/tables/speaker_boosted_bures_6spk.csv')
df.to_csv(out_csv, index=False)
print(f"\n[DONE] Saved speaker boosted bures results to {out_csv}")
