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
from src.core.models import get_wct, compute_trajectory_jitter

device = 'cuda:0'

print("================================================================================")
print("=== LAUNCHING UNIFIED BENCHMARK (N=200) WITH TORCHAUDIO-SQUIM (MOS & PESQ)   ===")
print("================================================================================")

# 1. Load Pretrained Models
print("Loading feature extractor, vocoder, and evaluators...")
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

print("Loading Torchaudio-SQUIM models (Subjective MOS & Objective PESQ/STOI)...")
squim_subj = torchaudio.pipelines.SQUIM_SUBJECTIVE.get_model().to(device).eval()
squim_obj = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().to(device).eval()

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

# 10 Target Speakers (5 Female, 5 Male)
target_speakers = [
    '121', '1089', '1221', '3570', '3729', # Female
    '237', '260', '1188', '4446', '4507'   # Male
]

# 10 Disjoint Source Speakers (2 test utterances each = 20 source utterances)
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
    mu_X, W_X, _ = get_wct(feat, min(128, len(feat)-1))
    xw = (feat - mu_X) @ W_X
    feat_norm = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
    src_data.append({
        'path': p,
        'spk': p.parent.parent.name,
        'feat': feat,
        'wav': wav,
        'feat_norm': feat_norm,
        'xw': xw,
        'mu_X': mu_X,
        'W_X': W_X,
        'ref': ref,
        'duration': len(feat) * 0.02
    })

# Background pool for shared phonetics
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

cov_shared_k = []
for k in range(K_clusters):
    idx_k = np.where(km_shared.labels_ == k)[0]
    Yk = bg_Y[idx_k]
    muk = np.mean(Yk, axis=0)
    covk = np.cov(Yk - muk, rowvar=False) if len(Yk) > 1 else np.zeros((1024, 1024))
    cov_shared_k.append(torch.tensor(covk, dtype=torch.float32, device=device))

# Precompute Target Models (T=20 utterances)
target_models = {}
print("Building target representations (T=20 utterances per speaker)...")
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
    
    mu_Y, W_Y, C_Y = get_wct(Y, 128)
    H_Y = (Y - mu_Y) @ W_Y
    H_Y_norm = H_Y / (np.linalg.norm(H_Y, axis=1, keepdims=True) + 1e-8)
    
    km_tgt = KMeans(n_clusters=K_clusters, random_state=42, n_init=1).fit(Y)
    cents_tgt = torch.tensor(km_tgt.cluster_centers_, dtype=torch.float32, device=device)
    cents_tgt_norm = torch.nn.functional.normalize(cents_tgt, dim=1)
    
    cl_wct = []
    for k in range(K_clusters):
        Yk = Y[km_tgt.labels_ == k]
        muk = np.mean(Yk, axis=0)
        covk = np.cov(Yk - muk, rowvar=False) if len(Yk) > 1 else np.zeros((1024, 1024))
        evals, evecs = np.linalg.eigh(covk)
        evals = np.maximum(evals, 1e-5)
        Ck = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
        cl_wct.append((muk, Ck))
        
    sim_Y = Y_norm @ cents_shared_norm.t()
    w_Y = torch.softmax(sim_Y * beta_val, dim=1)
    cl_mu_Y_shared = []
    cl_cov_Y_shared = []
    delta_cov_speaker = []
    for k in range(K_clusters):
        wk = w_Y[:, k:k+1]
        mass_k = wk.sum()
        muk = (Y_t * wk).sum(dim=0) / (mass_k + 1e-8)
        yc = Y_t - muk.unsqueeze(0)
        cov_Y_k = torch.mm(yc.t(), yc * wk) / (mass_k + 1e-8)
        cl_mu_Y_shared.append(muk)
        cl_cov_Y_shared.append(cov_Y_k)
        delta_cov_speaker.append(cov_Y_k - cov_shared_k[k])
        
    target_models[spk] = {
        'Y': Y,
        'Y_t': Y_t,
        'Y_norm': Y_norm,
        'prof': prof,
        'mu_Y': mu_Y,
        'W_Y': W_Y,
        'C_Y': C_Y,
        'H_Y': H_Y,
        'H_Y_norm': H_Y_norm,
        'cents_tgt_norm': cents_tgt_norm,
        'cl_wct': cl_wct,
        'cl_mu_Y_shared': cl_mu_Y_shared,
        'cl_cov_Y_shared': cl_cov_Y_shared,
        'delta_cov_speaker': delta_cov_speaker
    }

print("All target speaker models precomputed.")

methods = [
    ('Classic WCT', 'Global Statistical'),
    ('ICP-WCT (ours)', 'Global Topological'),
    ('kNN-VC (k=1)', 'Local Instance'),
    ('kNN-VC (k=4)', 'Local Instance'),
    ('LinearVC', 'Local Projection'),
    ('Soft Local WCT (Ours)', 'Piecewise Statistical'),
    ('Local Wasserstein (LWT - Ours)', 'Piecewise Monge-Kantorovich'),
    ('Boosted LWT (alpha=1.5 - Ours)', 'Speaker-Covariance Monge-Kantorovich')
]

detailed_records = []

print("\nStarting unified conversion loop with SQUIM MOS & PESQ (8 methods x 200 conversions = 1600 evaluations)...")
for m_idx, (method_name, approach_type) in enumerate(methods):
    t_start_method = time.time()
    print(f"\n[{m_idx+1}/{len(methods)}] Evaluating: {method_name} ({approach_type})...")
    
    count = 0
    for tgt_spk, tm in target_models.items():
        for s_item in src_data:
            xs = s_item['feat']
            xw = s_item['xw']
            ref = s_item['ref']
            duration = s_item['duration']
            Ns = len(xs)
            
            xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
            xs_n = torch.nn.functional.normalize(xs_t, dim=1)
            
            t0 = time.perf_counter()
            
            if method_name == 'Classic WCT':
                x_hat_np = xw @ tm['C_Y'].T + tm['mu_Y']
                
            elif method_name == 'ICP-WCT (ours)':
                R = np.eye(1024, dtype=np.float32)
                for _ in range(3):
                    H_src_rot = xw @ R
                    H_rot_norm = H_src_rot / (np.linalg.norm(H_src_rot, axis=1, keepdims=True) + 1e-8)
                    sim_mat = H_rot_norm @ tm['H_Y_norm'].T
                    best_idx = np.argmax(sim_mat, axis=1)
                    H_Y_matched = tm['H_Y'][best_idx]
                    S = xw.T @ H_Y_matched
                    U, _, Vt = np.linalg.svd(S)
                    R = (U @ Vt).astype(np.float32)
                x_hat_np = (xw @ R) @ tm['C_Y'].T + tm['mu_Y']
                
            elif method_name == 'kNN-VC (k=1)':
                sim_mat = torch.mm(xs_n, tm['Y_norm'].t())
                best_idx = torch.argmax(sim_mat, dim=1)
                x_hat = tm['Y_t'][best_idx]
                x_hat_np = x_hat.cpu().numpy()
                
            elif method_name == 'kNN-VC (k=4)':
                sim_mat = torch.mm(xs_n, tm['Y_norm'].t())
                top4_idx = torch.topk(sim_mat, k=4, dim=1).indices
                x_hat = torch.mean(tm['Y_t'][top4_idx], dim=1)
                x_hat_np = x_hat.cpu().numpy()
                
            elif method_name == 'LinearVC':
                sim_mat = torch.mm(xs_n, tm['Y_norm'].t())
                best_idx = torch.argmax(sim_mat, dim=1)
                Y_matched = tm['Y_t'][best_idx].cpu().numpy()
                X_bias = np.hstack([xs, np.ones((Ns, 1), dtype=np.float32)])
                W_lin, _, _, _ = np.linalg.lstsq(X_bias, Y_matched, rcond=None)
                x_hat_np = X_bias @ W_lin
                
            elif method_name == 'Soft Local WCT (Ours)':
                sim_mat = xs_n @ tm['cents_tgt_norm'].t()
                w_dyn = torch.softmax(sim_mat * beta_val, dim=1).cpu().numpy()
                x_hat_np = np.zeros_like(xs)
                for k in range(K_clusters):
                    muk, Ck = tm['cl_wct'][k]
                    x_hat_np += w_dyn[:, k:k+1] * (xw @ Ck.T + muk)
                    
            elif method_name == 'Local Wasserstein (LWT - Ours)':
                sim_mat = xs_n @ cents_shared_norm.t()
                w_dyn = torch.softmax(sim_mat * beta_val, dim=1)
                x_hat = torch.zeros_like(xs_t)
                for k in range(K_clusters):
                    wk = w_dyn[:, k:k+1]
                    mass_k = wk.sum()
                    if mass_k < 1e-4: continue
                    mu_X_k = (xs_t * wk).sum(dim=0) / mass_k
                    xc = xs_t - mu_X_k.unsqueeze(0)
                    cov_X_k = torch.mm(xc.t(), xc * wk) / mass_k
                    A_k = compute_bures_map_torch(cov_X_k, tm['cl_cov_Y_shared'][k], eps=1e-2)
                    T_k = torch.mm(xc, A_k) + tm['cl_mu_Y_shared'][k].unsqueeze(0)
                    x_hat += wk * T_k
                x_hat_np = x_hat.cpu().numpy()
                
            elif method_name == 'Boosted LWT (alpha=1.5 - Ours)':
                sim_mat = xs_n @ cents_shared_norm.t()
                w_dyn = torch.softmax(sim_mat * beta_val, dim=1)
                x_hat = torch.zeros_like(xs_t)
                for k in range(K_clusters):
                    wk = w_dyn[:, k:k+1]
                    mass_k = wk.sum()
                    if mass_k < 1e-4: continue
                    mu_X_k = (xs_t * wk).sum(dim=0) / mass_k
                    xc = xs_t - mu_X_k.unsqueeze(0)
                    cov_X_k = torch.mm(xc.t(), xc * wk) / mass_k
                    cov_boosted = cov_shared_k[k] + 1.5 * tm['delta_cov_speaker'][k]
                    cov_boosted = project_psd(cov_boosted, eps=1e-3)
                    A_k = compute_bures_map_torch(cov_X_k, cov_boosted, eps=1e-2)
                    T_k = torch.mm(xc, A_k) + tm['cl_mu_Y_shared'][k].unsqueeze(0)
                    x_hat += wk * T_k
                x_hat_np = x_hat.cpu().numpy()
                
            t1 = time.perf_counter()
            rtf = (t1 - t0) / duration
            
            # Synthesize Audio Waveform
            _, rjit = compute_trajectory_jitter(x_hat_np, xs)
            wc = voc(x_hat_np)
            hyp = tr(wc)
            cer = jiwer.cer(ref, hyp) * 100.0
            wer = jiwer.wer(ref, hyp) * 100.0
            sim = torch.dot(tm['prof'], emb(wc)).item()
            
            # SQUIM Subjective MOS & Objective PESQ / STOI
            wc_2d = wc.view(1, -1).to(device)
            w_ref_2d = s_item['wav'].view(1, -1).to(device)
            with torch.no_grad():
                mos = squim_subj(wc_2d, w_ref_2d).item()
                stoi, pesq, _ = squim_obj(wc_2d)
                pesq_val = pesq.item()
                stoi_val = stoi.item()
            
            detailed_records.append({
                'Method': method_name,
                'Approach_Type': approach_type,
                'Target_Speaker': tgt_spk,
                'Source_Speaker': s_item['spk'],
                'CER': cer,
                'WER': wer,
                'Sim': sim,
                'MOS': mos,
                'PESQ': pesq_val,
                'STOI': stoi_val,
                'Jitter': rjit,
                'RTF': rtf
            })
            
            count += 1
            if count % 50 == 0:
                print(f"  Processed {count}/200 conversions...")
                
    print(f"  -> Finished {method_name} in {time.time()-t_start_method:.1f}s")

# Save detailed conversions CSV
out_dir = Path('/local_scratch/ssadok/un_projet_audio/output/tables')
out_dir.mkdir(parents=True, exist_ok=True)
df_detail = pd.DataFrame(detailed_records)
df_detail.to_csv(out_dir / 'unified_200_conversions_detail.csv', index=False)
print(f"\nSaved detailed conversions to {out_dir / 'unified_200_conversions_detail.csv'}")

# Compute summary table
summary = df_detail.groupby(['Method', 'Approach_Type']).agg({
    'CER': ['mean', 'std'],
    'WER': ['mean', 'std'],
    'Sim': ['mean', 'std'],
    'MOS': ['mean', 'std'],
    'PESQ': ['mean', 'std'],
    'STOI': ['mean', 'std'],
    'Jitter': ['mean', 'std'],
    'RTF': ['mean', 'std']
}).reset_index()

summary.columns = [
    'Method', 'Approach_Type',
    'CER_mean', 'CER_std',
    'WER_mean', 'WER_std',
    'Sim_mean', 'Sim_std',
    'MOS_mean', 'MOS_std',
    'PESQ_mean', 'PESQ_std',
    'STOI_mean', 'STOI_std',
    'Jitter_mean', 'Jitter_std',
    'RTF_mean', 'RTF_std'
]

order = [
    'Classic WCT',
    'ICP-WCT (ours)',
    'kNN-VC (k=1)',
    'kNN-VC (k=4)',
    'LinearVC',
    'Soft Local WCT (Ours)',
    'Local Wasserstein (LWT - Ours)',
    'Boosted LWT (alpha=1.5 - Ours)'
]

summary['Method'] = pd.Categorical(summary['Method'], categories=order, ordered=True)
summary = summary.sort_values('Method').reset_index(drop=True)

out_summary = out_dir / 'unified_200_benchmark_summary.csv'
summary.to_csv(out_summary, index=False)
print(f"Saved benchmark summary to {out_summary}")

print("\n================================================================================")
print("=== FINAL UNIFIED BENCHMARK SUMMARY INCLUDING SQUIM MOS (N=200 PER METHOD)   ===")
print("================================================================================")
print(summary[['Method', 'CER_mean', 'WER_mean', 'Sim_mean', 'MOS_mean', 'PESQ_mean', 'Jitter_mean', 'RTF_mean']].to_string())

# Regenerate LaTeX Table with MOS
tex_path = out_dir / 'tab_overall_unified_200.tex'
with open(tex_path, 'w') as f:
    f.write("\\begin{table*}[t]\n")
    f.write("    \\centering\n")
    f.write("    \\caption{Comprehensive performance comparison across voice conversion paradigms evaluated on a strictly unified benchmark of $N=200$ conversions (10 target speakers, 20 source utterances from 10 distinct speakers, $T=20$ target utterances). Quality is evaluated using ASR error rates (CER, WER), speaker cosine similarity, perceptual speech quality (Torchaudio-SQUIM MOS), trajectory smoothness (Rel. Jitter), and computational efficiency (RTF).}\n")
    f.write("    \\label{tab:overall_unified_200}\n")
    f.write("    \\begin{tabular}{llcccccc}\n")
    f.write("        \\toprule\n")
    f.write("        \\textbf{Method} & \\textbf{Approach Type} & \\textbf{CER (\\%)} $\\downarrow$ & \\textbf{WER (\\%)} $\\downarrow$ & \\textbf{Speaker Sim} $\\uparrow$ & \\textbf{MOS} $\\uparrow$ & \\textbf{Rel. Jitter} $\\downarrow$ & \\textbf{RTF} $\\downarrow$ \\\\\n")
    f.write("        \\midrule\n")
    for _, row in summary.iterrows():
        m = row['Method']
        app = row['Approach_Type']
        cer = f"{row['CER_mean']:.2f} $\\pm$ {row['CER_std']:.2f}"
        wer = f"{row['WER_mean']:.2f} $\\pm$ {row['WER_std']:.2f}"
        sim = f"{row['Sim_mean']:.3f} $\\pm$ {row['Sim_std']:.3f}"
        mos = f"{row['MOS_mean']:.2f} $\\pm$ {row['MOS_std']:.2f}"
        jit = f"{row['Jitter_mean']:.2f}$\\times$"
        rtf = f"{row['RTF_mean']:.4f}"
        
        if 'Ours' in m:
            f.write(f"        \\textbf{{{m}}} & {app} & \\textbf{{{cer}}} & \\textbf{{{wer}}} & \\textbf{{{sim}}} & \\textbf{{{mos}}} & {jit} & \\textbf{{{rtf}}} \\\\\n")
        else:
            f.write(f"        {m} & {app} & {cer} & {wer} & {sim} & {mos} & {jit} & {rtf} \\\\\n")
    f.write("        \\bottomrule\n")
    f.write("    \\end{tabular}\n")
    f.write("\\end{table*}\n")

print(f"Saved updated LaTeX table to {tex_path}")
