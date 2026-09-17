import os, sys, time, torch, torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
import jiwer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.inference.speaker import EncoderClassifier
import torchaudio.functional as F

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.core.models import get_wct, compute_trajectory_jitter

device = 'cuda:0'

print("=== Overall Performance Benchmark against Fundamental Baselines (T=20, 6 Speakers) ===")
print("Loading models on device:", device)

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
    with torch.no_grad():
        feat, _ = wavlm.extract_features(w, output_layer=6)
    return feat.squeeze(0).cpu().numpy(), w.squeeze().cpu()

def voc(f):
    with torch.inference_mode():
        if isinstance(f, np.ndarray):
            f = torch.tensor(f, dtype=torch.float32, device=device)
        elif f.device != torch.device(device):
            f = f.to(device)
        return hifigan(f.unsqueeze(0)).squeeze(0).cpu()

def tr(w):
    inp = processor(w.squeeze().numpy(), sampling_rate=16000, return_tensors='pt', padding=True).to(device)
    with torch.no_grad(): log = asr(inp.input_values).logits
    return processor.batch_decode(torch.argmax(log, dim=-1))[0]

def emb(w):
    with torch.no_grad():
        return torch.nn.functional.normalize(spk_model.encode_batch(w.squeeze().float().unsqueeze(0).to(device)).squeeze().cpu(), dim=0)

root = Path('/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean')

# 6 balanced target speakers (3F, 3M)
target_speakers = ['121', '237', '1221', '260', '1089', '1188']
# 6 diverse source test sentences
src_spks = ['1284', '1320', '1580', '1995', '2300', '2830']
src_files = [list((root / s).rglob('*.flac'))[0] for s in src_spks]

print(f"Pre-extracting {len(src_files)} source test utterances...")
src_data = [(ext(p)[0], tr(ext(p)[1])) for p in src_files]

T_target_utts = 20
K_clusters = 25
beta_val = 20.0

methods = [
    ('Classic WCT', 'Global Statistical'),
    ('ICP-WCT (ours)', 'Global Topological'),
    ('kNN-VC', 'Local Instance'),
    ('LinearVC', 'Local Projection'),
    ('Soft Local WCT (Ours)', 'Piecewise Statistical')
]

records = []

for spk_idx, spk in enumerate(target_speakers):
    print(f"\n=======================================================")
    print(f"[{spk_idx+1}/{len(target_speakers)}] Target Speaker: {spk} (T={T_target_utts} utterances)")
    print(f"=======================================================")
    
    files = sorted(list((root / spk).rglob('*.flac')))[:T_target_utts]
    X_list, wavs = [], []
    for p in files:
        x, w = ext(p); X_list.append(x); wavs.append(w)
    Y = np.concatenate(X_list, axis=0) # [Nt, 1024]
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    
    prof = torch.stack([emb(w) for w in wavs[:10]]).mean(dim=0)
    prof = torch.nn.functional.normalize(prof, dim=0)
    
    # Precompute Global WCT parameters
    mu_Y, W_Y, C_Y = get_wct(Y, 128)
    
    # Pre-whiten target pool for ICP-WCT in 128D PCA subspace
    # Note: in WCT, projection into 128D: (Y - mu_Y) @ W_Y is [Nt, 1024]
    H_Y = (Y - mu_Y) @ W_Y
    H_Y_norm = H_Y / (np.linalg.norm(H_Y, axis=1, keepdims=True) + 1e-8)
    
    # Fit K-Means for Soft Local WCT
    km = KMeans(n_clusters=K_clusters, random_state=42, n_init=1).fit(Y)
    cents = km.cluster_centers_
    cents_norm = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-8)
    
    cl = []
    for k in range(K_clusters):
        Yk = Y[km.labels_ == k]
        muk = np.mean(Yk, axis=0)
        if len(Yk) > 128:
            _, _, Ck = get_wct(Yk, 128)
        else:
            cov_k = np.cov(Yk - muk, rowvar=False) if len(Yk) > 1 else np.zeros((1024, 1024))
            evals, evecs = np.linalg.eigh(cov_k)
            evals = np.maximum(evals, 1e-5)
            Ck = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
        cl.append((muk, Ck))

    # Pre-whiten source utterances
    whitened_src = []
    for xs, ref in src_data:
        mu_X, W_X, _ = get_wct(xs, min(128, len(xs)-1))
        xw = (xs - mu_X) @ W_X
        xs_norm = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-8)
        whitened_src.append((xs, xw, xs_norm, ref, mu_X, W_X))

    for method_name, approach_type in methods:
        cers, sims, rel_jitters, rtfs = [], [], [], []
        
        for xs, xw, xs_norm, ref, mu_X, W_X in whitened_src:
            Ns = len(xs)
            audio_duration = Ns * 0.02 # 50 fps = 20ms per frame
            
            # Measure Latency of conversion step
            t0 = time.perf_counter()
            
            if method_name == 'Classic WCT':
                # Global WCT
                x_hat = xw @ C_Y.T + mu_Y
                
            elif method_name == 'ICP-WCT (ours)':
                # Iterative Closest Point on whitened representations (3 iterations)
                R = np.eye(1024, dtype=np.float32)
                for _ in range(3):
                    H_src_rot = xw @ R
                    H_rot_norm = H_src_rot / (np.linalg.norm(H_src_rot, axis=1, keepdims=True) + 1e-8)
                    # Cosine matching in whitened space
                    sim_mat = H_rot_norm @ H_Y_norm.T
                    best_idx = np.argmax(sim_mat, axis=1)
                    H_Y_matched = H_Y[best_idx]
                    
                    # Orthogonal Procrustes
                    S = xw.T @ H_Y_matched
                    U, _, Vt = np.linalg.svd(S)
                    R = (U @ Vt).astype(np.float32)
                    
                H_final = xw @ R
                x_hat = H_final @ C_Y.T + mu_Y
                
            elif method_name == 'kNN-VC':
                # Standard kNN-VC with k=4 nearest neighbors
                sim_mat = xs_norm @ Y_norm.T
                top4_idx = np.argpartition(-sim_mat, 4, axis=1)[:, :4]
                # Average matched frames
                x_hat = np.mean(Y[top4_idx], axis=1)
                
            elif method_name == 'LinearVC':
                # LinearVC: 1-NN matching followed by dynamic least-squares solver
                sim_mat = xs_norm @ Y_norm.T
                best_idx = np.argmax(sim_mat, axis=1)
                Y_matched = Y[best_idx]
                
                # Fit linear regression [xs, 1] W = Y_matched
                X_bias = np.hstack([xs, np.ones((Ns, 1), dtype=np.float32)])
                W_lin, _, _, _ = np.linalg.lstsq(X_bias, Y_matched, rcond=None)
                x_hat = X_bias @ W_lin
                
            elif method_name == 'Soft Local WCT (Ours)':
                # Soft Local WCT with dynamic gating
                sim_mat = xs_norm @ cents_norm.T
                logits = sim_mat * beta_val
                logits -= np.max(logits, axis=1, keepdims=True)
                w_dyn = np.exp(logits)
                w_dyn /= w_dyn.sum(axis=1, keepdims=True)
                
                x_hat = np.zeros_like(xs)
                for k in range(K_clusters):
                    muk, Ck = cl[k]
                    x_hat += w_dyn[:, k:k+1] * (xw @ Ck.T + muk)
                    
            t1 = time.perf_counter()
            rtf = (t1 - t0) / audio_duration
            rtfs.append(rtf)
            
            # Trajectory Jitter
            jit, rel_jit = compute_trajectory_jitter(x_hat, xs)
            rel_jitters.append(rel_jit)
            
            # Synthesize & evaluate
            wc = voc(x_hat)
            cers.append(jiwer.cer(ref, tr(wc)))
            sims.append(torch.dot(prof, emb(wc)).item())
            
        mean_cer = float(np.mean(cers) * 100)
        mean_sim = float(np.mean(sims))
        mean_rtf = float(np.mean(rtfs))
        mean_rel_jit = float(np.mean(rel_jitters))
        
        records.append({
            'Speaker': spk,
            'Method': method_name,
            'Approach_Type': approach_type,
            'CER': mean_cer,
            'Sim': mean_sim,
            'RTF': mean_rtf,
            'Relative_Jitter': mean_rel_jit
        })
        print(f"  {method_name:25s} | CER: {mean_cer:5.2f}% | Sim: {mean_sim:.4f} | RTF: {mean_rtf:.5f} | RelJit: {mean_rel_jit:.3f}")

df = pd.DataFrame(records)
out_csv = Path('/local_scratch/ssadok/un_projet_audio/output/tables/overall_baselines_6spk.csv')
out_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)
print(f"\nSaved raw records to {out_csv}")

# Grouped summary across all 6 speakers
summary = df.groupby(['Method', 'Approach_Type'], sort=False).agg({
    'CER': ['mean', 'std'],
    'Sim': ['mean', 'std'],
    'RTF': ['mean', 'std'],
    'Relative_Jitter': ['mean', 'std']
})
print("\n" + "="*95)
print("=== OVERALL BASELINES BENCHMARK ACROSS 6 SPEAKERS (T=20) ===")
print("="*95)
print(summary)

# Flat summary
flat_summary = df.groupby(['Method', 'Approach_Type'], sort=False).mean(numeric_only=True).reset_index()
flat_summary.to_csv('/local_scratch/ssadok/un_projet_audio/output/tables/overall_baselines_summary.csv', index=False)
print("\nSaved flat summary to /local_scratch/ssadok/un_projet_audio/output/tables/overall_baselines_summary.csv")
