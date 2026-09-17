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

print("=== Quantitative Evaluation: ||J_geom||_F vs ||J_routing||_F vs Beta ===")
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

K_clusters = 25
# Dense sweep of beta values to map the transition curve
beta_values = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0]

records = []

for spk_idx, spk in enumerate(target_speakers):
    print(f"\n[{spk_idx+1}/{len(target_speakers)}] Target Speaker: {spk}")
    files = sorted(list((root / spk).rglob('*.flac')))[:15]
    X_list, wavs = [], []
    for p in files:
        x, w = ext(p); X_list.append(x); wavs.append(w)
    Y = np.concatenate(X_list, axis=0)
    
    prof = torch.stack([emb(w) for w in wavs[:5]]).mean(dim=0)
    prof = torch.nn.functional.normalize(prof, dim=0)
    
    # 1. Fit K-Means for local components
    km = KMeans(n_clusters=K_clusters, random_state=42, n_init=1).fit(Y)
    cents = km.cluster_centers_ # [K, D]
    cents_norm = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-8)
    
    # Precompute local coloring matrices C_k and means mu_k
    cl_mu = []
    cl_C = []
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
        cl_mu.append(muk)
        cl_C.append(Ck)
        
    cl_mu = np.stack(cl_mu, axis=0) # [K, D]
    cl_C = np.stack(cl_C, axis=0)   # [K, D, D]

    # Pre-process source data
    whitened_src = []
    for xs, ref in src_data:
        mu_X, W_X, _ = get_wct(xs, min(128, len(xs)-1))
        xw = (xs - mu_X) @ W_X
        xs_norm = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-8)
        
        # Precompute M_k = C_{Y,k} W_X and Q_{kj} = Tr(M_k^T M_j)
        # Note: xw @ Ck.T means in row notation, (x_w W_X) Ck^T
        # In matrix product: xw is 1xD, Ck is DxD.
        # f_k(x_w) = xw @ Ck.T + muk (row vector)
        # M_k = W_X @ Ck.T (so that x @ M_k = x @ W_X @ Ck.T)
        M = np.zeros((K_clusters, 1024, 1024), dtype=np.float32)
        for k in range(K_clusters):
            M[k] = W_X @ cl_C[k].T
            
        Q = np.zeros((K_clusters, K_clusters), dtype=np.float32)
        for k in range(K_clusters):
            for j in range(k, K_clusters):
                val = np.sum(M[k] * M[j])
                Q[k, j] = val
                Q[j, k] = val
                
        whitened_src.append((xs, xw, xs_norm, ref, Q, M))

    # Evaluate across beta values
    for beta in beta_values:
        j_geom_list = []
        j_routing_list = []
        j_ratio_list = []
        rel_jitter_list = []
        entropy_list = []
        cers, sims = [], []
        
        for xs, xw, xs_norm, ref, Q, M in whitened_src:
            Ns = len(xs)
            
            # 1. Cosine similarity and dynamic weights
            sim_mat = xs_norm @ cents_norm.T # [Ns, K]
            logits = sim_mat * beta
            logits -= np.max(logits, axis=1, keepdims=True)
            w_dynamic = np.exp(logits)
            w_dynamic /= w_dynamic.sum(axis=1, keepdims=True) # [Ns, K]
            
            # Gating entropy
            ent = -np.sum(w_dynamic * np.log(w_dynamic + 1e-12), axis=1) # [Ns]
            entropy_list.extend(ent)
            
            # 2. Local affine mapped points f_k(x_w)
            # f_k: [Ns, K, D]
            # xw: [Ns, D], cl_C: [K, D, D], cl_mu: [K, D]
            F_mat = np.zeros((Ns, K_clusters, 1024), dtype=np.float32)
            for k in range(K_clusters):
                F_mat[:, k, :] = xw @ cl_C[k].T + cl_mu[k]
                
            # Converted trajectory: x_hat = sum_k w_k f_k
            x_hat = np.sum(w_dynamic[:, :, np.newaxis] * F_mat, axis=1) # [Ns, D]
            
            # Trajectory jitter
            jit, rel_jit = compute_trajectory_jitter(x_hat, xs)
            rel_jitter_list.append(rel_jit)
            
            # 3. Compute ||J_geom||_F for each frame:
            # ||J_geom(x_t)||_F^2 = w(x_t)^T Q w(x_t)
            # Vectorized over all frames: (w @ Q * w).sum(axis=1)
            w_Q_w = np.sum((w_dynamic @ Q) * w_dynamic, axis=1)
            norm_j_geom = np.sqrt(np.maximum(w_Q_w, 1e-10)) # [Ns]
            j_geom_list.extend(norm_j_geom)
            
            # 4. Compute ||J_routing||_F for each frame:
            # Spatial gradient \nabla_x w_k:
            # p_k(x) = c_norm[k] - (x_norm . c_norm[k]) x_norm
            # P: [Ns, K, D]
            P = cents_norm[np.newaxis, :, :] - sim_mat[:, :, np.newaxis] * xs_norm[:, np.newaxis, :]
            # bar_p: [Ns, D] = sum_k w_k p_k
            bar_p = np.sum(w_dynamic[:, :, np.newaxis] * P, axis=1)
            # G: [Ns, K, D] = beta / ||x|| * w_k * (p_k - bar_p)
            xs_len = np.linalg.norm(xs, axis=1, keepdims=True)[:, :, np.newaxis] # [Ns, 1, 1]
            G_mat = (beta / (xs_len + 1e-8)) * w_dynamic[:, :, np.newaxis] * (P - bar_p[:, np.newaxis, :])
            
            # Fast Frobenius norm of J_routing:
            # ||J_routing||_F^2 = sum_{k, j} (F F^T)_{kj} (G G^T)_{kj}
            # For each frame t: F_t is [K, D], G_t is [K, D]
            # F_gram = F_t @ F_t^T [K, K], G_gram = G_t @ G_t^T [K, K]
            # Vectorized batch Gram computation:
            # F_gram: [Ns, K, K] = torch.bmm or einsum
            F_t = torch.tensor(F_mat, dtype=torch.float32, device=device)
            G_t = torch.tensor(G_mat, dtype=torch.float32, device=device)
            F_gram = torch.bmm(F_t, F_t.transpose(1, 2)) # [Ns, K, K]
            G_gram = torch.bmm(G_t, G_t.transpose(1, 2)) # [Ns, K, K]
            norm_j_routing_sq = torch.sum(F_gram * G_gram, dim=(1, 2)).clamp(min=0.0) # [Ns]
            norm_j_routing = torch.sqrt(norm_j_routing_sq).cpu().numpy()
            j_routing_list.extend(norm_j_routing)
            
            # Ratio per frame
            ratios = norm_j_routing / (norm_j_geom + 1e-8)
            j_ratio_list.extend(ratios)
            
            # Audio synthesis & evaluation
            wc = voc(x_hat)
            cers.append(jiwer.cer(ref, tr(wc)))
            sims.append(torch.dot(prof, emb(wc)).item())

        mean_j_geom = float(np.mean(j_geom_list))
        std_j_geom = float(np.std(j_geom_list))
        mean_j_routing = float(np.mean(j_routing_list))
        std_j_routing = float(np.std(j_routing_list))
        mean_ratio = float(np.mean(j_ratio_list))
        mean_rel_jit = float(np.mean(rel_jitter_list))
        mean_entropy = float(np.mean(entropy_list))
        mean_cer = float(np.mean(cers) * 100)
        mean_sim = float(np.mean(sims))
        
        records.append({
            'Speaker': spk,
            'Beta': beta,
            'J_geom_mean': mean_j_geom,
            'J_geom_std': std_j_geom,
            'J_routing_mean': mean_j_routing,
            'J_routing_std': std_j_routing,
            'J_ratio_mean': mean_ratio,
            'Relative_Jitter': mean_rel_jit,
            'Entropy': mean_entropy,
            'CER': mean_cer,
            'Sim': mean_sim
        })
        print(f"  Beta: {beta:5.1f} | ||J_geom||: {mean_j_geom:6.2f} | ||J_rout||: {mean_j_routing:6.2f} | Ratio: {mean_ratio:5.2f} | RelJit: {mean_rel_jit:.3f} | CER: {mean_cer:4.2f}% | Sim: {mean_sim:.3f}")

df = pd.DataFrame(records)
out_csv = Path('/local_scratch/ssadok/un_projet_audio/output/tables/jacobian_norms_vs_beta_6spk.csv')
out_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)
print(f"\nSaved complete results to {out_csv}")

# Grouped summary across all 6 speakers
summary = df.groupby('Beta').agg({
    'J_geom_mean': ['mean', 'std'],
    'J_routing_mean': ['mean', 'std'],
    'J_ratio_mean': ['mean', 'std'],
    'Relative_Jitter': ['mean', 'std'],
    'Entropy': ['mean', 'std'],
    'CER': ['mean', 'std'],
    'Sim': ['mean', 'std']
})
print("\n" + "="*90)
print("=== JACOBIAN NORMS VS BETA SUMMARY ACROSS 6 SPEAKERS ===")
print("="*90)
print(summary)

# Also save grouped summary
summary_flat = df.groupby('Beta').mean().reset_index()
summary_flat.to_csv('/local_scratch/ssadok/un_projet_audio/output/tables/jacobian_norms_vs_beta_summary.csv', index=False)
print("\nSaved summary to output/tables/jacobian_norms_vs_beta_summary.csv")
