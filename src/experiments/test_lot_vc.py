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

print("=== Prototyping Local Affine Optimal Transport VC ===")
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

def sinkhorn_log(cost, reg=0.05, max_iter=60):
    Ns, Nt = cost.shape
    log_a = torch.full((Ns,), -np.log(Ns), device=cost.device, dtype=cost.dtype)
    log_b = torch.full((Nt,), -np.log(Nt), device=cost.device, dtype=cost.dtype)
    u = torch.zeros(Ns, device=cost.device, dtype=cost.dtype)
    v = torch.zeros(Nt, device=cost.device, dtype=cost.dtype)
    K_mat = -cost / reg
    for _ in range(max_iter):
        u = log_a - torch.logsumexp(K_mat + v.unsqueeze(0), dim=1)
        v = log_b - torch.logsumexp(K_mat + u.unsqueeze(1), dim=0)
    log_P = K_mat + u.unsqueeze(1) + v.unsqueeze(0)
    return torch.exp(log_P)

root = Path('/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean')

# Test on 2 target speakers: 121 (F) and 237 (M)
test_targets = ['121', '237']
src_spks = ['1284', '1320', '1580', '1995']
src_files = [list((root / s).rglob('*.flac'))[0] for s in src_spks]
src_data = [(ext(p)[0], tr(ext(p)[1])) for p in src_files]

for tgt_spk in test_targets:
    print(f"\n==================================================")
    print(f"Target Speaker: {tgt_spk}")
    print(f"==================================================")
    files = sorted(list((root / tgt_spk).rglob('*.flac')))[:15]
    X_list, wavs = [], []
    for p in files:
        x, w = ext(p); X_list.append(x); wavs.append(w)
    Y = np.concatenate(X_list, axis=0) # [Nt, 1024]
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    Y_norm = torch.nn.functional.normalize(Y_t, dim=1)
    
    prof = torch.stack([emb(w) for w in wavs[:5]]).mean(dim=0)
    prof = torch.nn.functional.normalize(prof, dim=0)
    
    # 1. Target K-Means clustering (K=10)
    K = 10
    km = KMeans(n_clusters=K, random_state=42, n_init=1).fit(Y)
    y_labels = torch.tensor(km.labels_, dtype=torch.long, device=device)
    
    # Baseline: Soft Local WCT for reference
    cers_wct, sims_wct, jit_wct = [], [], []
    for xs, ref in src_data:
        mu_X, W_X, _ = get_wct(xs, min(128, len(xs)-1))
        xw = (xs - mu_X) @ W_X
        xs_norm = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-8)
        cents = km.cluster_centers_
        cents_norm = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-8)
        sim_mat = xs_norm @ cents_norm.T
        w_dyn = np.exp(sim_mat * 20.0)
        w_dyn /= w_dyn.sum(axis=1, keepdims=True)
        x_hat = np.zeros_like(xs)
        for k in range(K):
            Yk = Y[km.labels_ == k]
            muk = np.mean(Yk, axis=0)
            _, _, Ck = get_wct(Yk, min(128, len(Yk)-1)) if len(Yk) > 128 else (None, None, np.eye(1024))
            x_hat += w_dyn[:, k:k+1] * (xw @ Ck.T + muk)
        _, rjit = compute_trajectory_jitter(x_hat, xs)
        jit_wct.append(rjit)
        wc = voc(x_hat)
        cers_wct.append(jiwer.cer(ref, tr(wc)))
        sims_wct.append(torch.dot(prof, emb(wc)).item())
    print(f"Reference Soft Local WCT (K={K}) | CER: {np.mean(cers_wct)*100:.2f}% | Sim: {np.mean(sims_wct):.4f} | Jitter: {np.mean(jit_wct):.3f}")

    # Now test Local Affine Optimal Transport across different lambda regularizations
    for reg_sinkhorn in [0.05]:
        for lam in [0.01, 0.1, 1.0, 5.0]:
            cers, sims, jitters = [], [], []
            t0 = time.time()
            for xs, ref in src_data:
                Ns = len(xs)
                xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
                xs_n = torch.nn.functional.normalize(xs_t, dim=1)
                
                # 1. Cosine Cost matrix
                cost = 1.0 - torch.mm(xs_n, Y_norm.t()) # [Ns, Nt]
                
                # 2. Sinkhorn Transport Plan
                P = sinkhorn_log(cost, reg=reg_sinkhorn, max_iter=50) # [Ns, Nt]
                
                # 3. For each cluster k, compute local affine map
                # alpha_{i, k} = sum_{j in cluster k} P_{ij}
                x_hat = torch.zeros_like(xs_t)
                
                # Cluster membership matrix [Nt, K]
                one_hot_k = torch.nn.functional.one_hot(y_labels, num_classes=K).float() # [Nt, K]
                # Alpha: [Ns, K] = P @ one_hot_k
                Alpha = torch.mm(P, one_hot_k) # [Ns, K]
                
                # Routing weights w_k(x_i) = Alpha / sum_k Alpha
                W_routing = Alpha / (Alpha.sum(dim=1, keepdim=True) + 1e-8) # [Ns, K]
                
                # Target points mapped per cluster:
                # Y_matched_k[i] = (sum_{j in k} P_{ij} y_j) / Alpha[i, k]
                # P_k = P * one_hot_k[:, k]
                for k in range(K):
                    P_k = P * one_hot_k[:, k].unsqueeze(0) # [Ns, Nt]
                    mass_k = P_k.sum()
                    if mass_k < 1e-7:
                        continue
                    
                    # Marginal weights on source
                    w_src_k = P_k.sum(dim=1) # [Ns]
                    w_src_k_norm = w_src_k / (w_src_k.sum() + 1e-8)
                    
                    # Marginal weights on target
                    w_tgt_k = P_k.sum(dim=0) # [Nt]
                    w_tgt_k_norm = w_tgt_k / (w_tgt_k.sum() + 1e-8)
                    
                    # Barycenters
                    mu_x_k = torch.sum(xs_t * w_src_k_norm.unsqueeze(1), dim=0) # [D]
                    mu_y_k = torch.sum(Y_t * w_tgt_k_norm.unsqueeze(1), dim=0)   # [D]
                    
                    # Centered
                    xc = xs_t - mu_x_k.unsqueeze(0) # [Ns, D]
                    
                    # Target conditional destination for each source frame i:
                    # y_hat_i = sum_j P_k[i, j] y_j / (w_src_k[i] + 1e-8)
                    denom = w_src_k.clamp(min=1e-8).unsqueeze(1)
                    y_hat_k = torch.mm(P_k, Y_t) / denom # [Ns, D]
                    yc = y_hat_k - mu_y_k.unsqueeze(0)   # [Ns, D]
                    
                    # Weighted covariance matrices:
                    # Sigma_xx = sum_i w_src_k_norm[i] * xc_i xc_i^T = xc^T diag(w) xc
                    Sigma_xx = torch.mm(xc.t(), xc * w_src_k_norm.unsqueeze(1)) # [D, D]
                    Sigma_xy = torch.mm(xc.t(), yc * w_src_k_norm.unsqueeze(1)) # [D, D]
                    
                    # Regularized local affine operator M_k = (Sigma_xx + lam * I)^-1 (Sigma_xy + lam * I)
                    I_D = torch.eye(1024, device=device, dtype=torch.float32)
                    A_mat = Sigma_xx + lam * I_D
                    B_mat = Sigma_xy + lam * I_D
                    
                    M_k = torch.linalg.solve(A_mat, B_mat) # [D, D]
                    
                    # Local affine prediction: f_k(x) = (x - mu_x_k) @ M_k + mu_y_k
                    f_k = torch.mm(xc, M_k) + mu_y_k.unsqueeze(0) # [Ns, D]
                    
                    # Accumulate into x_hat using smooth routing W_routing
                    x_hat = x_hat + W_routing[:, k:k+1] * f_k
                    
                x_hat_np = x_hat.cpu().numpy()
                _, rjit = compute_trajectory_jitter(x_hat_np, xs)
                jitters.append(rjit)
                wc = voc(x_hat_np)
                cers.append(jiwer.cer(ref, tr(wc)))
                sims.append(torch.dot(prof, emb(wc)).item())
                
            elapsed = time.time() - t0
            print(f"  LOT-VC (lam={lam:5.2f}, eps={reg_sinkhorn}) | CER: {np.mean(cers)*100:5.2f}% | Sim: {np.mean(sims):.4f} | Jitter: {np.mean(jitters):.3f} | Latency: {elapsed/len(src_data)*1000:.1f}ms")
