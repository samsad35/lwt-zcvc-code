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

print("=== Testing Local Bures-Wasserstein Optimal Transport VC ===")
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

def compute_bures_map_torch(cov_X, cov_Y, eps=1e-4):
    """
    Computes symmetric Bures-Wasserstein Monge map A = Sigma_X^-1/2 (Sigma_X^1/2 Sigma_Y Sigma_X^1/2)^1/2 Sigma_X^-1/2
    using PyTorch on GPU.
    """
    D = cov_X.shape[0]
    I_D = torch.eye(D, device=cov_X.device, dtype=cov_X.dtype)
    
    # Regularize
    cX = cov_X + eps * I_D
    cY = cov_Y + eps * I_D
    
    # EVD of Sigma_X
    evals_X, evecs_X = torch.linalg.eigh(cX)
    evals_X = evals_X.clamp(min=eps)
    cX_half = evecs_X @ torch.diag(torch.sqrt(evals_X)) @ evecs_X.t()
    cX_inv_half = evecs_X @ torch.diag(1.0 / torch.sqrt(evals_X)) @ evecs_X.t()
    
    # M = cX_half @ cY @ cX_half
    M = cX_half @ cY @ cX_half
    evals_M, evecs_M = torch.linalg.eigh(M)
    evals_M = evals_M.clamp(min=eps**2)
    M_half = evecs_M @ torch.diag(torch.sqrt(evals_M)) @ evecs_M.t()
    
    # A = cX_inv_half @ M_half @ cX_inv_half
    A = cX_inv_half @ M_half @ cX_inv_half
    return A

root = Path('/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean')

test_targets = ['121', '237']
src_spks = ['1284', '1320', '1580', '1995']
src_files = [list((root / s).rglob('*.flac'))[0] for s in src_spks]
src_data = [(ext(p)[0], tr(ext(p)[1])) for p in src_files]

K_clusters = 10
beta_val = 20.0

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
    
    prof = torch.stack([emb(w) for w in wavs[:5]]).mean(dim=0)
    prof = torch.nn.functional.normalize(prof, dim=0)
    
    # 1. Fit Target K-Means
    km = KMeans(n_clusters=K_clusters, random_state=42, n_init=1).fit(Y)
    cents = km.cluster_centers_
    cents_t = torch.tensor(cents, dtype=torch.float32, device=device)
    cents_norm = torch.nn.functional.normalize(cents_t, dim=1)
    
    # Precompute target cluster covariances and means
    cl_mu_Y = []
    cl_cov_Y = []
    cl_C_wct = []
    for k in range(K_clusters):
        Yk = Y[km.labels_ == k]
        muk = np.mean(Yk, axis=0)
        covk = np.cov(Yk - muk, rowvar=False) if len(Yk) > 1 else np.zeros((1024, 1024))
        _, _, Ck = get_wct(Yk, min(128, len(Yk)-1)) if len(Yk) > 128 else (None, None, np.eye(1024))
        cl_mu_Y.append(torch.tensor(muk, dtype=torch.float32, device=device))
        cl_cov_Y.append(torch.tensor(covk, dtype=torch.float32, device=device))
        cl_C_wct.append(torch.tensor(Ck, dtype=torch.float32, device=device))
        
    # --- Benchmark 1: Soft Local WCT (Reference) ---
    cers_wct, sims_wct, jit_wct = [], [], []
    t0 = time.time()
    for xs, ref in src_data:
        mu_X, W_X, _ = get_wct(xs, min(128, len(xs)-1))
        xw = (xs - mu_X) @ W_X
        xw_t = torch.tensor(xw, dtype=torch.float32, device=device)
        xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
        xs_n = torch.nn.functional.normalize(xs_t, dim=1)
        sim_mat = xs_n @ cents_norm.t()
        w_dyn = torch.softmax(sim_mat * beta_val, dim=1)
        x_hat = torch.zeros_like(xs_t)
        for k in range(K_clusters):
            x_hat += w_dyn[:, k:k+1] * (torch.mm(xw_t, cl_C_wct[k].t()) + cl_mu_Y[k].unsqueeze(0))
        x_np = x_hat.cpu().numpy()
        _, rjit = compute_trajectory_jitter(x_np, xs)
        jit_wct.append(rjit)
        wc = voc(x_np)
        cers_wct.append(jiwer.cer(ref, tr(wc)))
        sims_wct.append(torch.dot(prof, emb(wc)).item())
    lat_wct = (time.time() - t0) / len(src_data) * 1000
    print(f"  Soft Local WCT (Reference)       | CER: {np.mean(cers_wct)*100:5.2f}% | Sim: {np.mean(sims_wct):.4f} | Jitter: {np.mean(jit_wct):.3f} | Latency: {lat_wct:5.1f}ms")

    # --- Benchmark 2: Local Bures-Wasserstein Transport (Global Source Cov -> Local Target Cov) ---
    # In this setting, A_k = Bures(Sigma_X, Sigma_{Y,k}), T_k(x) = mu_{Y,k} + (x - mu_X) @ A_k
    cers_bures_glob, sims_bures_glob, jit_bures_glob = [], [], []
    t0 = time.time()
    for xs, ref in src_data:
        xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
        xs_n = torch.nn.functional.normalize(xs_t, dim=1)
        mu_X_t = xs_t.mean(dim=0)
        cov_X_t = torch.cov((xs_t - mu_X_t.unsqueeze(0)).t())
        
        sim_mat = xs_n @ cents_norm.t()
        w_dyn = torch.softmax(sim_mat * beta_val, dim=1)
        
        x_hat = torch.zeros_like(xs_t)
        for k in range(K_clusters):
            # Compute exact Bures map A_k
            A_k = compute_bures_map_torch(cov_X_t, cl_cov_Y[k], eps=1e-3)
            # T_k(x) = (x - mu_X) @ A_k + mu_{Y,k}
            T_k = torch.mm(xs_t - mu_X_t.unsqueeze(0), A_k) + cl_mu_Y[k].unsqueeze(0)
            x_hat += w_dyn[:, k:k+1] * T_k
            
        x_np = x_hat.cpu().numpy()
        _, rjit = compute_trajectory_jitter(x_np, xs)
        jit_bures_glob.append(rjit)
        wc = voc(x_np)
        cers_bures_glob.append(jiwer.cer(ref, tr(wc)))
        sims_bures_glob.append(torch.dot(prof, emb(wc)).item())
    lat_bg = (time.time() - t0) / len(src_data) * 1000
    print(f"  Local Bures-Wasserstein (Glob X) | CER: {np.mean(cers_bures_glob)*100:5.2f}% | Sim: {np.mean(sims_bures_glob):.4f} | Jitter: {np.mean(jit_bures_glob):.3f} | Latency: {lat_bg:5.1f}ms")

    # --- Benchmark 3: Pure Local Bures-Wasserstein (Local Source Cov_k -> Local Target Cov_k) ---
    # For each cluster k, Sigma_{X,k} = sum_i w_k(x_i) (x_i - mu_{X,k})(x_i - mu_{X,k})^T / sum w_k
    cers_bures_loc, sims_bures_loc, jit_bures_loc = [], [], []
    t0 = time.time()
    for xs, ref in src_data:
        xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
        xs_n = torch.nn.functional.normalize(xs_t, dim=1)
        sim_mat = xs_n @ cents_norm.t()
        w_dyn = torch.softmax(sim_mat * beta_val, dim=1) # [Ns, K]
        
        x_hat = torch.zeros_like(xs_t)
        for k in range(K_clusters):
            wk = w_dyn[:, k:k+1] # [Ns, 1]
            mass_k = wk.sum()
            if mass_k < 1e-4:
                continue
            mu_X_k = (xs_t * wk).sum(dim=0) / mass_k # [D]
            xc = xs_t - mu_X_k.unsqueeze(0)
            cov_X_k = torch.mm(xc.t(), xc * wk) / mass_k # [D, D]
            
            # Compute exact Bures map A_k
            A_k = compute_bures_map_torch(cov_X_k, cl_cov_Y[k], eps=1e-2)
            T_k = torch.mm(xc, A_k) + cl_mu_Y[k].unsqueeze(0)
            x_hat += wk * T_k
            
        x_np = x_hat.cpu().numpy()
        _, rjit = compute_trajectory_jitter(x_np, xs)
        jit_bures_loc.append(rjit)
        wc = voc(x_np)
        cers_bures_loc.append(jiwer.cer(ref, tr(wc)))
        sims_bures_loc.append(torch.dot(prof, emb(wc)).item())
    lat_bl = (time.time() - t0) / len(src_data) * 1000
    print(f"  Local Bures-Wasserstein (Loc X)  | CER: {np.mean(cers_bures_loc)*100:5.2f}% | Sim: {np.mean(sims_bures_loc):.4f} | Jitter: {np.mean(jit_bures_loc):.3f} | Latency: {lat_bl:5.1f}ms")
