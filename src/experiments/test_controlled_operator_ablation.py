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

print("=== Prototyping Controlled Operator Ablation (Identical Routing) ===")
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

def compute_wct_map_torch(cov_X, cov_Y, eps=1e-2):
    D = cov_X.shape[0]
    I_D = torch.eye(D, device=cov_X.device, dtype=cov_X.dtype)
    cX = cov_X + eps * I_D
    cY = cov_Y + eps * I_D
    
    evals_X, evecs_X = torch.linalg.eigh(cX)
    evals_X = evals_X.clamp(min=eps)
    cX_inv_half = evecs_X @ torch.diag(1.0 / torch.sqrt(evals_X)) @ evecs_X.t()
    
    evals_Y, evecs_Y = torch.linalg.eigh(cY)
    evals_Y = evals_Y.clamp(min=eps)
    cY_half = evecs_Y @ torch.diag(torch.sqrt(evals_Y)) @ evecs_Y.t()
    
    M = cX_inv_half @ cY_half
    return M

root = Path('/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean')

# Test on 2 target speakers: 121 (F) and 237 (M)
test_targets = ['121', '237']
src_spks = ['1284', '1320', '1580', '1995']
src_files = [list((root / s).rglob('*.flac'))[0] for s in src_spks]
src_data = [(ext(p)[0], tr(ext(p)[1])) for p in src_files]

K_clusters = 10
beta_val = 20.0
T_target_utts = 20

for tgt_spk in test_targets:
    print(f"\n==================================================")
    print(f"Target Speaker: {tgt_spk}")
    print(f"==================================================")
    files = sorted(list((root / tgt_spk).rglob('*.flac')))[:T_target_utts]
    X_list, wavs = [], []
    for p in files:
        x, w = ext(p); X_list.append(x); wavs.append(w)
    Y = np.concatenate(X_list, axis=0)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    
    prof = torch.stack([emb(w) for w in wavs[:10]]).mean(dim=0)
    prof = torch.nn.functional.normalize(prof, dim=0)
    
    # 1. Target K-Means clustering (K=10)
    km = KMeans(n_clusters=K_clusters, random_state=42, n_init=1).fit(Y)
    cents = km.cluster_centers_
    cents_t = torch.tensor(cents, dtype=torch.float32, device=device)
    cents_norm = torch.nn.functional.normalize(cents_t, dim=1)
    
    # Precompute target cluster covariances, means, and cluster indices
    cl_mu_Y = []
    cl_cov_Y = []
    cl_Y_k = []
    cl_Y_k_norm = []
    for k in range(K_clusters):
        idx_k = np.where(km.labels_ == k)[0]
        Yk = Y[idx_k]
        muk = np.mean(Yk, axis=0)
        covk = np.cov(Yk - muk, rowvar=False) if len(Yk) > 1 else np.zeros((1024, 1024))
        cl_mu_Y.append(torch.tensor(muk, dtype=torch.float32, device=device))
        cl_cov_Y.append(torch.tensor(covk, dtype=torch.float32, device=device))
        Yk_t = torch.tensor(Yk, dtype=torch.float32, device=device)
        cl_Y_k.append(Yk_t)
        cl_Y_k_norm.append(torch.nn.functional.normalize(Yk_t, dim=1))
        
    for op_type in ['WCT', 'LS_Affine', 'Bures_Wasserstein']:
        cers, sims, jitters = [], [], []
        t0 = time.time()
        for xs, ref in src_data:
            xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
            xs_n = torch.nn.functional.normalize(xs_t, dim=1)
            
            # STRICTLY IDENTICAL ROUTING
            sim_mat = xs_n @ cents_norm.t()
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
                
                if op_type == 'WCT':
                    M_k = compute_wct_map_torch(cov_X_k, cl_cov_Y[k], eps=1e-2)
                    T_k = torch.mm(xc, M_k) + cl_mu_Y[k].unsqueeze(0)
                elif op_type == 'Bures_Wasserstein':
                    M_k = compute_bures_map_torch(cov_X_k, cl_cov_Y[k], eps=1e-2)
                    T_k = torch.mm(xc, M_k) + cl_mu_Y[k].unsqueeze(0)
                elif op_type == 'LS_Affine':
                    Yk_norm = cl_Y_k_norm[k]
                    Yk_t = cl_Y_k[k]
                    sub_sim = torch.mm(xs_n, Yk_norm.t())
                    best_match = torch.argmax(sub_sim, dim=1)
                    y_matched = Yk_t[best_match]
                    yc = y_matched - cl_mu_Y[k].unsqueeze(0)
                    
                    lam = 1.0
                    I_D = torch.eye(1024, device=device, dtype=torch.float32)
                    Sigma_xy_k = torch.mm(xc.t(), yc * wk) / mass_k
                    A_mat = cov_X_k + lam * I_D
                    B_mat = Sigma_xy_k + lam * I_D
                    M_k = torch.linalg.solve(A_mat, B_mat)
                    T_k = torch.mm(xc, M_k) + cl_mu_Y[k].unsqueeze(0)
                    
                x_hat += wk * T_k
                
            x_np = x_hat.cpu().numpy()
            _, rjit = compute_trajectory_jitter(x_np, xs)
            jitters.append(rjit)
            wc = voc(x_np)
            cers.append(jiwer.cer(ref, tr(wc)))
            sims.append(torch.dot(prof, emb(wc)).item())
            
        elapsed = (time.time() - t0) / len(src_data) * 1000
        print(f"  Controlled Op: {op_type:18s} | CER: {np.mean(cers)*100:5.2f}% | Sim: {np.mean(sims):.4f} | Jitter: {np.mean(jitters):.3f} | Latency: {elapsed:5.1f}ms")
