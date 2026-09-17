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

print("=== Prototyping Speaker-Covariance-Boosted Bures-Wasserstein Map ===")
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

test_targets = ['121', '237']
src_spks = ['1284', '1320', '1580', '1995']
src_files = [list((root / s).rglob('*.flac'))[0] for s in src_spks]
src_data = [(ext(p)[0], tr(ext(p)[1])) for p in src_files]

K_clusters = 10
beta_val = 20.0
T_target_utts = 20

# 1. First, build a Population Shared Phonetic Covariance Sigma_{shared, k}
# from 4 background speakers in test-clean: '1284', '1320', '1580', '1995'
print("Building Population Shared Phonetic Covariance from background speech...")
bg_spks = ['260', '1089', '1188', '1221']
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

print("Shared phonetic covariances computed for K=10.")

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

    # Test different values of alpha
    # alpha = 1.0 is standard LWT with shared clustering
    # alpha > 1.0 amplifies speaker-specific directions
    alphas = [0.5, 1.0, 1.2, 1.5, 2.0]
    
    for alpha in alphas:
        cers, sims, jitters = [], [], []
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
                # Sigma_{Y, k}(alpha) = Sigma_{shared, k} + alpha * Delta_Sigma_{speaker, k}
                # Projected onto PSD cone to ensure valid Riemannian transport
                cov_Y_boosted = cov_shared_k[k] + alpha * delta_cov_speaker[k]
                cov_Y_boosted = project_psd(cov_Y_boosted, eps=1e-3)
                
                # Compute Bures-Wasserstein Map
                A_k = compute_bures_map_torch(cov_X_k, cov_Y_boosted, eps=1e-2)
                T_k = torch.mm(xc, A_k) + cl_mu_Y[k].unsqueeze(0)
                
                x_hat += wk * T_k
                
            x_np = x_hat.cpu().numpy()
            _, rjit = compute_trajectory_jitter(x_np, xs)
            jitters.append(rjit)
            wc = voc(x_np)
            cers.append(jiwer.cer(ref, tr(wc)))
            sims.append(torch.dot(prof, emb(wc)).item())
            
        print(f"  Alpha = {alpha:4.1f} | CER: {np.mean(cers)*100:5.2f}% | Sim: {np.mean(sims):.4f} | Jitter: {np.mean(jitters):.3f}")
