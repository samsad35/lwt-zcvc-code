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

print("=== TESTING LOCAL ORTHOGONAL PROCRUSTES VARIANTS ===")

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

# Test on 2 target speakers (1F: 121, 1M: 237) and 4 source utterances
test_targets = ['121', '237']
test_src_files = [
    list((root / '1284').rglob('*.flac'))[0],
    list((root / '1320').rglob('*.flac'))[0],
    list((root / '1580').rglob('*.flac'))[0],
    list((root / '1995').rglob('*.flac'))[0],
]

src_data = []
for p in test_src_files:
    feat, wav = ext(p)
    ref = tr(wav)
    src_data.append({'feat': feat, 'ref': ref, 'duration': len(feat)*0.02})

# Precompute background shared clusters
bg_spks = ['4970', '4992', '5142', '5639', '5683', '61']
bg_feats = [ext(sorted(list((root / s).rglob('*.flac')))[0])[0] for s in bg_spks]
bg_Y = np.concatenate(bg_feats, axis=0)

K_clusters = 10
beta_val = 20.0
km_shared = KMeans(n_clusters=K_clusters, random_state=42, n_init=1).fit(bg_Y)
cents_sh = torch.tensor(km_shared.cluster_centers_, dtype=torch.float32, device=device)
cents_sh_norm = torch.nn.functional.normalize(cents_sh, dim=1)

target_data = {}
for spk in test_targets:
    files = sorted(list((root / spk).rglob('*.flac')))[:20]
    X_list, wavs = [], []
    for p in files:
        x, w = ext(p); X_list.append(x); wavs.append(w)
    Y = np.concatenate(X_list, axis=0)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    Y_norm = torch.nn.functional.normalize(Y_t, dim=1)
    prof = torch.stack([emb(w) for w in wavs[:10]]).mean(dim=0)
    prof = torch.nn.functional.normalize(prof, dim=0)
    
    sim_Y = Y_norm @ cents_sh_norm.t()
    w_Y = torch.softmax(sim_Y * beta_val, dim=1)
    
    mu_list, cov_list = [], []
    for k in range(K_clusters):
        wk = w_Y[:, k:k+1]
        mass_k = wk.sum()
        muk = (Y_t * wk).sum(dim=0) / (mass_k + 1e-8)
        yc = Y_t - muk.unsqueeze(0)
        covk = torch.mm(yc.t(), yc * wk) / (mass_k + 1e-8)
        mu_list.append(muk)
        cov_list.append(covk)
        
    target_data[spk] = {
        'Y_t': Y_t,
        'Y_norm': Y_norm,
        'prof': prof,
        'mu_list': mu_list,
        'cov_list': cov_list
    }

# --------------------------------------------------------------------------
# Method 1: Local Statistical Procrustes (from Covariances SVD)
# min_{R^T R = I} || R Sigma_X^{1/2} - Sigma_Y^{1/2} ||_F^2
# S = Sigma_X^{1/2} Sigma_Y^{1/2} -> U S V^T -> R = U V^T
# --------------------------------------------------------------------------
def eval_stat_procrustes():
    cers, sims, jits = [], [], []
    for spk in test_targets:
        td = target_data[spk]
        for s in src_data:
            xs = s['feat']
            xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
            xs_n = torch.nn.functional.normalize(xs_t, dim=1)
            sim_xs = xs_n @ cents_sh_norm.t()
            w_dyn = torch.softmax(sim_xs * beta_val, dim=1)
            
            x_hat = torch.zeros_like(xs_t)
            for k in range(K_clusters):
                wk = w_dyn[:, k:k+1]
                mass_k = wk.sum()
                if mass_k < 1e-4: continue
                mu_X_k = (xs_t * wk).sum(dim=0) / mass_k
                xc = xs_t - mu_X_k.unsqueeze(0)
                cov_X_k = torch.mm(xc.t(), xc * wk) / mass_k
                
                # Sigma_X^{1/2} and Sigma_Y^{1/2}
                evals_X, evecs_X = torch.linalg.eigh(cov_X_k + 1e-3 * torch.eye(1024, device=device))
                evals_X = evals_X.clamp(min=1e-3)
                cX_half = evecs_X @ torch.diag(torch.sqrt(evals_X)) @ evecs_X.t()
                
                evals_Y, evecs_Y = torch.linalg.eigh(td['cov_list'][k] + 1e-3 * torch.eye(1024, device=device))
                evals_Y = evals_Y.clamp(min=1e-3)
                cY_half = evecs_Y @ torch.diag(torch.sqrt(evals_Y)) @ evecs_Y.t()
                
                # Cross-covariance matrix M = cX_half @ cY_half
                M = cX_half @ cY_half
                U, S, Vh = torch.linalg.svd(M)
                R_k = U @ Vh  # R_k is strictly orthogonal: R_k^T R_k = I
                
                T_k = torch.mm(xc, R_k) + td['mu_list'][k].unsqueeze(0)
                x_hat += wk * T_k
                
            x_np = x_hat.cpu().numpy()
            _, rjit = compute_trajectory_jitter(x_np, xs)
            wc = voc(x_np)
            hyp = tr(wc)
            cer = jiwer.cer(s['ref'], hyp) * 100.0
            sim = torch.dot(td['prof'], emb(wc)).item()
            cers.append(cer); sims.append(sim); jits.append(rjit)
    return np.mean(cers), np.mean(sims), np.mean(jits)

# --------------------------------------------------------------------------
# Method 2: Local Matched Instance Procrustes (from 1-NN matches per cluster)
# For cluster k: X_k matched to Y_k_matched -> S_k = X_k^T Y_k_matched -> R_k = U V^T
# --------------------------------------------------------------------------
def eval_matched_procrustes():
    cers, sims, jits = [], [], []
    for spk in test_targets:
        td = target_data[spk]
        for s in src_data:
            xs = s['feat']
            xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
            xs_n = torch.nn.functional.normalize(xs_t, dim=1)
            sim_xs = xs_n @ cents_sh_norm.t()
            w_dyn = torch.softmax(sim_xs * beta_val, dim=1)
            
            # Find nearest neighbors in target for source frames
            sim_nn = torch.mm(xs_n, td['Y_norm'].t())
            best_idx = torch.argmax(sim_nn, dim=1)
            Y_matched = td['Y_t'][best_idx]
            
            x_hat = torch.zeros_like(xs_t)
            for k in range(K_clusters):
                wk = w_dyn[:, k:k+1]
                mass_k = wk.sum()
                if mass_k < 1e-4: continue
                mu_X_k = (xs_t * wk).sum(dim=0) / mass_k
                xc = xs_t - mu_X_k.unsqueeze(0)
                
                # Target matched centered
                mu_Y_k = td['mu_list'][k]
                yc = Y_matched - mu_Y_k.unsqueeze(0)
                
                # S_k = xc^T (yc * wk)
                S_k = torch.mm(xc.t(), yc * wk)
                U, _, Vh = torch.linalg.svd(S_k)
                R_k = U @ Vh
                
                T_k = torch.mm(xc, R_k) + mu_Y_k.unsqueeze(0)
                x_hat += wk * T_k
                
            x_np = x_hat.cpu().numpy()
            _, rjit = compute_trajectory_jitter(x_np, xs)
            wc = voc(x_np)
            hyp = tr(wc)
            cer = jiwer.cer(s['ref'], hyp) * 100.0
            sim = torch.dot(td['prof'], emb(wc)).item()
            cers.append(cer); sims.append(sim); jits.append(rjit)
    return np.mean(cers), np.mean(sims), np.mean(jits)

print("\nEvaluating Method 1: Local Statistical Procrustes (Covariance Isometry)...")
c1, s1, j1 = eval_stat_procrustes()
print(f"  Method 1 -> CER: {c1:.2f}% | Sim: {s1:.4f} | Jitter: {j1:.3f}")

print("\nEvaluating Method 2: Local Matched Instance Procrustes (Point Isometry)...")
c2, s2, j2 = eval_matched_procrustes()
print(f"  Method 2 -> CER: {c2:.2f}% | Sim: {s2:.4f} | Jitter: {j2:.3f}")
