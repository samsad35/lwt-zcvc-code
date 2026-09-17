import os, sys, torch, torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
import jiwer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.inference.speaker import EncoderClassifier
import torchaudio.functional as F

device = 'cuda:0'

print("=== 2D Continuum Grid Evaluation (K x Beta on 6 Balanced Speakers) ===")
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
        return hifigan(torch.tensor(f, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).cpu()

def get_wct(X, k=128):
    mu = np.mean(X, axis=0)
    cov = np.cov(X - mu, rowvar=False)
    e_val, e_vec = np.linalg.eigh(cov)
    idx = np.argsort(e_val)[::-1]
    e_val, e_vec = e_val[idx], e_vec[:, idx]
    k = min(k, len(e_val))
    e_val = np.maximum(e_val[:k], 1e-5)
    e_vec = e_vec[:, :k]
    W = e_vec @ np.diag(1.0 / np.sqrt(e_val)) @ e_vec.T
    C = e_vec @ np.diag(np.sqrt(e_val)) @ e_vec.T
    return mu, W, C

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

# Beta values requested by the user: 2, 20, 100, 1000
beta_list = [2.0, 20.0, 100.0, 1000.0]
# K intermediate clusters
K_intermediate = [3, 8, 25, 80, 250]

records = []

for spk in target_speakers:
    print(f"\n=======================================================")
    print(f"--> Target Speaker: {spk}")
    print(f"=======================================================")
    files = sorted(list((root / spk).rglob('*.flac')))[:15]
    X_list, wavs = [], []
    for p in files:
        x, w = ext(p); X_list.append(x); wavs.append(w)
    Y = np.concatenate(X_list, axis=0) # [Nt, 1024]
    
    prof = torch.stack([emb(w) for w in wavs[:5]]).mean(dim=0)
    prof = torch.nn.functional.normalize(prof, dim=0)
    
    mu_Y, W_Y, C_Y = get_wct(Y, 128)
    
    # Pre-whiten all source data for this speaker
    whitened_src = []
    for xs, ref in src_data:
        mu_X, W_X, _ = get_wct(xs, min(128, len(xs)-1))
        xw = (xs - mu_X) @ W_X
        xs_norm = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-8)
        whitened_src.append((xs, xw, xs_norm, ref))

    # 1. Classic WCT (K=1, independent of beta)
    cers_1, sims_1 = [], []
    for xs, xw, _, ref in whitened_src:
        x_hat = xw @ C_Y.T + mu_Y
        wc = voc(x_hat)
        cers_1.append(jiwer.cer(ref, tr(wc)))
        sims_1.append(torch.dot(prof, emb(wc)).item())
    mean_sim_1 = np.mean(sims_1)
    mean_cer_1 = np.mean(cers_1) * 100
    print(f"  [Classic WCT (K=1)]: Sim = {mean_sim_1:.4f} | CER = {mean_cer_1:.2f}%")
    
    for b in beta_list:
        records.append({
            'Speaker': spk, 'Beta': b, 'K': 1, 'Method': 'Classic WCT (K=1)',
            'Sim': mean_sim_1, 'CER': mean_cer_1
        })

    # 2. Intermediate K values for each Beta
    for K in K_intermediate:
        km = KMeans(n_clusters=K, random_state=42, n_init=1).fit(Y)
        cents = km.cluster_centers_
        cents_norm = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-8)
        
        cl = []
        for k in range(K):
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

        # Evaluate across the 4 beta values
        for b in beta_list:
            cers_kb, sims_kb = [], []
            for xs, xw, xs_norm, ref in whitened_src:
                sim_mat = xs_norm @ cents_norm.T # [Ns, K]
                logits = sim_mat * b
                logits -= np.max(logits, axis=1, keepdims=True)
                w = np.exp(logits)
                w /= w.sum(axis=1, keepdims=True)
                
                x_hat = np.zeros_like(xs)
                for k in range(K):
                    muk, Ck = cl[k]
                    x_hat += w[:, k:k+1] * (xw @ Ck.T + muk)
                    
                wc = voc(x_hat)
                cers_kb.append(jiwer.cer(ref, tr(wc)))
                sims_kb.append(torch.dot(prof, emb(wc)).item())
                
            m_sim = np.mean(sims_kb)
            m_cer = np.mean(cers_kb) * 100
            records.append({
                'Speaker': spk, 'Beta': b, 'K': K, 'Method': f"Soft WCT (K={K})",
                'Sim': m_sim, 'CER': m_cer
            })
        print(f"  [K={K:3d}] Done for all betas (Sim at beta=20: {m_sim:.4f}, CER: {m_cer:.2f}%)")

    # 3. 1-NN (K=Nt, beta -> inf, independent of beta)
    Y_t = torch.tensor(Y).float().to(device)
    tgt_norm = torch.nn.functional.normalize(Y_t, dim=1)
    cers_1nn, sims_1nn = [], []
    for xs, xw, xs_norm, ref in whitened_src:
        xs_t = torch.tensor(xs).float().to(device)
        xs_n = torch.nn.functional.normalize(xs_t, dim=1)
        sim_mat = xs_n @ tgt_norm.T
        idx = torch.argmax(sim_mat, dim=1)
        x_hat = Y_t[idx].cpu().numpy()
        
        wc = voc(x_hat)
        cers_1nn.append(jiwer.cer(ref, tr(wc)))
        sims_1nn.append(torch.dot(prof, emb(wc)).item())
        
    mean_sim_1nn = np.mean(sims_1nn)
    mean_cer_1nn = np.mean(cers_1nn) * 100
    print(f"  [1-NN (K=Nt)]: Sim = {mean_sim_1nn:.4f} | CER = {mean_cer_1nn:.2f}%")
    
    for b in beta_list:
        records.append({
            'Speaker': spk, 'Beta': b, 'K': 9999, 'Method': '1-NN (K=Nt)',
            'Sim': mean_sim_1nn, 'CER': mean_cer_1nn
        })

df = pd.DataFrame(records)
out_csv = Path('/local_scratch/ssadok/un_projet_audio/continuum_2d_grid_6spk.csv')
df.to_csv(out_csv, index=False)
print(f"\nSaved full 2D grid results to {out_csv}")

summary = df.groupby(['Beta', 'K'], sort=False).mean(numeric_only=True)
print("\n" + "="*60)
print("=== 2D GRID SUMMARY ACROSS 6 SPEAKERS (Mean Sim & CER) ===")
print("="*60)
print(summary)
