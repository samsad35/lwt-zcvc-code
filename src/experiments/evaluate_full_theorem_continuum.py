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

print("=== Full Theorem Continuum Evaluation (6 Balanced Speakers) ===")
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
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    Y_norm = torch.nn.functional.normalize(Y_t, dim=1)
    
    prof = torch.stack([emb(w) for w in wavs[:5]]).mean(dim=0)
    prof = torch.nn.functional.normalize(prof, dim=0)
    
    mu_Y, W_Y, C_Y = get_wct(Y, 128)
    
    # Pre-whiten source data
    whitened_src = []
    for xs, ref in src_data:
        mu_X, W_X, _ = get_wct(xs, min(128, len(xs)-1))
        xw = (xs - mu_X) @ W_X
        xs_norm = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-8)
        whitened_src.append((xs, xw, xs_norm, ref))

    # ----------------------------------------------------
    # Step 1: Classic WCT (K=1, global order 1)
    # ----------------------------------------------------
    cers_k1, sims_k1 = [], []
    for xs, xw, _, ref in whitened_src:
        x_hat = xw @ C_Y.T + mu_Y
        wc = voc(x_hat)
        cers_k1.append(jiwer.cer(ref, tr(wc)))
        sims_k1.append(torch.dot(prof, emb(wc)).item())
    records.append({
        'Speaker': spk, 'Stage': 'Spatial K', 'Step_order': 1,
        'Name': 'Classic WCT (K=1)', 'K': 1, 'Beta': 20.0,
        'Sim': np.mean(sims_k1), 'CER': np.mean(cers_k1) * 100
    })
    print(f"  [Classic WCT (K=1)]: Sim = {np.mean(sims_k1):.4f} | CER = {np.mean(cers_k1)*100:.2f}%")

    # ----------------------------------------------------
    # Step 2: Spatial Resolution Increase (K=3, 8, 25, 80 at beta=20)
    # ----------------------------------------------------
    for K in [3, 8, 25, 80]:
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
            
        cers_k, sims_k = [], []
        for xs, xw, xs_norm, ref in whitened_src:
            sim_mat = xs_norm @ cents_norm.T
            logits = sim_mat * 20.0
            logits -= np.max(logits, axis=1, keepdims=True)
            w = np.exp(logits); w /= w.sum(axis=1, keepdims=True)
            
            x_hat = np.zeros_like(xs)
            for k in range(K):
                muk, Ck = cl[k]
                x_hat += w[:, k:k+1] * (xw @ Ck.T + muk)
                
            wc = voc(x_hat)
            cers_k.append(jiwer.cer(ref, tr(wc)))
            sims_k.append(torch.dot(prof, emb(wc)).item())
            
        step_ord = 2 if K==3 else (3 if K==8 else (4 if K==25 else 5))
        records.append({
            'Speaker': spk, 'Stage': 'Spatial K', 'Step_order': step_ord,
            'Name': f'Soft WCT (K={K})', 'K': K, 'Beta': 20.0,
            'Sim': np.mean(sims_k), 'CER': np.mean(cers_k) * 100
        })
        print(f"  [Soft WCT (K={K:2d})]: Sim = {np.mean(sims_k):.4f} | CER = {np.mean(cers_k)*100:.2f}%")

    # ----------------------------------------------------
    # Step 3: Covariance Collapse K=Nt (C -> 0) & Temperature Hardening (beta: 20 -> 40 -> 80 -> inf)
    # ----------------------------------------------------
    for b_idx, beta in enumerate([20.0, 40.0, 80.0, 'inf']):
        cers_b, sims_b = [], []
        for xs, xw, xs_norm, ref in whitened_src:
            xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
            xs_n = torch.nn.functional.normalize(xs_t, dim=1)
            sim_mat = xs_n @ Y_norm.T # [Ns, Nt]
            
            if beta == 'inf':
                idx = torch.argmax(sim_mat, dim=1)
                x_hat = Y_t[idx]
            else:
                w = torch.softmax(sim_mat * beta, dim=1)
                x_hat = w @ Y_t
                
            wc = voc(x_hat)
            cers_b.append(jiwer.cer(ref, tr(wc)))
            sims_b.append(torch.dot(prof, emb(wc)).item())
            
        step_ord = 6 + b_idx
        b_name = "1-NN (kNN-VC)" if beta == 'inf' else f"K=Nt (β={beta:g})"
        records.append({
            'Speaker': spk, 'Stage': 'Temperature Beta', 'Step_order': step_ord,
            'Name': b_name, 'K': 9999, 'Beta': 9999.0 if beta == 'inf' else beta,
            'Sim': np.mean(sims_b), 'CER': np.mean(cers_b) * 100
        })
        print(f"  [{b_name:15s}]: Sim = {np.mean(sims_b):.4f} | CER = {np.mean(cers_b)*100:.2f}%")

df = pd.DataFrame(records)
out_csv = Path('/local_scratch/ssadok/un_projet_audio/output/theorem_continuum_6spk.csv')
out_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)
print(f"\nSaved full results to {out_csv}")

summary = df.groupby(['Step_order', 'Name'], sort=True).agg({
    'Sim': ['mean', 'std'],
    'CER': ['mean', 'std']
})
print("\n" + "="*65)
print("=== THEOREM CONTINUUM SUMMARY ACROSS 6 SPEAKERS ===")
print("="*65)
print(summary)
