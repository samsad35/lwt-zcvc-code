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

print("=== Temperature Beta Ablation Study (Fixed K=25, 6 Speakers) ===")
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

target_speakers = ['121', '237', '1221', '260', '1089', '1188']
src_spks = ['1284', '1320', '1580', '1995', '2300', '2830']
src_files = [list((root / s).rglob('*.flac'))[0] for s in src_spks]

print(f"Pre-extracting {len(src_files)} source test utterances...")
src_data = [(ext(p)[0], tr(ext(p)[1])) for p in src_files]

# Fixed K = 25 (Optimal Sweet Spot from Spatial Experiment)
FIXED_K = 25
# Varying Beta: from near-uniform (beta=2) to hard argmax (beta -> inf)
beta_values = [2.0, 5.0, 10.0, 20.0, 40.0, 100.0, "inf"]
records = []

for spk in target_speakers:
    print(f"\n=======================================================")
    print(f"--> Target Speaker: {spk} (Fixed K={FIXED_K})")
    print(f"=======================================================")
    files = sorted(list((root / spk).rglob('*.flac')))[:15]
    X_list, wavs = [], []
    for p in files:
        x, w = ext(p); X_list.append(x); wavs.append(w)
    Y = np.concatenate(X_list, axis=0)
    
    prof = torch.stack([emb(w) for w in wavs[:5]]).mean(dim=0)
    prof = torch.nn.functional.normalize(prof, dim=0)
    
    # Train K-Means with K = FIXED_K
    km = KMeans(n_clusters=FIXED_K, random_state=42, n_init=1).fit(Y)
    cents = km.cluster_centers_
    cents_norm = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-8)
    
    cl = []
    for k in range(FIXED_K):
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
        
    for beta in beta_values:
        cers, sims, entropies = [], [], []
        
        for xs, ref in src_data:
            mu_X, W_X, _ = get_wct(xs, min(128, len(xs)-1))
            xw = (xs - mu_X) @ W_X
            xs_norm = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-8)
            sim_mat = xs_norm @ cents_norm.T # [Ns, K]
            
            if beta == "inf":
                # Hard Argmax assignment (Dirac distribution, Entropy = 0)
                best_k = np.argmax(sim_mat, axis=1) # [Ns]
                x_hat = np.zeros_like(xs)
                for i in range(len(xs)):
                    k_idx = best_k[i]
                    muk, Ck = cl[k_idx]
                    x_hat[i] = xw[i] @ Ck.T + muk
                mean_entropy = 0.0
            else:
                # Softmax assignment
                logits = sim_mat * beta
                logits -= np.max(logits, axis=1, keepdims=True)
                w = np.exp(logits)
                w /= np.sum(w, axis=1, keepdims=True) # [Ns, K]
                
                # Shannon Entropy H(w) = - sum(w * log(w))
                w_clipped = np.clip(w, 1e-12, 1.0)
                entropy_per_frame = -np.sum(w * np.log(w_clipped), axis=1)
                mean_entropy = np.mean(entropy_per_frame)
                
                x_hat = np.zeros_like(xs)
                for k in range(FIXED_K):
                    muk, Ck = cl[k]
                    x_hat += w[:, k:k+1] * (xw @ Ck.T + muk)
                    
            wc = voc(x_hat)
            cers.append(jiwer.cer(ref, tr(wc)))
            sims.append(torch.dot(prof, emb(wc)).item())
            entropies.append(mean_entropy)
            
        b_label = "Hard (β→∞)" if beta == "inf" else f"β={beta:g}"
        records.append({
            'Speaker': spk,
            'Beta_raw': 9999.0 if beta == "inf" else float(beta),
            'Beta_label': b_label,
            'Entropy': np.mean(entropies),
            'Sim': np.mean(sims),
            'CER': np.mean(cers) * 100
        })
        print(f"  [{b_label:12s}] Sim: {np.mean(sims):.4f} | CER: {np.mean(cers)*100:.2f}% | Entropy: {np.mean(entropies):.3f}")

df = pd.DataFrame(records)
out_csv = Path('/local_scratch/ssadok/un_projet_audio/beta_ablation_6spk.csv')
df.to_csv(out_csv, index=False)
print(f"\nSaved beta ablation results to {out_csv}")

summary = df.groupby('Beta_label', sort=False).agg({
    'Entropy': ['mean', 'std'],
    'Sim': ['mean', 'std'],
    'CER': ['mean', 'std']
})
print("\n" + "="*65)
print("=== TEMPERATURE BETA ABLATION SUMMARY (6 SPEAKERS) ===")
print("="*65)
print(summary)
