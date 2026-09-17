import os, sys, torch, torchaudio
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import jiwer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.inference.speaker import EncoderClassifier
import torchaudio.functional as F

device = 'cuda:0'

print("Loading models...")
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

# Test on Speaker 121
tgt_spk = '121'
print(f"\n--- Testing Exact Theoretical Continuum on Target: {tgt_spk} ---")
tgt_files = sorted(list((root / tgt_spk).rglob('*.flac')))[:20]
X_tgt_list, tgt_wavs = [], []
for p in tgt_files:
    x, w = ext(p); X_tgt_list.append(x); tgt_wavs.append(w)
Y = np.concatenate(X_tgt_list, axis=0) # [Nt, 1024]
tgt_prof = torch.stack([emb(w) for w in tgt_wavs[:10]]).mean(dim=0)
tgt_prof = torch.nn.functional.normalize(tgt_prof, dim=0)

# Global WCT matrices of target
mu_Y, W_Y, C_Y = get_wct(Y, 128)
# Target frames in whitened space
Y_w = (Y - mu_Y) @ W_Y

# Source sentences (5 diverse sentences)
src_files = [list((root / s).rglob('*.flac'))[0] for s in ['1320', '2300', '2830', '1284', '1995']]
src_data = [(ext(p)[0], tr(ext(p)[1])) for p in src_files]

# Test K across the full continuum
K_values = [1, 5, 20, 80, 300, "1-NN", "4-NN"]

for K in K_values:
    cers, sims = [], []
    
    if K == 1:
        # K = 1: Classic WCT
        for X_src, ref in src_data:
            mu_X, W_X, _ = get_wct(X_src, min(128, len(X_src)-1))
            X_w = (X_src - mu_X) @ W_X
            X_hat = X_w @ C_Y.T + mu_Y
            w_c = voc(X_hat)
            cers.append(jiwer.cer(ref, tr(w_c)))
            sims.append(torch.dot(tgt_prof, emb(w_c)).item())
        print(f"K = 1 (Classic WCT)     : Sim = {np.mean(sims):.4f} | CER = {np.mean(cers)*100:.2f}%")

    elif isinstance(K, int):
        # K in [5, 20, 80, 300]: K-Means in target space Y
        km = KMeans(n_clusters=K, random_state=42, n_init=1).fit(Y)
        centroids = km.cluster_centers_ # [K, 1024]
        
        # Local cluster statistics in original space Y
        cluster_info = []
        for k in range(K):
            Y_k = Y[km.labels_ == k]
            mu_k = np.mean(Y_k, axis=0)
            if len(Y_k) > 128:
                _, _, C_k = get_wct(Y_k, 128)
            else:
                cov_k = np.cov(Y_k - mu_k, rowvar=False)
                evals, evecs = np.linalg.eigh(cov_k)
                evals = np.maximum(evals, 1e-5)
                C_k = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
            cluster_info.append((mu_k, C_k))
            
        beta = 20.0
        # Centroids in original WavLM space
        centroids = km.cluster_centers_
        centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
        
        for X_src, ref in src_data:
            mu_X, W_X, _ = get_wct(X_src, min(128, len(X_src)-1))
            X_w = (X_src - mu_X) @ W_X # [Ns, 1024]
            
            # Cosine similarity to centroids in shared WavLM space
            X_src_norm = X_src / (np.linalg.norm(X_src, axis=1, keepdims=True) + 1e-8)
            sims_mat = X_src_norm @ centroids_norm.T # [Ns, K]
            logits = beta * sims_mat
            logits -= np.max(logits, axis=1, keepdims=True)
            w = np.exp(logits)
            w = w / w.sum(axis=1, keepdims=True) # [Ns, K]
            
            X_hat = np.zeros_like(X_src)
            for k in range(K):
                mu_k, C_k = cluster_info[k]
                X_hat += w[:, k:k+1] * (X_w @ C_k.T + mu_k)
                
            w_c = voc(X_hat)
            cers.append(jiwer.cer(ref, tr(w_c)))
            sims.append(torch.dot(tgt_prof, emb(w_c)).item())
        print(f"K = {K:2d} (Soft Local WCT) : Sim = {np.mean(sims):.4f} | CER = {np.mean(cers)*100:.2f}%")

    elif K == "1-NN":
        # 1-NN: K = Nt, beta -> inf, C_Y -> 0, mu_Y -> y_j
        Y_t = torch.tensor(Y).float().to(device)
        tgt_norm = torch.nn.functional.normalize(Y_t, dim=1)
        for X_src, ref in src_data:
            X_src_t = torch.tensor(X_src).float().to(device)
            src_norm = torch.nn.functional.normalize(X_src_t, dim=1)
            sim_mat = src_norm @ tgt_norm.T # [Ns, Nt]
            nearest_idx = torch.argmax(sim_mat, dim=1) # [Ns]
            X_hat = Y_t[nearest_idx].cpu().numpy()
            
            w_c = voc(X_hat)
            cers.append(jiwer.cer(ref, tr(w_c)))
            sims.append(torch.dot(tgt_prof, emb(w_c)).item())
        print(f"1-NN (K = Nt, beta -> inf): Sim = {np.mean(sims):.4f} | CER = {np.mean(cers)*100:.2f}%")

    elif K == "4-NN":
        # 4-NN: kNN-VC
        Y_t = torch.tensor(Y).float().to(device)
        tgt_norm = torch.nn.functional.normalize(Y_t, dim=1)
        for X_src, ref in src_data:
            X_src_t = torch.tensor(X_src).float().to(device)
            src_norm = torch.nn.functional.normalize(X_src_t, dim=1)
            sim_mat = src_norm @ tgt_norm.T
            top4_idx = torch.topk(sim_mat, k=4, dim=1)[1]
            X_hat = torch.stack([Y_t[idx].mean(dim=0) for idx in top4_idx]).cpu().numpy()
            
            w_c = voc(X_hat)
            cers.append(jiwer.cer(ref, tr(w_c)))
            sims.append(torch.dot(tgt_prof, emb(w_c)).item())
        print(f"4-NN (kNN-VC)              : Sim = {np.mean(sims):.4f} | CER = {np.mean(cers)*100:.2f}%")
