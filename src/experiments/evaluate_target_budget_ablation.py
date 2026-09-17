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

print("=== Target Data Budget Ablation: Ours (Soft WCT K=25) vs. kNN-VC (1-NN) ===")
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
    k = min(k, len(e_val), max(1, len(X)-1))
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

# Target frame budgets Nt to test (~2s, ~5s, ~10s, ~20s, ~40s, ~80s)
budgets = [100, 250, 500, 1000, 2000, 4000]

records = []

for spk in target_speakers:
    print(f"\n=======================================================")
    print(f"--> Target Speaker: {spk}")
    print(f"=======================================================")
    files = sorted(list((root / spk).rglob('*.flac')))[:15]
    X_list, wavs = [], []
    for p in files:
        x, w = ext(p); X_list.append(x); wavs.append(w)
    Y_full = np.concatenate(X_list, axis=0) # [Nt_total, 1024]
    
    # Ground truth speaker profile (computed from full clean target reference)
    prof = torch.stack([emb(w) for w in wavs[:5]]).mean(dim=0)
    prof = torch.nn.functional.normalize(prof, dim=0)
    
    # Pre-whiten source data
    whitened_src = []
    for xs, ref in src_data:
        mu_X, W_X, _ = get_wct(xs, min(128, len(xs)-1))
        xw = (xs - mu_X) @ W_X
        xs_norm = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-8)
        whitened_src.append((xs, xw, xs_norm, ref))

    for Nt_target in budgets:
        # Slice target budget
        actual_Nt = min(Nt_target, len(Y_full))
        Y_sub = Y_full[:actual_Nt]
        Y_t = torch.tensor(Y_sub, dtype=torch.float32, device=device)
        Y_norm = torch.nn.functional.normalize(Y_t, dim=1)
        dur_sec = actual_Nt * 0.02 # 20ms per frame

        print(f"\n--- Budget Nt = {actual_Nt} frames (~{dur_sec:.1f}s) ---")

        # ----------------------------------------------------
        # 1. kNN-VC (1-NN on Y_sub)
        # ----------------------------------------------------
        cers_knn, sims_knn = [], []
        for xs, xw, xs_norm, ref in whitened_src:
            xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
            xs_n = torch.nn.functional.normalize(xs_t, dim=1)
            sim_mat = xs_n @ Y_norm.T # [Ns, actual_Nt]
            idx = torch.argmax(sim_mat, dim=1)
            x_hat_knn = Y_t[idx]
            
            wc_knn = voc(x_hat_knn)
            cers_knn.append(jiwer.cer(ref, tr(wc_knn)))
            sims_knn.append(torch.dot(prof, emb(wc_knn)).item())

        mean_sim_knn = np.mean(sims_knn)
        mean_cer_knn = np.mean(cers_knn) * 100
        records.append({
            'Speaker': spk, 'Method': 'kNN-VC (1-NN)', 'Nt': actual_Nt,
            'Duration_sec': dur_sec, 'Sim': mean_sim_knn, 'CER': mean_cer_knn
        })
        print(f"  kNN-VC   | Sim = {mean_sim_knn:.4f} | CER = {mean_cer_knn:.2f}%")

        # ----------------------------------------------------
        # 2. Ours: Soft Local WCT (K=min(25, Nt//4), beta=20)
        # ----------------------------------------------------
        K_ours = min(25, max(3, actual_Nt // 4))
        km = KMeans(n_clusters=K_ours, random_state=42, n_init=1).fit(Y_sub)
        cents = km.cluster_centers_
        cents_norm = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-8)
        
        cl = []
        for k in range(K_ours):
            Yk = Y_sub[km.labels_ == k]
            if len(Yk) == 0:
                Yk = Y_sub[:1]
            muk = np.mean(Yk, axis=0)
            if len(Yk) > 64:
                _, _, Ck = get_wct(Yk, 64)
            else:
                cov_k = np.cov(Yk - muk, rowvar=False) if len(Yk) > 1 else np.zeros((1024, 1024))
                evals, evecs = np.linalg.eigh(cov_k + 1e-4 * np.eye(1024))
                evals = np.maximum(evals, 1e-5)
                Ck = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
            cl.append((muk, Ck))

        cers_ours, sims_ours = [], []
        for xs, xw, xs_norm, ref in whitened_src:
            sim_mat = xs_norm @ cents_norm.T
            logits = sim_mat * 20.0
            logits -= np.max(logits, axis=1, keepdims=True)
            w = np.exp(logits); w /= w.sum(axis=1, keepdims=True)
            
            x_hat_ours = np.zeros_like(xs)
            for k in range(K_ours):
                muk, Ck = cl[k]
                x_hat_ours += w[:, k:k+1] * (xw @ Ck.T + muk)
                
            wc_ours = voc(x_hat_ours)
            cers_ours.append(jiwer.cer(ref, tr(wc_ours)))
            sims_ours.append(torch.dot(prof, emb(wc_ours)).item())

        mean_sim_ours = np.mean(sims_ours)
        mean_cer_ours = np.mean(cers_ours) * 100
        records.append({
            'Speaker': spk, 'Method': 'Ours (Soft Local WCT)', 'Nt': actual_Nt,
            'Duration_sec': dur_sec, 'Sim': mean_sim_ours, 'CER': mean_cer_ours
        })
        print(f"  Ours     | Sim = {mean_sim_ours:.4f} | CER = {mean_cer_ours:.2f}%")

df = pd.DataFrame(records)
out_csv = Path('/local_scratch/ssadok/un_projet_audio/output/target_budget_ablation_6spk.csv')
out_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)
print(f"\nSaved target budget results to {out_csv}")

summary = df.groupby(['Nt', 'Duration_sec', 'Method']).agg({
    'Sim': ['mean', 'std'],
    'CER': ['mean', 'std']
})
print("\n" + "="*70)
print("=== SUMMARY: TARGET BUDGET ABLATION ACROSS 6 SPEAKERS ===")
print("="*70)
print(summary)
