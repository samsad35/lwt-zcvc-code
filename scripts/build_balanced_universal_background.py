import os, sys, time, torch, torchaudio
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import torchaudio.functional as AF

device = 'cuda:0'
print("=== Building Balanced Universal Background Phonetic Model (40 Speakers: 20M / 20F) ===")

train_root = Path('/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/train-clean-100')
spk_dirs = sorted([d.name for d in train_root.iterdir() if d.is_dir()])

speakers_txt = Path('/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/SPEAKERS.TXT')
genders = {}
with open(speakers_txt) as f:
    for line in f:
        if line.startswith(';') or not line.strip(): continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 3:
            genders[parts[0]] = parts[1]

males = [s for s in spk_dirs if genders.get(s) == 'M'][:20]
females = [s for s in spk_dirs if genders.get(s) == 'F'][:20]

print(f"Selected 20 Males and 20 Females (40 speakers total, strictly 50/50 balanced).")

# Load WavLM
print("Loading WavLM-Large...")
wavlm = torch.hub.load('bshall/knn-vc', 'wavlm_large', trust_repo=True, device=device).eval()

def ext(p):
    w, sr = torchaudio.load(str(p))
    w = w.to(device)
    if sr != 16000: w = AF.resample(w, sr, 16000)
    if w.dim() == 1: w = w.unsqueeze(0)
    elif w.dim() == 2 and w.shape[0] > 1: w = w.mean(dim=0, keepdim=True)
    with torch.no_grad(): feat, _ = wavlm.extract_features(w, output_layer=6)
    return feat.squeeze(0).cpu().numpy()

bg_feats = []
print("Extracting features from 40 speakers (2 utterances each = 80 utterances)...")
t0 = time.time()
for s in males + females:
    files = sorted(list((train_root / s).rglob('*.flac')))[:2]
    for f in files:
        feat = ext(f)
        bg_feats.append(feat)

bg_Y = np.concatenate(bg_feats, axis=0)
print(f"Extracted {len(bg_Y)} total frames in {time.time()-t0:.1f}s.")

K_clusters = 10
print(f"Fitting K-Means (K={K_clusters}) on balanced 50/50 dataset...")
km = KMeans(n_clusters=K_clusters, random_state=42, n_init=1).fit(bg_Y)
cents_shared = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)
cents_shared_norm = torch.nn.functional.normalize(cents_shared, dim=1)

cov_shared_k = []
for k in range(K_clusters):
    idx_k = np.where(km.labels_ == k)[0]
    Yk = bg_Y[idx_k]
    muk = np.mean(Yk, axis=0)
    covk = np.cov(Yk - muk, rowvar=False) if len(Yk) > 1 else np.zeros((1024, 1024))
    cov_shared_k.append(torch.tensor(covk, dtype=torch.float32, device=device))

cache_path = Path("output/cache/shared_clusters_k10_balanced_40spk.pt")
torch.save({
    'cents_shared': cents_shared,
    'cents_shared_norm': cents_shared_norm,
    'cov_shared_k': cov_shared_k,
    'num_speakers': len(males) + len(females),
    'num_males': len(males),
    'num_females': len(females)
}, cache_path)

print(f"[SUCCESS] Saved balanced universal background model to: {cache_path}")
