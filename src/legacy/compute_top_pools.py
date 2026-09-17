import os
import sys
import warnings
import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
from scipy.linalg import svd, orthogonal_procrustes
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from transformers import WavLMModel, AutoFeatureExtractor

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.append(str(Path(__file__).parent.parent))
from une import LibriSpeech

def get_whitened(X, k=128):
    mu = np.mean(X, axis=0)
    X_c = X - mu
    cov = np.cov(X_c, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx][:k]
    vecs = vecs[:, idx][:, :k]
    D_inv_half = np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-5)))
    W = vecs @ D_inv_half @ vecs.T
    X_w = X_c @ W
    return X_w, vecs, cov

def compute_metrics(X, Y, k=128):
    X_w, U_X, cov_X = get_whitened(X, k)
    Y_w, U_Y, cov_Y = get_whitened(Y, k)
    
    cov_X_norm = cov_X / np.trace(cov_X)
    cov_Y_norm = cov_Y / np.trace(cov_Y)
    cov_dist = np.linalg.norm(cov_X_norm - cov_Y_norm, 'fro')
    
    M = U_X.T @ U_Y
    sigma = svd(M, compute_uv=False)
    sigma = np.clip(sigma, -1.0, 1.0)
    grassmann_dist = np.sqrt(np.sum(np.arccos(sigma)**2))
    
    return cov_dist, grassmann_dist

def extract_hf(p, model, extractor):
    wav, sr = torchaudio.load(str(p))
    if sr != 16000:
        wav = F.resample(wav, sr, 16000)
    wav = wav.squeeze().numpy()
    inputs = extractor(wav, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        h = outputs.hidden_states[6].squeeze(0).cpu().numpy()
    return h

def main():
    print("Loading HuggingFace model...")
    extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
    model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device).eval()
    
    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    librispeech = LibriSpeech(root=dataset_path, ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    
    spk1_id = "121"
    spk2_id = "237"
    
    # We take 20 utterances per speaker to build large asymptotic pools
    spk1_paths = df_all[df_all['path'].astype(str).str.contains(f"/{spk1_id}/")]['path'].tolist()[:20]
    spk2_paths = df_all[df_all['path'].astype(str).str.contains(f"/{spk2_id}/")]['path'].tolist()[:20]
    
    print("Extracting asymptotic features...")
    X_A_list = [extract_hf(p, model, extractor) for p in spk1_paths]
    X_B_list = [extract_hf(p, model, extractor) for p in spk2_paths]
    
    # Create two massive pools for Speaker A (Pool 1 and Pool 2) to cancel phonetic variance
    X_A_pool1 = np.concatenate(X_A_list[:10], axis=0)
    X_A_pool2 = np.concatenate(X_A_list[10:20], axis=0)
    
    # Create a massive pool for Speaker B
    X_B_pool1 = np.concatenate(X_B_list[:10], axis=0)
    X_B_pool2 = np.concatenate(X_B_list[10:20], axis=0)
    
    print("Computing metrics on large phonetically-balanced pools...")
    
    # Intra-speaker (A vs A)
    cov_same, grass_same = compute_metrics(X_A_pool1, X_A_pool2)
    
    # Inter-speaker (A vs B)
    cov_diff, grass_diff = compute_metrics(X_A_pool1, X_B_pool1)
    
    print("\n================ TOPOLOGICAL SEPARATION (ASYMPTOTIC POOLS) ================")
    print(f"{'Covariance Distance (Same)':<30} : {cov_same:.4f}")
    print(f"{'Covariance Distance (Diff)':<30} : {cov_diff:.4f}")
    print(f"{'Grassmann Distance (Same)':<30} : {grass_same:.4f}")
    print(f"{'Grassmann Distance (Diff)':<30} : {grass_diff:.4f}")

if __name__ == "__main__":
    main()
