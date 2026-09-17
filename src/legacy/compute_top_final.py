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

warnings.filterwarnings("ignore")
device = "cpu"

sys.path.append(str(Path(__file__).parent.parent))
from une import LibriSpeech

def icp(X, Y, max_iterations=5, tolerance=1e-5):
    R = np.eye(X.shape[1])
    src = X.copy()
    prev_error = float('inf')
    for i in range(max_iterations):
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(Y)
        distances, indices = nbrs.kneighbors(src)
        matched_Y = Y[indices.flatten()]
        R_step, _ = orthogonal_procrustes(src, matched_Y)
        src = src @ R_step
        R = R @ R_step
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    return R, src, matched_Y

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
    
    pca_vals, pca_vecs = np.linalg.eigh(cov_Y)
    idx = np.argsort(pca_vals)[::-1]
    pca_vecs = pca_vecs[:, idx][:, :k]
    
    X_128 = X @ pca_vecs
    Y_128 = Y @ pca_vecs
    
    Y_w_128, _, _ = get_whitened(Y_128, k)
    X_w_128, _, _ = get_whitened(X_128, k)
    
    R_star, X_w_rot, Y_w_matched = icp(X_w_128, Y_w_128, max_iterations=3)
    E_rot = np.linalg.norm(Y_w_matched - X_w_rot, 'fro') / np.linalg.norm(Y_w_matched, 'fro')
    
    return cov_dist, grassmann_dist, E_rot

def extract_features(wav_path, wavlm_large):
    wav, sr = torchaudio.load(str(wav_path))
    wav = wav.to(device)
    if sr != 16000:
        wav = F.resample(wav, sr, 16000)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    with torch.no_grad():
        features_large, _ = wavlm_large.extract_features(wav, output_layer=6)
        h_x = features_large.squeeze(0).cpu().numpy()
    return h_x

def main():
    print("Loading models via direct local import...")
    sys.path.append("/scratch/pictor/ssadok/.cache/torch/hub/bshall_knn-vc_master")
    from wavlm.WavLM import WavLM, WavLMConfig
    checkpoint = torch.load("/scratch/pictor/ssadok/.cache/torch/hub/checkpoints/WavLM-Large.pt", map_location=device)
    cfg = WavLMConfig(checkpoint['cfg'])
    wavlm_large = WavLM(cfg).to(device).eval()
    wavlm_large.load_state_dict(checkpoint['model'])
    print("WavLM loaded perfectly offline.")

    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    librispeech = LibriSpeech(root=dataset_path, ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    
    spk1_id = "121"
    spk2_id = "237"
    
    spk1_paths = df_all[df_all['path'].astype(str).str.contains(f"/{spk1_id}/")]['path'].tolist()[:20]
    spk2_paths = df_all[df_all['path'].astype(str).str.contains(f"/{spk2_id}/")]['path'].tolist()[:20]
    
    print("Extracting features for pooling...")
    X_A_list = [extract_features(p, wavlm_large) for p in spk1_paths]
    X_B_list = [extract_features(p, wavlm_large) for p in spk2_paths]
    
    X_A_pool1 = np.concatenate(X_A_list[:10], axis=0)
    X_A_pool2 = np.concatenate(X_A_list[10:20], axis=0)
    X_B_pool1 = np.concatenate(X_B_list[:10], axis=0)
    
    print("Computing metrics...")
    # Intra-speaker (A vs A)
    cov_same, grass_same, proc_same = compute_metrics(X_A_pool1, X_A_pool2)
    
    # Inter-speaker (A vs B)
    cov_diff, grass_diff, proc_diff = compute_metrics(X_A_pool1, X_B_pool1)
    
    print("\n================ TOPOLOGICAL SEPARATION (POOLED) ================")
    print(f"{'Covariance Distance (Same)':<30} : {cov_same:.4f}")
    print(f"{'Covariance Distance (Diff)':<30} : {cov_diff:.4f}")
    print(f"{'Grassmann Distance (Same)':<30} : {grass_same:.4f}")
    print(f"{'Grassmann Distance (Diff)':<30} : {grass_diff:.4f}")
    print(f"{'Procrustes Error (Same)':<30} : {proc_same*100:.2f}%")
    print(f"{'Procrustes Error (Diff)':<30} : {proc_diff*100:.2f}%")

if __name__ == "__main__":
    main()
