import torch
import torchaudio
import numpy as np
from pathlib import Path
import warnings
import sys
import torchaudio.functional as F
from scipy.linalg import svd, orthogonal_procrustes
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
device = "cpu"

sys.path.append(str(Path(__file__).parent.parent))
from une import LibriSpeech

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

def icp(X, Y, max_iterations=20, tolerance=1e-5):
    # X and Y should be (N, D)
    # Find rotation R that maps X to Y
    R = np.eye(X.shape[1])
    src = X.copy()
    prev_error = float('inf')
    
    for i in range(max_iterations):
        # Find nearest neighbors in Y for each point in src
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(Y)
        distances, indices = nbrs.kneighbors(src)
        matched_Y = Y[indices.flatten()]
        
        # Compute Procrustes
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
    # Covariances and Whitening
    X_w, U_X, cov_X = get_whitened(X, k)
    Y_w, U_Y, cov_Y = get_whitened(Y, k)
    
    # 1. Covariance Distance (Frobenius on normalized covs)
    cov_X_norm = cov_X / np.trace(cov_X)
    cov_Y_norm = cov_Y / np.trace(cov_Y)
    cov_dist = np.linalg.norm(cov_X_norm - cov_Y_norm, 'fro')
    
    # 2. Grassmann Distance
    M = U_X.T @ U_Y
    sigma = svd(M, compute_uv=False)
    sigma = np.clip(sigma, -1.0, 1.0)
    grassmann_dist = np.sqrt(np.sum(np.arccos(sigma)**2))
    
    # 3. Procrustes Error (using 128D PCA space for ICP)
    pca_vals, pca_vecs = np.linalg.eigh(cov_Y)
    idx = np.argsort(pca_vals)[::-1]
    pca_vecs = pca_vecs[:, idx][:, :k]
    
    X_128 = X @ pca_vecs
    Y_128 = Y @ pca_vecs
    
    Y_w_128, _, _ = get_whitened(Y_128, k)
    X_w_128, _, _ = get_whitened(X_128, k)
    
    R_star, X_w_rot, Y_w_matched = icp(X_w_128, Y_w_128, max_iterations=10)
    E_rot = np.linalg.norm(Y_w_matched - X_w_rot, 'fro') / np.linalg.norm(Y_w_matched, 'fro')
    
    return cov_dist, grassmann_dist, E_rot

def main():
    print("Loading models (offline mode)...")
    try:
        repo_dir = "/scratch/pictor/ssadok/.cache/torch/hub/bshall_knn-vc_master"
        wavlm_large = torch.hub.load(repo_dir, "wavlm_large", source="local", device=device).eval()
        print("WavLM loaded.")
    except Exception as e:
        print("Error loading WavLM:", e)
        return
    
    print("Loading dataset table...")
    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    librispeech = LibriSpeech(root=dataset_path, ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    print("Dataset table loaded.")
    
    # Get two distinct speakers
    spk1_id = "121"
    spk2_id = "237"
    
    spk1_paths = df_all[df_all['path'].astype(str).str.contains(f"/{spk1_id}/")]['path'].tolist()[:4]
    spk2_paths = df_all[df_all['path'].astype(str).str.contains(f"/{spk2_id}/")]['path'].tolist()[:4]
    
    print(f"Extracting features for Speaker A ({spk1_id}) and Speaker B ({spk2_id})...")
    X_A_list = [extract_features(p, wavlm_large) for p in spk1_paths]
    X_B_list = [extract_features(p, wavlm_large) for p in spk2_paths]
    
    print("Computing SAME SPEAKER pairs (A1 vs A2)...")
    same_metrics = []
    for i in range(len(X_A_list)):
        for j in range(i+1, len(X_A_list)):
            same_metrics.append(compute_metrics(X_A_list[i], X_A_list[j]))
    
    print("Computing DIFFERENT SPEAKER pairs (A1 vs B1)...")
    diff_metrics = []
    for i in range(len(X_A_list)):
        for j in range(len(X_B_list)):
            diff_metrics.append(compute_metrics(X_A_list[i], X_B_list[j]))
            
    same_avg = np.mean(same_metrics, axis=0)
    diff_avg = np.mean(diff_metrics, axis=0)
    
    print("\n================ TOPOLOGICAL SEPARATION ================")
    print(f"{'Metric':<25} | {'Same Speaker':<15} | {'Different Speaker':<15}")
    print("-" * 60)
    print(f"{'Covariance Distance':<25} | {same_avg[0]:<15.4f} | {diff_avg[0]:<15.4f}")
    print(f"{'Grassmann Distance':<25} | {same_avg[1]:<15.4f} | {diff_avg[1]:<15.4f}")
    print(f"{'Procrustes Error (%)':<25} | {same_avg[2]*100:<15.2f} | {diff_avg[2]*100:<15.2f}")
    print("========================================================\n")

if __name__ == "__main__":
    main()
