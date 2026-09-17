import torch
import torchaudio
import numpy as np
import warnings
import sys
from scipy.linalg import svd, orthogonal_procrustes
from transformers import WavLMModel, AutoFeatureExtractor
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from une import LibriSpeech

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

def icp(X, Y, max_iterations=10, tolerance=1e-5):
    from sklearn.neighbors import NearestNeighbors
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
    
    R_star, X_w_rot, Y_w_matched = icp(X_w_128, Y_w_128, max_iterations=5)
    E_rot = np.linalg.norm(Y_w_matched - X_w_rot, 'fro') / np.linalg.norm(Y_w_matched, 'fro')
    
    return cov_dist, grassmann_dist, E_rot

def extract_hf(p, model, extractor):
    wav, sr = torchaudio.load(str(p))
    if sr != 16000:
        import torchaudio.functional as F
        wav = F.resample(wav, sr, 16000)
    wav = wav.squeeze().numpy()
    inputs = extractor(wav, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # bshall knn-vc uses layer 6
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
    
    spk1_paths = df_all[df_all['path'].astype(str).str.contains(f"/{spk1_id}/")]['path'].tolist()[:4]
    spk2_paths = df_all[df_all['path'].astype(str).str.contains(f"/{spk2_id}/")]['path'].tolist()[:4]
    
    print("Extracting features...")
    X_A_list = [extract_hf(p, model, extractor) for p in spk1_paths]
    X_B_list = [extract_hf(p, model, extractor) for p in spk2_paths]
    
    print("Computing metrics...")
    same_metrics = []
    for i in range(len(X_A_list)):
        for j in range(i+1, len(X_A_list)):
            same_metrics.append(compute_metrics(X_A_list[i], X_A_list[j]))
            
    diff_metrics = []
    for i in range(len(X_A_list)):
        for j in range(len(X_B_list)):
            diff_metrics.append(compute_metrics(X_A_list[i], X_B_list[j]))
            
    same_avg = np.mean(same_metrics, axis=0)
    diff_avg = np.mean(diff_metrics, axis=0)
    
    print("\n================ TOPOLOGICAL SEPARATION ================")
    print(f"{'Covariance Distance (Same)':<30} : {same_avg[0]:.4f}")
    print(f"{'Covariance Distance (Diff)':<30} : {diff_avg[0]:.4f}")
    print(f"{'Grassmann Distance (Same)':<30} : {same_avg[1]:.4f}")
    print(f"{'Grassmann Distance (Diff)':<30} : {diff_avg[1]:.4f}")
    print(f"{'Procrustes Error (Same)':<30} : {same_avg[2]*100:.2f}%")
    print(f"{'Procrustes Error (Diff)':<30} : {diff_avg[2]*100:.2f}%")

if __name__ == "__main__":
    main()
