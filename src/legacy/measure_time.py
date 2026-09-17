import time
import numpy as np
import torch
import warnings
from sklearn.metrics import pairwise_distances

def measure_linearvc(X_src, X_tgt_pool, n_iters=100):
    start = time.perf_counter()
    for _ in range(n_iters):
        # 1. kNN matching
        # normalize
        src_norm = X_src / (np.linalg.norm(X_src, axis=1, keepdims=True) + 1e-8)
        tgt_norm = X_tgt_pool / (np.linalg.norm(X_tgt_pool, axis=1, keepdims=True) + 1e-8)
        # dot product
        dists = 1 - np.dot(src_norm, tgt_norm.T)
        best_indices = dists.argmin(axis=1)
        X_tgt_matched = X_tgt_pool[best_indices]
        
        # 2. Least squares
        X_src_bias = np.hstack([X_src, np.ones((X_src.shape[0], 1))])
        W_tgt, _, _, _ = np.linalg.lstsq(X_src_bias, X_tgt_matched, rcond=None)
        
        # 3. Projection
        X_conv = np.matmul(X_src_bias, W_tgt)
    end = time.perf_counter()
    return (end - start) / n_iters

def measure_gmm_wct(X_src, tgt_centroids, cluster_matrices, mu_src, W_src, n_iters=100):
    start = time.perf_counter()
    for _ in range(n_iters):
        # 1. Global Whitening
        X_w = np.matmul(X_src - mu_src, W_src)
        
        # 2. Softmax weighting
        dists = pairwise_distances(X_src, tgt_centroids, metric='euclidean')
        logits = -2.0 * dists
        logits -= np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # 3. Soft Coloring
        X_conv = np.zeros_like(X_src)
        for k in range(len(tgt_centroids)):
            mu_k, C_k = cluster_matrices[k]
            x_k = np.matmul(X_w, C_k) + mu_k
            X_conv += weights[:, k:k+1] * x_k
            
    end = time.perf_counter()
    return (end - start) / n_iters

if __name__ == "__main__":
    np.random.seed(42)
    # 5 seconds of audio at 50 FPS = 250 frames
    N_s = 250
    D = 128 # Assuming we operate in 128D PCA space for both or 1024 for Linear? 
    # LinearVC usually works in 1024D directly. GMM-WCT works in 128D. 
    # Let's measure LinearVC in 1024D and GMM-WCT in 128D as implemented.
    
    X_src_1024 = np.random.randn(N_s, 1024)
    X_src_128 = np.random.randn(N_s, 128)
    
    # Target T=30 (approx 6000 frames)
    N_t = 6000
    X_tgt_pool_1024 = np.random.randn(N_t, 1024)
    
    # GMM-WCT offline params
    K = 5
    tgt_centroids = np.random.randn(K, 128)
    cluster_matrices = [(np.random.randn(128), np.random.randn(128, 128)) for _ in range(K)]
    mu_src = np.random.randn(128)
    W_src = np.random.randn(128, 128)
    
    print("Benchmarking VC Step for 5-second audio snippet (250 frames)...")
    
    t_lin = measure_linearvc(X_src_1024, X_tgt_pool_1024, 50)
    print(f"LinearVC (T=30): {t_lin*1000:.2f} ms per utterance")
    
    t_gmm = measure_gmm_wct(X_src_128, tgt_centroids, cluster_matrices, mu_src, W_src, 500)
    print(f"GMM-WCT (K=5):   {t_gmm*1000:.2f} ms per utterance")
    
    rtf_lin = t_lin / 5.0
    rtf_gmm = t_gmm / 5.0
    
    print(f"\nReal-Time Factor (RTF) of VC Step (lower is better):")
    print(f"LinearVC RTF: {rtf_lin:.5f}")
    print(f"GMM-WCT RTF:  {rtf_gmm:.5f}")
    print(f"Speedup:      {rtf_lin / rtf_gmm:.1f}x")
