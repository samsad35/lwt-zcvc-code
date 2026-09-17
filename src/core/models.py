import numpy as np
import torch

def get_wct(X, k=128):
    """Computes whitening matrix W and coloring matrix C for input X."""
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

def compute_trajectory_jitter(x_traj, x_src=None):
    """
    Computes the temporal trajectory jitter (spectral velocity / jumpiness).
    If x_src is provided, returns relative jitter normalized by source velocity.
    """
    diffs = np.linalg.norm(x_traj[1:] - x_traj[:-1], axis=1)
    jitter = np.mean(diffs)
    if x_src is not None:
        src_diffs = np.linalg.norm(x_src[1:] - x_src[:-1], axis=1)
        rel_jitter = jitter / (np.mean(src_diffs) + 1e-8)
        return jitter, rel_jitter
    return jitter
