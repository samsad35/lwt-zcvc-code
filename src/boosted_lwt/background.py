"""
Universal Background Model (UBM) & Target Speaker Representation.

Handles:
  - Loading pre-trained 40-speaker balanced phonetic centroids and covariances.
  - Computing target speaker representations (cluster statistics and covariance deviations).
"""

from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import numpy as np
import torch
from .models import extract_features


def load_universal_background(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Union[str, torch.device] = "cuda",
) -> Dict[str, Any]:
    """
    Load the pre-trained 40-speaker balanced universal background model (UBM).
    Contains K=10 phonetic cluster centroids and covariances from LibriSpeech (20M / 20F).
    """
    device = torch.device(device)

    # Search paths
    candidates = []
    if checkpoint_path is not None:
        candidates.append(Path(checkpoint_path))

    # Package local checkpoint
    pkg_dir = Path(__file__).resolve().parent.parent.parent
    candidates.append(pkg_dir / "checkpoints" / "shared_clusters_k10_balanced_40spk.pt")
    candidates.append(pkg_dir / "output" / "cache" / "shared_clusters_k10_balanced_40spk.pt")
    candidates.append(pkg_dir / "output" / "cache" / "shared_clusters_k10.pt")

    resolved_path = None
    for cand in candidates:
        if cand.exists():
            resolved_path = cand
            break

    if resolved_path is None:
        raise FileNotFoundError(
            f"Universal background checkpoint not found. Looked in: {[str(c) for c in candidates]}. "
            "Please provide a valid checkpoint_path to load_universal_background()."
        )

    data = torch.load(resolved_path, map_location=device)
    
    cents_shared = data['cents_shared'].to(device)
    cents_shared_norm = data['cents_shared_norm'].to(device)
    cov_shared_k = [c.to(device) for c in data['cov_shared_k']]

    return {
        'path': resolved_path,
        'cents_shared': cents_shared,
        'cents_shared_norm': cents_shared_norm,
        'cov_shared_k': cov_shared_k,
        'K_clusters': len(cov_shared_k),
    }


def compute_wct_matrices(Y_t: torch.Tensor, k_components: int = 128):
    """Compute global Whitening and Coloring matrices for global WCT baseline."""
    Y_np = Y_t.cpu().numpy()
    mu_Y = np.mean(Y_np, axis=0)
    yc = Y_np - mu_Y
    cov_Y = np.cov(yc, rowvar=False)

    evals, evecs = np.linalg.eigh(cov_Y)
    idx = np.argsort(evals)[::-1]
    k_comp = min(k_components, len(Y_np) - 1, len(evals))
    evals = np.maximum(evals[idx][:k_comp], 1e-5)
    evecs = evecs[:, idx][:, :k_comp]

    W_Y = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    C_Y = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
    return mu_Y, W_Y, C_Y


def build_target_representation(
    target_audio_paths_or_tensors: List[Union[str, Path, torch.Tensor]],
    wavlm: torch.nn.Module,
    cents_shared_norm: torch.Tensor,
    cov_shared_k: List[torch.Tensor],
    device: Union[str, torch.device] = "cuda",
    beta: float = 20.0,
    K_clusters: int = 10,
) -> Dict[str, Any]:
    """
    Build target speaker representation from a list of reference audio files or tensors.
    
    Computes:
      - Concatenated representations Y_t
      - Cluster membership weights w_Y
      - Target cluster centroids cl_mu_Y_shared
      - Target cluster covariances cl_cov_Y_shared
      - Covariance deviations delta_cov_speaker = Cov_Y - Cov_shared
      - Global WCT matrices (for baseline comparison)
    """
    device = torch.device(device)
    Y_list, wav_list = [], []

    for item in target_audio_paths_or_tensors:
        feat, wav = extract_features(item, wavlm, device)
        Y_list.append(feat)
        wav_list.append(wav)

    Y_t = torch.cat(Y_list, dim=0)  # (N_tgt_frames, 1024)
    Y_norm = torch.nn.functional.normalize(Y_t, dim=1)

    # Soft cluster assignments relative to universal background centroids
    sim_Y = torch.mm(Y_norm, cents_shared_norm.t())
    w_Y = torch.softmax(sim_Y * beta, dim=1)

    cl_mu_Y_shared = []
    cl_cov_Y_shared = []
    delta_cov_speaker = []

    for k in range(K_clusters):
        wk = w_Y[:, k:k+1]
        mass_k = wk.sum()
        muk = (Y_t * wk).sum(dim=0) / (mass_k + 1e-8)
        yc = Y_t - muk.unsqueeze(0)
        cov_Y_k = torch.mm(yc.t(), yc * wk) / (mass_k + 1e-8)

        cl_mu_Y_shared.append(muk)
        cl_cov_Y_shared.append(cov_Y_k)
        delta_cov_speaker.append(cov_Y_k - cov_shared_k[k])

    # Global WCT matrices
    mu_Y, W_Y, C_Y = compute_wct_matrices(Y_t, k_components=128)

    return {
        'Y_t': Y_t,
        'Y_norm': Y_norm,
        'cl_mu_Y_shared': cl_mu_Y_shared,
        'cl_cov_Y_shared': cl_cov_Y_shared,
        'delta_cov_speaker': delta_cov_speaker,
        'mu_Y': mu_Y,
        'W_Y': W_Y,
        'C_Y': C_Y,
        'ref_wav': wav_list[0] if wav_list else None,
        'num_frames': Y_t.shape[0],
    }
