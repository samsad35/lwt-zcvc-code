#!/usr/bin/env python3
"""
Generate and Listen to Voice Conversion Samples with Boosted Local Wasserstein Transport (Boosted LWT).

Compares:
  - Source Audio (Original)
  - Target Reference Audio (Authentic)
  - Boosted LWT (alpha=1.5 - Ours)
  - Baseline LWT (alpha=1.0 - Standard Bures-Wasserstein)
  - kNN-VC (k=4 - Nearest Neighbor instance matching)
  - Classic WCT (Global Whitening & Coloring Transform)

Usage:
  # Generate curated demo suite (4 diverse pairs: F->M, M->F, F->F, M->M) + HTML player:
  python scripts/generate_boosted_lwt_samples.py --demo

  # Convert a custom source wav to a target speaker:
  python scripts/generate_boosted_lwt_samples.py --source_wav path/to/source.wav --target_spk 1089 --alpha 1.5
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Core Audio & Math Helper Functions
# ---------------------------------------------------------------------------

def extract_features(wav_path, wavlm, device):
    """Load audio, resample to 16kHz, and extract WavLM layer 6 representation."""
    wav, sr = torchaudio.load(str(wav_path))
    wav = wav.to(device)
    if sr != 16000:
        wav = F.resample(wav, sr, 16000)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    with torch.no_grad():
        feat, _ = wavlm.extract_features(wav, output_layer=6)
    return feat.squeeze(0), wav.squeeze().cpu()

def vocode(features, hifigan, device):
    """Synthesize 16kHz waveform from 1024-d WavLM features using HiFi-GAN."""
    with torch.inference_mode():
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32, device=device)
        elif features.device != device:
            features = features.to(device)
        if features.dim() == 2:
            features = features.unsqueeze(0)
        audio = hifigan(features).squeeze().cpu().numpy()
        # Normalize to avoid clipping
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val * 0.99
        return audio

def get_wct_matrices(X, k=128):
    """Compute global Whitening (W) and Coloring (C) matrices."""
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    mu = np.mean(X, axis=0)
    X_c = X - mu
    cov = np.cov(X_c, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evals = np.maximum(evals[idx][:k], 1e-5)
    evecs = evecs[:, idx][:, :k]
    D_inv_half = np.diag(1.0 / np.sqrt(evals))
    D_half = np.diag(np.sqrt(evals))
    W = evecs @ D_inv_half @ evecs.T
    C = evecs @ D_half @ evecs.T
    return mu, W, C

def project_psd(cov, eps=1e-3):
    """Project a symmetric matrix onto the cone of positive semi-definite matrices."""
    evals, evecs = torch.linalg.eigh(cov)
    evals = torch.clamp(evals, min=eps)
    return evecs @ torch.diag(evals) @ evecs.t()

def compute_bures_map_torch(cov_X, cov_Y, eps=1e-2):
    """Compute the Bures-Wasserstein Monge transport matrix A = Cov_X^{-1/2} (Cov_X^{1/2} Cov_Y Cov_X^{1/2})^{1/2} Cov_X^{-1/2}."""
    d = cov_X.shape[0]
    I = torch.eye(d, device=cov_X.device)
    cov_X_reg = cov_X + eps * I
    cov_Y_reg = cov_Y + eps * I
    
    evals_X, evecs_X = torch.linalg.eigh(cov_X_reg)
    evals_X = torch.clamp(evals_X, min=eps)
    X_half = evecs_X @ torch.diag(torch.sqrt(evals_X)) @ evecs_X.t()
    X_inv_half = evecs_X @ torch.diag(1.0 / torch.sqrt(evals_X)) @ evecs_X.t()
    
    M = X_half @ cov_Y_reg @ X_half
    evals_M, evecs_M = torch.linalg.eigh(M)
    evals_M = torch.clamp(evals_M, min=eps)
    M_half = evecs_M @ torch.diag(torch.sqrt(evals_M)) @ evecs_M.t()
    
    A = X_inv_half @ M_half @ X_inv_half
    return A

# ---------------------------------------------------------------------------
# Background Clusters & Target Model Builder
# ---------------------------------------------------------------------------

def get_shared_background_clusters(root_path, wavlm, device, K_clusters=10, cache_path=Path("output/cache/shared_clusters_k10.pt")):
    """Load or compute universal background phonetic clusters (UBM) from disjoint speakers."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f"[Cache] Loading shared clusters from {cache_path}...")
        data = torch.load(cache_path, map_location=device)
        return data['cents_shared'], data['cents_shared_norm'], data['cov_shared_k']

    print("[Background] Precomputing shared phonetic clusters from neutral background speakers...")
    bg_spks = ['4970', '4992', '5142', '5639', '5683', '61']
    bg_feats = []
    for spk in bg_spks:
        spk_files = sorted(list((root_path / spk).rglob('*.flac')))[:6]
        for f in spk_files:
            feat, _ = extract_features(f, wavlm, device)
            bg_feats.append(feat.cpu().numpy())
    bg_Y = np.concatenate(bg_feats, axis=0)

    km = KMeans(n_clusters=K_clusters, random_state=42, n_init=1).fit(bg_Y)
    cents = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)
    cents_norm = torch.nn.functional.normalize(cents, dim=1)

    cov_shared = []
    for k in range(K_clusters):
        idx_k = np.where(km.labels_ == k)[0]
        Yk = bg_Y[idx_k]
        muk = np.mean(Yk, axis=0)
        covk = np.cov(Yk - muk, rowvar=False) if len(Yk) > 1 else np.zeros((1024, 1024))
        cov_shared.append(torch.tensor(covk, dtype=torch.float32, device=device))

    torch.save({
        'cents_shared': cents,
        'cents_shared_norm': cents_norm,
        'cov_shared_k': cov_shared
    }, cache_path)
    print(f"[Cache] Saved shared clusters to {cache_path}.")
    return cents, cents_norm, cov_shared

def build_target_representation(target_files, wavlm, cents_shared_norm, cov_shared_k, device, K_clusters=10, beta=20.0):
    """Precompute all target matrices: WCT, kNN reference index, and cluster covariances."""
    print(f"[Target] Extracting representations from {len(target_files)} target utterances...")
    Y_list, wav_list = [], []
    for f in target_files:
        feat, wav = extract_features(f, wavlm, device)
        Y_list.append(feat)
        wav_list.append(wav)
    
    Y_t = torch.cat(Y_list, dim=0) # (N_tgt, 1024)
    Y_norm = torch.nn.functional.normalize(Y_t, dim=1)
    
    # Global WCT matrices
    mu_Y, W_Y, C_Y = get_wct_matrices(Y_t, k=128)
    
    # Local cluster statistics
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
        
    return {
        'Y_t': Y_t,
        'Y_norm': Y_norm,
        'mu_Y': mu_Y,
        'W_Y': W_Y,
        'C_Y': C_Y,
        'cl_mu_Y_shared': cl_mu_Y_shared,
        'cl_cov_Y_shared': cl_cov_Y_shared,
        'delta_cov_speaker': delta_cov_speaker,
        'ref_wav': wav_list[0] # first genuine reference audio
    }

# ---------------------------------------------------------------------------
# Conversion Algorithms
# ---------------------------------------------------------------------------

def convert_lwt(xs_t, target_model, cents_shared_norm, cov_shared_k, alpha=1.5, beta=20.0, K_clusters=10):
    """Local Wasserstein Transport with optional Speaker Covariance Boosting (alpha)."""
    xs_n = torch.nn.functional.normalize(xs_t, dim=1)
    sim_X = torch.mm(xs_n, cents_shared_norm.t())
    w_X = torch.softmax(sim_X * beta, dim=1)
    
    x_hat = torch.zeros_like(xs_t)
    for k in range(K_clusters):
        wk = w_X[:, k:k+1]
        mass_k = wk.sum()
        if mass_k < 1e-4:
            continue
        mu_X_k = (xs_t * wk).sum(dim=0) / mass_k
        xc = xs_t - mu_X_k.unsqueeze(0)
        cov_X_k = torch.mm(xc.t(), xc * wk) / mass_k
        
        # Boosted target covariance
        cov_target = cov_shared_k[k] + alpha * target_model['delta_cov_speaker'][k]
        cov_target = project_psd(cov_target, eps=1e-3)
        
        # Exact Bures-Wasserstein Monge transport map
        A_k = compute_bures_map_torch(cov_X_k, cov_target, eps=1e-2)
        T_k = torch.mm(xc, A_k) + target_model['cl_mu_Y_shared'][k].unsqueeze(0)
        x_hat += wk * T_k
        
    return x_hat

def convert_knn_vc(xs_t, target_model, k=4):
    """kNN-VC nearest neighbor frame matching."""
    xs_n = torch.nn.functional.normalize(xs_t, dim=1)
    sim_mat = torch.mm(xs_n, target_model['Y_norm'].t())
    topk_idx = torch.topk(sim_mat, k=k, dim=1).indices
    x_hat = torch.mean(target_model['Y_t'][topk_idx], dim=1)
    return x_hat

def convert_classic_wct(xs_t, target_model):
    """Classic Global Whitening and Coloring Transform."""
    xs_np = xs_t.cpu().numpy()
    mu_X, W_X, _ = get_wct_matrices(xs_np, k=min(128, len(xs_np)-1))
    xw = (xs_np - mu_X) @ W_X
    x_hat_np = xw @ target_model['C_Y'].T + target_model['mu_Y']
    return torch.tensor(x_hat_np, dtype=torch.float32, device=xs_t.device)

# ---------------------------------------------------------------------------
# HTML Demo Player Generator
# ---------------------------------------------------------------------------

def generate_html_player(demo_results, out_dir):
    """Generate a clean, responsive HTML audio player to listen and compare samples side-by-side."""
    html_path = out_dir / "index.html"
    
    cards_html = ""
    for idx, res in enumerate(demo_results, 1):
        pair_name = res['pair_name']
        src_spk = res['src_spk']
        tgt_spk = res['tgt_spk']
        transcript = res.get('transcript', '')
        
        cards_html += f"""
        <div class="card">
            <div class="card-header">
                <h2>Conversion {idx}: {pair_name}</h2>
                <div class="badge">Source: {src_spk} &rarr; Target: {tgt_spk}</div>
            </div>
            <div class="card-body">
                <p class="transcript"><strong>Source Text:</strong> "{transcript}"</p>
                <div class="audio-grid">
                    <div class="audio-box ref-box">
                        <span class="label">Original Source Audio</span>
                        <span class="sublabel">Locuteur Source ({src_spk})</span>
                        <audio controls preload="none">
                            <source src="{res['rel_source']}" type="audio/wav">
                        </audio>
                    </div>
                    <div class="audio-box ref-box">
                        <span class="label">Target Reference Audio</span>
                        <span class="sublabel">Locuteur Cible Authentique ({tgt_spk})</span>
                        <audio controls preload="none">
                            <source src="{res['rel_target']}" type="audio/wav">
                        </audio>
                    </div>
                    <div class="audio-box highlight-box">
                        <span class="label gold">&#9733; Boosted LWT (&alpha;=1.5 - Ours)</span>
                        <span class="sublabel">Speaker-Covariance Monge-Kantorovich (Top Sim & EER)</span>
                        <audio controls preload="none">
                            <source src="{res['rel_boosted']}" type="audio/wav">
                        </audio>
                    </div>
                    <div class="audio-box">
                        <span class="label">Baseline LWT (&alpha;=1.0)</span>
                        <span class="sublabel">Standard Local Bures-Wasserstein</span>
                        <audio controls preload="none">
                            <source src="{res['rel_baseline']}" type="audio/wav">
                        </audio>
                    </div>
                    <div class="audio-box">
                        <span class="label">kNN-VC (k=4)</span>
                        <span class="sublabel">Local Instance Matching Baseline</span>
                        <audio controls preload="none">
                            <source src="{res['rel_knn']}" type="audio/wav">
                        </audio>
                    </div>
                    <div class="audio-box">
                        <span class="label">Classic WCT</span>
                        <span class="sublabel">Global Gaussian Normalization</span>
                        <audio controls preload="none">
                            <source src="{res['rel_wct']}" type="audio/wav">
                        </audio>
                    </div>
                </div>
            </div>
        </div>
        """
        
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Conversion Audio Samples | Boosted LWT vs Baselines</title>
    <style>
        :root {{
            --bg: #0f172a;
            --card-bg: #1e293b;
            --box-bg: #334155;
            --accent: #38bdf8;
            --gold: #f59e0b;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --border: #475569;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 2rem 1rem;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        header {{
            text-align: center;
            margin-bottom: 2.5rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border);
        }}
        h1 {{
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
            color: var(--text);
        }}
        p.subtitle {{
            color: var(--text-muted);
            font-size: 1.1rem;
            max-width: 750px;
            margin: 0 auto;
        }}
        .card {{
            background-color: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.75rem;
        }}
        .card-header h2 {{
            margin: 0;
            font-size: 1.35rem;
            color: var(--accent);
        }}
        .badge {{
            background: #0284c7;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .transcript {{
            background: rgba(0,0,0,0.2);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-style: italic;
            color: #cbd5e1;
            margin-bottom: 1.25rem;
        }}
        .audio-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
        }}
        .audio-box {{
            background: var(--box-bg);
            padding: 1rem;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            border: 1px solid var(--border);
        }}
        .audio-box.ref-box {{
            border-left: 4px solid var(--text-muted);
        }}
        .audio-box.highlight-box {{
            border: 2px solid var(--gold);
            background: #242c3d;
            box-shadow: 0 0 10px rgba(245, 158, 11, 0.2);
        }}
        .label {{
            font-weight: 700;
            font-size: 0.95rem;
            margin-bottom: 0.2rem;
        }}
        .label.gold {{
            color: var(--gold);
        }}
        .sublabel {{
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-bottom: 0.75rem;
        }}
        audio {{
            width: 100%;
            height: 38px;
            border-radius: 6px;
            outline: none;
        }}
        footer {{
            text-align: center;
            margin-top: 3rem;
            color: var(--text-muted);
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Voice Conversion Audio Listening Demo</h1>
            <p class="subtitle">
                Listening comparison across Voice Conversion paradigms on LibriSpeech test-clean utterances. 
                Evaluate naturalness, speaker identity preservation, and temporal stability.
            </p>
        </header>

        {cards_html}

        <footer>
            <p>Generated by <code>scripts/generate_boosted_lwt_samples.py</code> &bull; Local Wasserstein Transport (LWT)</p>
        </footer>
    </div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\n[HTML Player] Created interactive audio player at: {html_path.resolve()}")

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate and listen to VC samples with Boosted LWT")
    parser.add_argument("--demo", action="store_true", default=True, help="Generate curated 4-pair demo suite across genders")
    parser.add_argument("--source_wav", type=str, default=None, help="Path to a custom source .wav / .flac file")
    parser.add_argument("--target_spk", type=str, default=None, help="Target speaker ID in LibriSpeech (e.g. 1089, 121, 260)")
    parser.add_argument("--alpha", type=float, default=1.5, help="Speaker covariance boost factor (default: 1.5)")
    parser.add_argument("--beta", type=float, default=20.0, help="Soft clustering inverse temperature (default: 20.0)")
    parser.add_argument("--out_dir", type=str, default="output/audio_samples/boosted_lwt_demo", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Computation device")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root_librispeech = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")

    print(f"=== Boosted LWT Audio Generation ===")
    print(f"Device: {args.device} | Boost alpha: {args.alpha} | Beta: {args.beta}")
    print(f"Output Directory: {out_dir.resolve()}\n")

    print("[Model Loading] Loading WavLM-Large and HiFi-GAN Vocoder...")
    wavlm = torch.hub.load('bshall/knn-vc', 'wavlm_large', trust_repo=True, device=args.device).eval()
    hifigan, _ = torch.hub.load('bshall/knn-vc', 'hifigan_wavlm', trust_repo=True, prematched=True, device=args.device)
    hifigan.eval()
    print("[Model Loading] Done.\n")

    # Load / compute shared background clusters
    cents_shared, cents_shared_norm, cov_shared_k = get_shared_background_clusters(
        root_librispeech, wavlm, args.device, K_clusters=10
    )

    # If user provided a specific custom source and target
    if args.source_wav is not None and args.target_spk is not None:
        source_path = Path(args.source_wav)
        tgt_spk = str(args.target_spk)
        print(f"\n[Custom Conversion] {source_path.name} -> Target Speaker {tgt_spk}")
        
        tgt_files = sorted(list((root_librispeech / tgt_spk).rglob('*.flac')))[:20]
        if len(tgt_files) == 0:
            raise FileNotFoundError(f"No audio files found for speaker {tgt_spk} in {root_librispeech}")
            
        target_model = build_target_representation(tgt_files, wavlm, cents_shared_norm, cov_shared_k, args.device)
        xs_t, src_wav = extract_features(source_path, wavlm, args.device)

        # 1. Boosted LWT
        print(f"Generating Boosted LWT (alpha={args.alpha})...")
        x_boosted = convert_lwt(xs_t, target_model, cents_shared_norm, cov_shared_k, alpha=args.alpha, beta=args.beta)
        audio_boosted = vocode(x_boosted, hifigan, args.device)
        
        # Save output
        out_boosted = out_dir / f"converted_boosted_alpha{args.alpha}_{source_path.stem}_to_{tgt_spk}.wav"
        sf.write(out_boosted, audio_boosted, 16000)
        print(f"-> Saved: {out_boosted}")
        return

    # Otherwise run curated demo suite
    print("Preparing Curated Multi-Gender Demo Suite (4 conversions)...")
    demo_configs = [
        {
            'pair_name': 'Female to Male (Cross-Gender)',
            'src_spk': '1284 (Female)',
            'src_id': '1284',
            'tgt_spk': '1089 (Male)',
            'tgt_id': '1089',
        },
        {
            'pair_name': 'Male to Female (Cross-Gender)',
            'src_spk': '1320 (Male)',
            'src_id': '1320',
            'tgt_spk': '121 (Female)',
            'tgt_id': '121',
        },
        {
            'pair_name': 'Female to Female (Same-Gender)',
            'src_spk': '1580 (Female)',
            'src_id': '1580',
            'tgt_spk': '237 (Female)',
            'tgt_id': '237',
        },
        {
            'pair_name': 'Male to Male (Same-Gender)',
            'src_spk': '672 (Male)',
            'src_id': '672',
            'tgt_spk': '1188 (Male)',
            'tgt_id': '1188',
        }
    ]

    demo_results = []
    
    for c_idx, cfg in enumerate(demo_configs, 1):
        print(f"\n=======================================================")
        print(f"[{c_idx}/4] Processing: {cfg['pair_name']}")
        print(f"Source: {cfg['src_spk']} | Target: {cfg['tgt_spk']}")
        print(f"=======================================================")
        
        src_dir = root_librispeech / cfg['src_id']
        tgt_dir = root_librispeech / cfg['tgt_id']
        
        src_files = sorted(list(src_dir.rglob('*.flac')))
        tgt_files = sorted(list(tgt_dir.rglob('*.flac')))[:20]
        
        # Pick a clean source utterance
        src_file = src_files[0]
        
        # Get transcript if available
        transcript = ""
        trans_files = list(src_file.parent.glob("*.trans.txt"))
        if trans_files:
            with open(trans_files[0]) as tf:
                for line in tf:
                    if line.startswith(src_file.stem):
                        transcript = line[len(src_file.stem):].strip()
                        break
                        
        print(f"Source file: {src_file.name}")
        if transcript:
            print(f"Transcript: \"{transcript}\"")
            
        # Build target model
        target_model = build_target_representation(tgt_files, wavlm, cents_shared_norm, cov_shared_k, args.device)
        
        # Extract source features
        xs_t, src_wav = extract_features(src_file, wavlm, args.device)
        
        # Subfolder for this conversion
        pair_slug = f"pair{c_idx}_{cfg['src_id']}_to_{cfg['tgt_id']}"
        pair_dir = out_dir / pair_slug
        pair_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save original source audio
        src_out = pair_dir / "source.wav"
        sf.write(src_out, src_wav.numpy(), 16000)
        
        # 2. Save target authentic reference audio
        tgt_ref_out = pair_dir / "target_ref.wav"
        sf.write(tgt_ref_out, target_model['ref_wav'].numpy(), 16000)
        
        # 3. Boosted LWT (alpha=1.5 - Ours)
        print("Synthesizing: Boosted LWT (alpha=1.5)...")
        x_boosted = convert_lwt(xs_t, target_model, cents_shared_norm, cov_shared_k, alpha=1.5, beta=args.beta)
        audio_boosted = vocode(x_boosted, hifigan, args.device)
        out_boosted = pair_dir / "converted_boosted_lwt.wav"
        sf.write(out_boosted, audio_boosted, 16000)
        
        # 4. Baseline LWT (alpha=1.0)
        print("Synthesizing: Baseline LWT (alpha=1.0)...")
        x_base = convert_lwt(xs_t, target_model, cents_shared_norm, cov_shared_k, alpha=1.0, beta=args.beta)
        audio_base = vocode(x_base, hifigan, args.device)
        out_base = pair_dir / "converted_baseline_lwt.wav"
        sf.write(out_base, audio_base, 16000)
        
        # 5. kNN-VC (k=4)
        print("Synthesizing: kNN-VC (k=4)...")
        x_knn = convert_knn_vc(xs_t, target_model, k=4)
        audio_knn = vocode(x_knn, hifigan, args.device)
        out_knn = pair_dir / "converted_knnvc_k4.wav"
        sf.write(out_knn, audio_knn, 16000)
        
        # 6. Classic WCT
        print("Synthesizing: Classic WCT...")
        x_wct = convert_classic_wct(xs_t, target_model)
        audio_wct = vocode(x_wct, hifigan, args.device)
        out_wct = pair_dir / "converted_classic_wct.wav"
        sf.write(out_wct, audio_wct, 16000)
        
        demo_results.append({
            'pair_name': cfg['pair_name'],
            'src_spk': cfg['src_spk'],
            'tgt_spk': cfg['tgt_spk'],
            'transcript': transcript,
            'rel_source': f"{pair_slug}/source.wav",
            'rel_target': f"{pair_slug}/target_ref.wav",
            'rel_boosted': f"{pair_slug}/converted_boosted_lwt.wav",
            'rel_baseline': f"{pair_slug}/converted_baseline_lwt.wav",
            'rel_knn': f"{pair_slug}/converted_knnvc_k4.wav",
            'rel_wct': f"{pair_slug}/converted_classic_wct.wav",
        })
        
    # Generate HTML demo player
    generate_html_player(demo_results, out_dir)
    
    print("\n=======================================================")
    print(" ALL SAMPLES GENERATED SUCCESSFULLY!")
    print(f"Directory containing all audio files: {out_dir.resolve()}")
    print(f"Open in browser to listen: file://{out_dir.resolve()}/index.html")
    print("=======================================================")

if __name__ == "__main__":
    main()
