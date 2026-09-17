#!/usr/bin/env python3
"""
Generate complete ICASSP 2026 Voice Conversion Demo Suite:
1. 9 Curated Benchmark Conversions across genders (F->M, M->F, F->F, M->M)
   Methods compared:
   - Source Audio (Original)
   - Target Reference Audio (Authentic)
   - Boosted LWT (alpha=1.5 - Proposed)
   - Baseline LWT (alpha=1.0 - Standard Bures-Wasserstein)
   - LinearVC (Official Interspeech 2025)
   - kNN-VC (k=4)
   - Classic WCT (Global Gaussian)
2. Comprehensive Alpha Ablation Study: alpha in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
   on two diverse cross-gender pairs (1284->1089 and 1320->121).
3. Professional, responsive, interactive ICASSP HTML demo webpage.
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
    elif wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
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
    """Compute Bures-Wasserstein Monge transport matrix A."""
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
    Y_list, wav_list = [], []
    for f in target_files:
        feat, wav = extract_features(f, wavlm, device)
        Y_list.append(feat)
        wav_list.append(wav)
    
    Y_t = torch.cat(Y_list, dim=0)
    Y_norm = torch.nn.functional.normalize(Y_t, dim=1)
    
    mu_Y, W_Y, C_Y = get_wct_matrices(Y_t, k=128)
    
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
        'ref_wav': wav_list[0]
    }

# ---------------------------------------------------------------------------
# Voice Conversion Algorithms
# ---------------------------------------------------------------------------

def convert_lwt(xs_t, target_model, cents_shared_norm, cov_shared_k, alpha=1.5, beta=20.0, K_clusters=10):
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
    xs_n = torch.nn.functional.normalize(xs_t, dim=1)
    sim_mat = torch.mm(xs_n, target_model['Y_norm'].t())
    topk_idx = torch.topk(sim_mat, k=k, dim=1).indices
    x_hat = torch.mean(target_model['Y_t'][topk_idx], dim=1)
    return x_hat

def convert_classic_wct(xs_t, target_model):
    xs_np = xs_t.cpu().numpy()
    mu_X, W_X, _ = get_wct_matrices(xs_np, k=min(128, len(xs_np)-1))
    xw = (xs_np - mu_X) @ W_X
    x_hat_np = xw @ target_model['C_Y'].T + target_model['mu_Y']
    return torch.tensor(x_hat_np, dtype=torch.float32, device=xs_t.device)

def train_and_convert_linearvc(xs_t, src_ref_files, tgt_ref_files, wavlm, device, n_frames_max=8192):
    """Train official LinearVC projection matrix W on GPU and convert test utterance xs_t."""
    x_list = [extract_features(f, wavlm, device)[0] for f in src_ref_files[:20]]
    X_ref = torch.cat(x_list, dim=0)[:n_frames_max]
    
    y_list = [extract_features(f, wavlm, device)[0] for f in tgt_ref_files[:20]]
    Y_ref = torch.cat(y_list, dim=0)[:n_frames_max]
    
    s_norm = torch.nn.functional.normalize(X_ref, p=2, dim=-1)
    m_norm = torch.nn.functional.normalize(Y_ref, p=2, dim=-1)
    dists = 1.0 - torch.mm(s_norm, m_norm.t())
    best_idx = torch.argmin(dists, dim=-1)
    Y_matched = Y_ref[best_idx]
    
    W = torch.linalg.lstsq(X_ref, Y_matched).solution
    return torch.mm(xs_t, W)

# ---------------------------------------------------------------------------
# HTML Demo Page Generator (ICASSP 2026 Grade)
# ---------------------------------------------------------------------------

def get_alpha_description(alpha):
    desc_map = {
        0.0: "Neutral shared phonetic covariance only (No speaker-specific covariance boost)",
        0.5: "Gentle target covariance injection (Subtle timbre shift)",
        1.0: "Standard Local Bures-Wasserstein Transport (Exact target cluster covariance)",
        1.5: "Optimal Boosted LWT (Proposed - Peak Pareto similarity & preserved intelligibility)",
        2.0: "High target covariance boost (Enhanced target vocal tract resonance)",
        2.5: "Strong covariance expansion (Target characteristics saturated)",
        3.0: "Extreme covariance boost (Acoustic limit: noticeable jitter & spectral roughness)"
    }
    return desc_map.get(alpha, f"Covariance boost alpha = {alpha}")

def generate_icassp_html_page(demo_results, ablation_results, out_dir):
    html_path = out_dir / "index.html"
    
    # Build ablation cards HTML
    ablation_cards_html = ""
    for ab in ablation_results:
        pair_title = ab['pair_title']
        src_label = ab['src_label']
        tgt_label = ab['tgt_label']
        transcript = ab['transcript']
        rel_src = ab['rel_source']
        rel_tgt = ab['rel_target']
        samples = ab['samples']
        
        alpha_buttons = ""
        alpha_audio_players = ""
        for a_val, rel_p in samples.items():
            active_class = "active" if a_val == 1.5 else ""
            badge_text = "Proposed &#9733;" if a_val == 1.5 else ("Neutral" if a_val == 0.0 else ("Standard LWT" if a_val == 1.0 else ("Extreme" if a_val == 3.0 else "")))
            badge_html = f'<span class="btn-badge">{badge_text}</span>' if badge_text else ""
            
            alpha_buttons += f"""
            <button class="alpha-btn {active_class}" onclick="selectAlpha('{ab['id']}', '{a_val}', this)">
                &alpha; = {a_val:.1f} {badge_html}
            </button>
            """
            
            display_style = "block" if a_val == 1.5 else "none"
            alpha_audio_players += f"""
            <div id="player_{ab['id']}_{str(a_val).replace('.', '_')}" class="alpha-player" style="display: {display_style};">
                <div class="alpha-meta">
                    <span class="alpha-title">&alpha; = {a_val:.1f} &mdash; <em>{get_alpha_description(a_val)}</em></span>
                </div>
                <audio controls preload="none">
                    <source src="{rel_p}" type="audio/wav">
                </audio>
            </div>
            """
            
        ablation_cards_html += f"""
        <div class="card ablation-card" id="{ab['id']}">
            <div class="card-header">
                <h3>{pair_title}</h3>
                <div class="badge-row">
                    <span class="badge cat-badge">Cross-Gender</span>
                    <span class="badge spk-badge">{src_label} &rarr; {tgt_label}</span>
                </div>
            </div>
            <div class="card-body">
                <p class="transcript"><strong>Source Text:</strong> "{transcript}"</p>
                
                <div class="ref-row">
                    <div class="audio-box ref-box">
                        <span class="label">Original Source Audio</span>
                        <span class="sublabel">{src_label}</span>
                        <audio controls preload="none">
                            <source src="{rel_src}" type="audio/wav">
                        </audio>
                    </div>
                    <div class="audio-box ref-box">
                        <span class="label">Target Reference Audio</span>
                        <span class="sublabel">{tgt_label}</span>
                        <audio controls preload="none">
                            <source src="{rel_tgt}" type="audio/wav">
                        </audio>
                    </div>
                </div>

                <div class="alpha-interactive-container">
                    <div class="alpha-selector-header">
                        <h4>Interactive Covariance Boost Factor &alpha; &isin; [0.0, 3.0]</h4>
                        <span class="alpha-hint">Click an &alpha; level below to listen to the continuous morphing of timbre and stability:</span>
                    </div>
                    <div class="alpha-btn-group">
                        {alpha_buttons}
                    </div>
                    <div class="alpha-player-box">
                        {alpha_audio_players}
                    </div>
                </div>
            </div>
        </div>
        """

    # Build benchmark cards HTML
    benchmark_cards_html = ""
    for idx, res in enumerate(demo_results, 1):
        pair_name = res['pair_name']
        src_spk = res['src_spk']
        tgt_spk = res['tgt_spk']
        transcript = res.get('transcript', '')
        category = res.get('category', 'Cross-Gender')
        
        linearvc_html = ""
        if res.get('rel_linearvc'):
            linearvc_html = f"""
            <div class="audio-box">
                <span class="label">LinearVC (Interspeech 2025)</span>
                <span class="sublabel">Global Least-Squares Projection</span>
                <audio controls preload="none">
                    <source src="{res['rel_linearvc']}" type="audio/wav">
                </audio>
            </div>
            """
            
        benchmark_cards_html += f"""
        <div class="card benchmark-card">
            <div class="card-header">
                <div class="header-left">
                    <span class="pair-num">Pair {idx}</span>
                    <h2>{pair_name}</h2>
                </div>
                <div class="badge-row">
                    <span class="badge cat-badge">{category}</span>
                    <span class="badge spk-badge">{src_spk} &rarr; {tgt_spk}</span>
                </div>
            </div>
            <div class="card-body">
                <p class="transcript"><strong>Source Utterance:</strong> "{transcript}"</p>
                <div class="audio-grid">
                    <div class="audio-box ref-box">
                        <span class="label">Original Source</span>
                        <span class="sublabel">{src_spk}</span>
                        <audio controls preload="none">
                            <source src="{res['rel_source']}" type="audio/wav">
                        </audio>
                    </div>
                    <div class="audio-box ref-box">
                        <span class="label">Target Reference</span>
                        <span class="sublabel">{tgt_spk}</span>
                        <audio controls preload="none">
                            <source src="{res['rel_target']}" type="audio/wav">
                        </audio>
                    </div>
                    <div class="audio-box highlight-box">
                        <span class="label gold">&#9733; Boosted LWT (&alpha;=1.5 - Proposed)</span>
                        <span class="sublabel">Local Monge-Kantorovich + Covariance Boost</span>
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
                    {linearvc_html}
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

    template_path = Path("scripts/demo_template.html")
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found at {template_path}")
    template = template_path.read_text(encoding="utf-8")
    full_html = template.replace("{{ABLATION_CARDS}}", ablation_cards_html).replace("{{BENCHMARK_CARDS}}", benchmark_cards_html)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"\n[HTML Demo] Created complete ICASSP demo page at: {html_path.resolve()}")

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Complete ICASSP Demo Generator")
    parser.add_argument("--out_dir", type=str, default="output/audio_samples/boosted_lwt_demo", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Computation device")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root_librispeech = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")

    print("=== ICASSP 2026 Complete Audio Demo Suite Generator ===")
    print(f"Device: {args.device}")
    print(f"Output: {out_dir.resolve()}\n")

    print("[1/5] Loading WavLM-Large and HiFi-GAN Vocoder...")
    wavlm = torch.hub.load('bshall/knn-vc', 'wavlm_large', trust_repo=True, device=args.device).eval()
    hifigan, _ = torch.hub.load('bshall/knn-vc', 'hifigan_wavlm', trust_repo=True, prematched=True, device=args.device)
    hifigan.eval()
    print("[1/5] Models ready.\n")

    print("[2/5] Loading shared universal background phonetic clusters...")
    cents_shared, cents_shared_norm, cov_shared_k = get_shared_background_clusters(
        root_librispeech, wavlm, args.device, K_clusters=10
    )
    print("[2/5] Shared clusters ready.\n")

    all_pairs_config = [
        {
            'slug': 'pair1_1284_to_1089',
            'pair_name': 'Conversion 1: Female to Male',
            'category': 'Cross-Gender',
            'src_spk': '1284 (Female)',
            'src_id': '1284',
            'tgt_spk': '1089 (Male)',
            'tgt_id': '1089',
        },
        {
            'slug': 'pair2_1320_to_121',
            'pair_name': 'Conversion 2: Male to Female',
            'category': 'Cross-Gender',
            'src_spk': '1320 (Male)',
            'src_id': '1320',
            'tgt_spk': '121 (Female)',
            'tgt_id': '121',
        },
        {
            'slug': 'pair3_1580_to_237',
            'pair_name': 'Conversion 3: Female to Female',
            'category': 'Same-Gender',
            'src_spk': '1580 (Female)',
            'src_id': '1580',
            'tgt_spk': '237 (Female)',
            'tgt_id': '237',
        },
        {
            'slug': 'pair4_672_to_1188',
            'pair_name': 'Conversion 4: Male to Male',
            'category': 'Same-Gender',
            'src_spk': '672 (Male)',
            'src_id': '672',
            'tgt_spk': '1188 (Male)',
            'tgt_id': '1188',
        },
        {
            'slug': 'pair5_1995_to_260',
            'pair_name': 'Conversion 5: Female to Male',
            'category': 'Cross-Gender',
            'src_spk': '1995 (Female)',
            'src_id': '1995',
            'tgt_spk': '260 (Male)',
            'tgt_id': '260',
        },
        {
            'slug': 'pair6_908_to_3570',
            'pair_name': 'Conversion 6: Male to Female',
            'category': 'Cross-Gender',
            'src_spk': '908 (Male)',
            'src_id': '908',
            'tgt_spk': '3570 (Female)',
            'tgt_id': '3570',
        },
        {
            'slug': 'pair7_121_to_1580',
            'pair_name': 'Conversion 7: Female to Female',
            'category': 'Same-Gender',
            'src_spk': '121 (Female)',
            'src_id': '121',
            'tgt_spk': '1580 (Female)',
            'tgt_id': '1580',
        },
        {
            'slug': 'pair8_260_to_1089',
            'pair_name': 'Conversion 8: Male to Male',
            'category': 'Same-Gender',
            'src_spk': '260 (Male)',
            'src_id': '260',
            'tgt_spk': '1089 (Male)',
            'tgt_id': '1089',
        },
        {
            'slug': 'pair9_61_to_1995',
            'pair_name': 'Conversion 9: Male to Female',
            'category': 'Cross-Gender',
            'src_spk': '61 (Male)',
            'src_id': '61',
            'tgt_spk': '1995 (Female)',
            'tgt_id': '1995',
        }
    ]

    print("[3/5] Processing Benchmark Voice Conversion Pairs (9 pairs)...")
    demo_results = []
    
    for c_idx, cfg in enumerate(all_pairs_config, 1):
        pair_slug = cfg['slug']
        pair_dir = out_dir / pair_slug
        pair_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- [{c_idx}/9] {cfg['pair_name']}: {cfg['src_spk']} -> {cfg['tgt_spk']} ---")
        
        src_dir = root_librispeech / cfg['src_id']
        tgt_dir = root_librispeech / cfg['tgt_id']
        
        src_files = sorted(list(src_dir.rglob('*.flac')))
        tgt_files = sorted(list(tgt_dir.rglob('*.flac')))
        
        src_test_file = src_files[0]
        
        # Transcript
        transcript = ""
        for tf in src_test_file.parent.glob("*.trans.txt"):
            for line in tf.read_text().splitlines():
                if line.startswith(src_test_file.stem):
                    transcript = line[len(src_test_file.stem):].strip()
                    break
        print(f"Transcript: \"{transcript[:60]}...\"")
        
        target_model = build_target_representation(tgt_files[:20], wavlm, cents_shared_norm, cov_shared_k, args.device)
        xs_t, src_wav = extract_features(src_test_file, wavlm, args.device)
        
        # 1. Source wav
        src_out = pair_dir / "source.wav"
        if not src_out.exists():
            sf.write(src_out, src_wav.numpy(), 16000)
            
        # 2. Target ref wav
        tgt_ref_out = pair_dir / "target_ref.wav"
        if not tgt_ref_out.exists():
            sf.write(tgt_ref_out, target_model['ref_wav'].numpy(), 16000)
            
        # 3. Boosted LWT (alpha=1.5)
        out_boosted = pair_dir / "converted_boosted_lwt.wav"
        if not out_boosted.exists():
            print("  Synthesizing Boosted LWT (alpha=1.5)...")
            x_b = convert_lwt(xs_t, target_model, cents_shared_norm, cov_shared_k, alpha=1.5)
            sf.write(out_boosted, vocode(x_b, hifigan, args.device), 16000)
            
        # 4. Baseline LWT (alpha=1.0)
        out_base = pair_dir / "converted_baseline_lwt.wav"
        if not out_base.exists():
            print("  Synthesizing Baseline LWT (alpha=1.0)...")
            x_0 = convert_lwt(xs_t, target_model, cents_shared_norm, cov_shared_k, alpha=1.0)
            sf.write(out_base, vocode(x_0, hifigan, args.device), 16000)
            
        # 5. LinearVC (Official 2025)
        out_linearvc = pair_dir / "converted_linearvc.wav"
        if not out_linearvc.exists():
            print("  Synthesizing LinearVC (Official 2025)...")
            x_lin = train_and_convert_linearvc(xs_t, src_files[1:21], tgt_files[:20], wavlm, args.device)
            sf.write(out_linearvc, vocode(x_lin, hifigan, args.device), 16000)
            
        # 6. kNN-VC (k=4)
        out_knn = pair_dir / "converted_knnvc_k4.wav"
        if not out_knn.exists():
            print("  Synthesizing kNN-VC (k=4)...")
            x_k = convert_knn_vc(xs_t, target_model, k=4)
            sf.write(out_knn, vocode(x_k, hifigan, args.device), 16000)
            
        # 7. Classic WCT
        out_wct = pair_dir / "converted_classic_wct.wav"
        if not out_wct.exists():
            print("  Synthesizing Classic WCT...")
            x_w = convert_classic_wct(xs_t, target_model)
            sf.write(out_wct, vocode(x_w, hifigan, args.device), 16000)
            
        demo_results.append({
            'pair_name': cfg['pair_name'],
            'category': cfg['category'],
            'src_spk': cfg['src_spk'],
            'tgt_spk': cfg['tgt_spk'],
            'transcript': transcript,
            'rel_source': f"{pair_slug}/source.wav",
            'rel_target': f"{pair_slug}/target_ref.wav",
            'rel_boosted': f"{pair_slug}/converted_boosted_lwt.wav",
            'rel_baseline': f"{pair_slug}/converted_baseline_lwt.wav",
            'rel_linearvc': f"{pair_slug}/converted_linearvc.wav",
            'rel_knn': f"{pair_slug}/converted_knnvc_k4.wav",
            'rel_wct': f"{pair_slug}/converted_classic_wct.wav",
        })

    print("\n[4/5] Generating Alpha Ablation Audio Suite (alpha in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])...")
    ablation_configs = [
        {
            'id': 'ablation_pair1',
            'pair_title': 'Ablation Pair 1: Female (1284) &rarr; Male (1089) [Cross-Gender]',
            'src_label': '1284 (Female)',
            'tgt_label': '1089 (Male)',
            'src_id': '1284',
            'tgt_id': '1089',
        },
        {
            'id': 'ablation_pair2',
            'pair_title': 'Ablation Pair 2: Male (1320) &rarr; Female (121) [Cross-Gender]',
            'src_label': '1320 (Male)',
            'tgt_label': '121 (Female)',
            'src_id': '1320',
            'tgt_id': '121',
        }
    ]
    
    alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    ablation_results = []
    
    ablation_base_dir = out_dir / "ablation_alpha"
    ablation_base_dir.mkdir(parents=True, exist_ok=True)
    
    for ab_cfg in ablation_configs:
        ab_id = ab_cfg['id']
        ab_pair_dir = ablation_base_dir / ab_id
        ab_pair_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Generating Alpha Ablation for: {ab_cfg['pair_title']} ---")
        
        src_dir = root_librispeech / ab_cfg['src_id']
        tgt_dir = root_librispeech / ab_cfg['tgt_id']
        src_file = sorted(list(src_dir.rglob('*.flac')))[0]
        tgt_files = sorted(list(tgt_dir.rglob('*.flac')))[:20]
        
        transcript = ""
        for tf in src_file.parent.glob("*.trans.txt"):
            for line in tf.read_text().splitlines():
                if line.startswith(src_file.stem):
                    transcript = line[len(src_file.stem):].strip()
                    break
                    
        target_model = build_target_representation(tgt_files, wavlm, cents_shared_norm, cov_shared_k, args.device)
        xs_t, src_wav = extract_features(src_file, wavlm, args.device)
        
        # Save source & target refs
        src_path = ab_pair_dir / "source.wav"
        if not src_path.exists():
            sf.write(src_path, src_wav.numpy(), 16000)
            
        tgt_ref_path = ab_pair_dir / "target_ref.wav"
        if not tgt_ref_path.exists():
            sf.write(tgt_ref_path, target_model['ref_wav'].numpy(), 16000)
            
        samples_dict = {}
        for a in alphas:
            a_out = ab_pair_dir / f"alpha_{a:.1f}.wav"
            if not a_out.exists():
                print(f"  Synthesizing alpha={a:.1f}...")
                x_alpha = convert_lwt(xs_t, target_model, cents_shared_norm, cov_shared_k, alpha=a)
                audio_a = vocode(x_alpha, hifigan, args.device)
                sf.write(a_out, audio_a, 16000)
            samples_dict[a] = f"ablation_alpha/{ab_id}/alpha_{a:.1f}.wav"
            
        ablation_results.append({
            'id': ab_id,
            'pair_title': ab_cfg['pair_title'],
            'src_label': ab_cfg['src_label'],
            'tgt_label': ab_cfg['tgt_label'],
            'transcript': transcript,
            'rel_source': f"ablation_alpha/{ab_id}/source.wav",
            'rel_target': f"ablation_alpha/{ab_id}/target_ref.wav",
            'samples': samples_dict
        })

    print("\n[5/5] Building Complete ICASSP Demo Webpage...")
    generate_icassp_html_page(demo_results, ablation_results, out_dir)
    print("\n=======================================================")
    print(" ALL SAMPLES & ICASSP DEMO PAGE READY!")
    print(f" Local Webpage: file://{out_dir.resolve()}/index.html")
    print("=======================================================")

if __name__ == "__main__":
    main()
