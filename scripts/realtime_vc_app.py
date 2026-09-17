#!/usr/bin/env python3
"""
Real-Time & Live Streaming Voice Conversion Web Application using Boosted LWT.

Features:
  - Mode 1: Continuous Live Streaming (Voice Changer): As you speak into the microphone,
            audio is streamed via WebSocket, converted in real-time (~15ms on GPU),
            and played out through the speakers continuously with smooth cross-fading.
  - Mode 2: Sentence / Studio Mode: Record an utterance (with Auto-VAD or manual),
            convert, and compare source vs target audio side-by-side.
  - Pre-loaded LibriSpeech target speakers (male & female).
  - Microphone device selector (Built-in mic vs Headset mic).
  - Microphone Gain booster (1.0x to 5.0x) + Real-time VU-meter input bar.
  - Support for HTTP and self-signed HTTPS (--ssl).

Usage:
  uv run python scripts/realtime_vc_app.py --port 7860 --device cuda:0 [--ssl]
"""

import os
import sys
import io
import time
import json
import argparse
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F
from aiohttp import web, WSMsgType

# ---------------------------------------------------------------------------
# Audio Processing & Transport Functions
# ---------------------------------------------------------------------------

def decode_audio_bytes(raw_bytes):
    """Decode audio bytes (WebM, OGG, WAV, MP3, etc.) into 16kHz mono float32 numpy array."""
    try:
        data, sr = sf.read(io.BytesIO(raw_bytes))
        if sr != 16000:
            t = torch.tensor(data, dtype=torch.float32)
            if t.dim() > 1:
                t = t.mean(dim=-1)
            t = F.resample(t.unsqueeze(0), sr, 16000).squeeze(0)
            data = t.numpy()
        elif data.ndim > 1:
            data = data.mean(axis=-1)
        return data.astype(np.float32)
    except Exception:
        cmd = ['ffmpeg', '-i', 'pipe:0', '-f', 'wav', '-ar', '16000', '-ac', '1', 'pipe:1']
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = proc.communicate(input=raw_bytes)
        data, _ = sf.read(io.BytesIO(out))
        return data.astype(np.float32)

def project_psd(cov, eps=1e-3):
    """Project a symmetric matrix onto the cone of positive semi-definite matrices."""
    evals, evecs = torch.linalg.eigh(cov)
    evals = torch.clamp(evals, min=eps)
    return evecs @ torch.diag(evals) @ evecs.t()

def compute_bures_map_torch(cov_X, cov_Y, eps=1e-2):
    """Compute exact Monge map A = Cov_X^{-1/2} (Cov_X^{1/2} Cov_Y Cov_X^{1/2})^{1/2} Cov_X^{-1/2}."""
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

def get_wct_matrices(X, k=128):
    """Global WCT whitening and coloring matrices."""
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

# ---------------------------------------------------------------------------
# App State & Model Initialization
# ---------------------------------------------------------------------------

class VoiceConversionEngine:
    def __init__(self, device='cuda:0'):
        self.device = device
        print(f"[Engine] Initializing on device: {self.device}")
        
        print("[Engine] Loading WavLM-Large...")
        self.wavlm = torch.hub.load('bshall/knn-vc', 'wavlm_large', trust_repo=True, device=self.device).eval()
        
        print("[Engine] Loading HiFi-GAN...")
        self.hifigan, _ = torch.hub.load('bshall/knn-vc', 'hifigan_wavlm', trust_repo=True, prematched=True, device=self.device)
        self.hifigan.eval()
        
        self.root_librispeech = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
        self.target_meta = {
            '1089': {'name': 'Peter Bobbe', 'gender': 'Homme', 'desc': 'Voix masculine grave, posée et claire'},
            '121':  {'name': 'Nikolle Doolin', 'gender': 'Femme', 'desc': 'Voix féminine claire, nette et mélodieuse'},
            '237':  {'name': 'Rachelellen', 'gender': 'Femme', 'desc': 'Voix féminine chaleureuse et expressive'},
            '260':  {'name': 'Brad Bush', 'gender': 'Homme', 'desc': 'Voix masculine médium, articulée'}
        }
        
        self.load_or_build_cache()
        print("[Engine] Ready for real-time and streaming conversions!\n")

    def load_or_build_cache(self):
        cache_path = Path("output/cache/realtime_targets_4spk.pt")
        shared_cache = Path("output/cache/shared_clusters_k10.pt")
        
        # 1. Load shared background clusters
        if shared_cache.exists():
            data = torch.load(shared_cache, map_location=self.device, weights_only=False)
            self.cents_shared = data['cents_shared']
            self.cents_shared_norm = data['cents_shared_norm']
            self.cov_shared_k = data['cov_shared_k']
            print("[Engine] Loaded universal background clusters from cache.")
        else:
            raise FileNotFoundError(f"Missing background cluster cache at {shared_cache}")
            
        # 2. Load or precompute target models
        if cache_path.exists():
            print(f"[Engine] Loading target models from {cache_path}...")
            self.target_models = torch.load(cache_path, map_location=self.device, weights_only=False)
        else:
            print("[Engine] Precomputing target speaker models...")
            self.target_models = {}
            for spk_id in self.target_meta.keys():
                tgt_files = sorted(list((self.root_librispeech / spk_id).rglob('*.flac')))[:20]
                Y_list, ref_wav = [], None
                for idx, f in enumerate(tgt_files):
                    w, sr = torchaudio.load(str(f))
                    if sr != 16000: w = F.resample(w, sr, 16000)
                    if w.dim() == 2 and w.shape[0] > 1:
                        w = w.mean(dim=0, keepdim=True)
                    elif w.dim() == 1:
                        w = w.unsqueeze(0)
                    if idx == 0: ref_wav = w.squeeze().cpu()
                    with torch.no_grad():
                        feat, _ = self.wavlm.extract_features(w.to(self.device), output_layer=6)
                    Y_list.append(feat.squeeze(0))
                
                Y_t = torch.cat(Y_list, dim=0)
                Y_norm = torch.nn.functional.normalize(Y_t, dim=1)
                mu_Y, W_Y, C_Y = get_wct_matrices(Y_t, k=128)
                
                sim_Y = torch.mm(Y_norm, self.cents_shared_norm.t())
                w_Y = torch.softmax(sim_Y * 20.0, dim=1)
                
                cl_mu_Y_shared, cl_cov_Y_shared, delta_cov_speaker = [], [], []
                for k in range(10):
                    wk = w_Y[:, k:k+1]
                    mass_k = wk.sum()
                    muk = (Y_t * wk).sum(dim=0) / (mass_k + 1e-8)
                    yc = Y_t - muk.unsqueeze(0)
                    cov_Y_k = torch.mm(yc.t(), yc * wk) / (mass_k + 1e-8)
                    cl_mu_Y_shared.append(muk)
                    cl_cov_Y_shared.append(cov_Y_k)
                    delta_cov_speaker.append(cov_Y_k - self.cov_shared_k[k])
                    
                self.target_models[spk_id] = {
                    'Y_t': Y_t,
                    'Y_norm': Y_norm,
                    'mu_Y': mu_Y,
                    'C_Y': C_Y,
                    'cl_mu_Y_shared': cl_mu_Y_shared,
                    'cl_cov_Y_shared': cl_cov_Y_shared,
                    'delta_cov_speaker': delta_cov_speaker,
                    'ref_wav': ref_wav
                }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.target_models, cache_path)
            print(f"[Engine] Saved precomputed targets to {cache_path}.")

    def convert(self, audio_data, target_id='1089', method='boosted_lwt', alpha=1.5, beta=20.0):
        t0 = time.perf_counter()
        
        w = torch.tensor(audio_data, dtype=torch.float32, device=self.device)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        elif w.dim() == 2 and w.shape[0] > 1:
            w = w.mean(dim=0, keepdim=True)
            
        with torch.no_grad():
            feat, _ = self.wavlm.extract_features(w, output_layer=6)
            xs_t = feat.squeeze(0) # (Ns, 1024)
            
        tm = self.target_models[target_id]
        Ns = xs_t.shape[0]
        
        if method in ['boosted_lwt', 'lwt_baseline']:
            a = 1.0 if method == 'lwt_baseline' else alpha
            xs_n = torch.nn.functional.normalize(xs_t, dim=1)
            sim_X = torch.mm(xs_n, self.cents_shared_norm.t())
            w_X = torch.softmax(sim_X * beta, dim=1)
            
            x_hat = torch.zeros_like(xs_t)
            for k in range(10):
                wk = w_X[:, k:k+1]
                mass_k = wk.sum()
                if mass_k < 1e-4: continue
                mu_X_k = (xs_t * wk).sum(dim=0) / mass_k
                xc = xs_t - mu_X_k.unsqueeze(0)
                cov_X_k = torch.mm(xc.t(), xc * wk) / mass_k
                
                cov_target = self.cov_shared_k[k] + a * tm['delta_cov_speaker'][k]
                cov_target = project_psd(cov_target, eps=1e-3)
                
                A_k = compute_bures_map_torch(cov_X_k, cov_target, eps=1e-2)
                T_k = torch.mm(xc, A_k) + tm['cl_mu_Y_shared'][k].unsqueeze(0)
                x_hat += wk * T_k
                
        elif method == 'knn_vc':
            xs_n = torch.nn.functional.normalize(xs_t, dim=1)
            sim_mat = torch.mm(xs_n, tm['Y_norm'].t())
            topk_idx = torch.topk(sim_mat, k=4, dim=1).indices
            x_hat = torch.mean(tm['Y_t'][topk_idx], dim=1)
            
        elif method == 'wct':
            xs_np = xs_t.cpu().numpy()
            mu_X, W_X, _ = get_wct_matrices(xs_np, k=min(128, len(xs_np)-1))
            xw = (xs_np - mu_X) @ W_X
            x_hat_np = xw @ tm['C_Y'].T + tm['mu_Y']
            x_hat = torch.tensor(x_hat_np, dtype=torch.float32, device=self.device)
        else:
            raise ValueError(f"Unknown method {method}")
            
        with torch.inference_mode():
            converted_audio = self.hifigan(x_hat.unsqueeze(0)).squeeze().cpu().numpy()
            
        max_val = np.max(np.abs(converted_audio))
        if max_val > 1.0:
            converted_audio = converted_audio / max_val * 0.99
            
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        audio_dur = len(audio_data) / 16000.0
        rtf = (elapsed_ms / 1000.0) / max(0.01, audio_dur)
        
        return converted_audio, elapsed_ms, rtf

    def convert_streaming_chunk(self, new_chunk, history_buffer, prev_tail, target_id='1089', method='boosted_lwt', alpha=1.5, beta=20.0):
        """Convert continuous live microphone chunks with context window and cross-fading."""
        combined = np.concatenate([history_buffer, new_chunk])
        
        converted_full, elapsed_ms, _ = self.convert(
            combined, target_id=target_id, method=method, alpha=alpha, beta=beta
        )
        
        chunk_len = len(new_chunk)
        if len(converted_full) >= chunk_len:
            out_slice = converted_full[-chunk_len:].copy()
        else:
            out_slice = converted_full.copy()
            
        fade_len = min(320, len(out_slice), len(prev_tail))
        if fade_len > 0:
            fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
            fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)
            out_slice[:fade_len] = out_slice[:fade_len] * fade_in + prev_tail[-fade_len:] * fade_out
            
        new_history = combined[-3200:].copy()
        new_tail = out_slice[-320:].copy()
        
        return out_slice, new_history, new_tail, elapsed_ms

# ---------------------------------------------------------------------------
# HTML, CSS & JavaScript Frontend
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Conversion Live Studio | Boosted LWT</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-main: #0b0f19;
            --bg-card: #131b2e;
            --bg-box: #1c2640;
            --border: #2a3859;
            --accent: #38bdf8;
            --accent-glow: rgba(56, 189, 248, 0.35);
            --gold: #f59e0b;
            --gold-glow: rgba(245, 158, 11, 0.3);
            --green: #10b981;
            --red: #ef4444;
            --red-glow: rgba(239, 68, 68, 0.4);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-main);
            color: var(--text-main);
            line-height: 1.6;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 1.5rem 1rem;
        }

        .container { width: 100%; max-width: 960px; }

        header { text-align: center; margin-bottom: 1.5rem; }

        .pill-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(56, 189, 248, 0.12);
            color: var(--accent);
            border: 1px solid rgba(56, 189, 248, 0.3);
            padding: 0.35rem 0.9rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
        }

        .pill-badge .dot {
            width: 8px;
            height: 8px;
            background: var(--green);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--green);
        }

        h1 {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
            margin-bottom: 0.3rem;
        }

        p.subtitle {
            color: var(--text-muted);
            font-size: 1.05rem;
            max-width: 680px;
            margin: 0 auto;
        }

        .mode-tabs {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .tab-btn {
            background: var(--bg-card);
            border: 2px solid var(--border);
            color: var(--text-muted);
            padding: 0.75rem 1.5rem;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .tab-btn:hover {
            color: var(--text-main);
            border-color: var(--accent);
        }

        .tab-btn.active {
            color: #ffffff;
            background: #1e293b;
            border-color: var(--accent);
            box-shadow: 0 0 15px var(--accent-glow);
        }

        .tab-btn.active.live-tab {
            border-color: var(--red);
            box-shadow: 0 0 20px var(--red-glow);
        }

        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .targets-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .target-card {
            background: var(--bg-card);
            border: 2px solid var(--border);
            border-radius: 14px;
            padding: 1.1rem 0.9rem;
            cursor: pointer;
            transition: all 0.2s ease;
            position: relative;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .target-card:hover {
            border-color: var(--accent);
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
        }

        .target-card.selected {
            border-color: var(--gold);
            background: #172038;
            box-shadow: 0 0 20px var(--gold-glow);
        }

        .target-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.5rem;
        }

        .avatar {
            font-size: 1.8rem;
            background: var(--bg-box);
            width: 44px;
            height: 44px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid var(--border);
        }

        .spk-info h3 {
            font-size: 1.05rem;
            font-weight: 700;
        }

        .spk-id {
            font-size: 0.75rem;
            color: var(--accent);
            font-weight: 600;
        }

        .spk-desc {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-bottom: 0.75rem;
            min-height: 2.4rem;
        }

        .btn-preview {
            background: var(--bg-box);
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 0.35rem 0.6rem;
            border-radius: 8px;
            font-size: 0.75rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.4rem;
            transition: background 0.2s;
        }

        .btn-preview:hover { background: #2a3859; }

        .alert-headphones {
            background: rgba(245, 158, 11, 0.12);
            border: 1px solid rgba(245, 158, 11, 0.35);
            color: #fde68a;
            border-radius: 10px;
            padding: 0.75rem 1rem;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 1.5rem;
        }

        /* Mic & Input Settings Bar */
        .mic-settings-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            background: var(--bg-box);
            padding: 0.75rem 1.25rem;
            border-radius: 12px;
            border: 1px solid var(--border);
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
        }

        .mic-select-group, .gain-group, .vu-meter-group {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .vu-meter-bar {
            width: 110px;
            height: 12px;
            background: #0b0f19;
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid var(--border);
        }

        .vu-meter-fill {
            width: 0%;
            height: 100%;
            background: linear-gradient(90deg, #10b981 0%, #f59e0b 70%, #ef4444 100%);
            transition: width 0.05s ease;
        }

        .studio-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.75rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
            text-align: center;
            position: relative;
        }

        .visualizer-container {
            width: 100%;
            height: 95px;
            background: rgba(11, 15, 25, 0.7);
            border-radius: 12px;
            margin-bottom: 1.5rem;
            border: 1px solid var(--border);
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        canvas#oscilloscope { width: 100%; height: 100%; }

        .controls-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1.5rem;
            flex-wrap: wrap;
        }

        .btn-live-stream {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: #ffffff;
            border: none;
            padding: 1.1rem 2.8rem;
            border-radius: 9999px;
            font-size: 1.2rem;
            font-weight: 800;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
            transition: all 0.2s ease;
        }

        .btn-live-stream:hover {
            transform: scale(1.03);
            box-shadow: 0 6px 25px rgba(16, 185, 129, 0.5);
        }

        .btn-live-stream.active {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            box-shadow: 0 0 30px rgba(239, 68, 68, 0.7);
            animation: pulse-live 1.2s infinite;
        }

        @keyframes pulse-live {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.04); }
        }

        .btn-record {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
            color: #ffffff;
            border: none;
            padding: 1.1rem 2.5rem;
            border-radius: 9999px;
            font-size: 1.15rem;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            box-shadow: 0 4px 15px var(--accent-glow);
            transition: all 0.2s ease;
        }

        .btn-record:hover {
            transform: scale(1.03);
            box-shadow: 0 6px 25px var(--accent-glow);
        }

        .btn-record.recording {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            box-shadow: 0 0 25px rgba(239, 68, 68, 0.6);
            animation: pulse-live 1.5s infinite;
        }

        .vad-toggle {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            background: var(--bg-box);
            padding: 0.6rem 1rem;
            border-radius: 10px;
            border: 1px solid var(--border);
            font-size: 0.85rem;
            cursor: pointer;
        }

        .vad-toggle input { accent-color: var(--accent); cursor: pointer; width: 18px; height: 18px; }

        .status-message {
            margin-top: 1.25rem;
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-muted);
            min-height: 1.5rem;
        }

        .status-message.live-on { color: #34d399; }
        .status-message.converting { color: var(--accent); }
        .status-message.playing { color: var(--gold); }

        .results-section {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            display: none;
        }

        .results-section.show { display: block; }

        .results-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-top: 1rem;
        }

        @media (max-width: 650px) {
            .results-grid { grid-template-columns: 1fr; }
        }

        .audio-player-box {
            background: var(--bg-box);
            padding: 1.25rem;
            border-radius: 12px;
            border: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .audio-player-box.target-box {
            border: 2px solid var(--gold);
            box-shadow: 0 0 15px var(--gold-glow);
        }

        .box-title {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 700;
            font-size: 0.95rem;
        }

        audio { width: 100%; height: 40px; border-radius: 8px; }

        .perf-badge {
            background: #0284c7;
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
        }

        details.settings-details {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin-bottom: 2rem;
        }

        details summary {
            font-weight: 700;
            font-size: 0.95rem;
            cursor: pointer;
            color: var(--text-muted);
            user-select: none;
        }

        details summary:hover { color: var(--text-main); }

        .settings-content {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1.25rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
        }

        .setting-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .setting-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
        }

        select, input[type="range"] {
            background: var(--bg-box);
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 0.5rem;
            border-radius: 8px;
            outline: none;
        }

        footer {
            margin-top: auto;
            color: var(--text-muted);
            font-size: 0.85rem;
            text-align: center;
            padding: 1rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="pill-badge">
                <div class="dot"></div>
                <span>Moteur Boosted LWT &bull; Inférence GPU ~15ms</span>
            </div>
            <h1>Studio de Conversion Vocale</h1>
            <p class="subtitle">
                Transformez votre voix en direct vers les locuteurs de LibriSpeech avec le transport optimal de Bures-Wasserstein.
            </p>
        </header>

        <!-- Mode Tabs -->
        <div class="mode-tabs">
            <button class="tab-btn live-tab active" id="tab-live" onclick="switchMode('live')">
                <span>🔴</span> Streaming Temps Réel (Live)
            </button>
            <button class="tab-btn" id="tab-sentence" onclick="switchMode('sentence')">
                <span>🎙️</span> Mode Phrase (Studio)
            </button>
        </div>

        <!-- 1. Selection Cibles -->
        <div class="section-title">
            <span>🎯 Voix Cible Sélectionnée</span>
        </div>
        <div class="targets-grid" id="targets-container"></div>

        <!-- Headphone Warning for Live -->
        <div class="alert-headphones" id="alert-headphones">
            <span>🎧</span>
            <span><strong>Conseil Casque :</strong> Si vous utilisez des écouteurs sans micro intégré, vérifiez ci-dessous que la source micro est bien votre <em>Microphone Intégré PC</em> !</span>
        </div>

        <!-- Microphone Settings Bar (Selector + Gain + VU-Meter) -->
        <div class="mic-settings-bar">
            <div class="mic-select-group">
                <label for="sel-audio-device">🎙️ Source Micro :</label>
                <select id="sel-audio-device" onchange="onDeviceChanged()"></select>
            </div>
            <div class="gain-group">
                <label for="rng-gain">🔊 Amplification : <span id="lbl-gain">2.0x</span></label>
                <input type="range" id="rng-gain" min="1.0" max="6.0" step="0.5" value="2.0" oninput="updateGain(this.value)">
            </div>
            <div class="vu-meter-group">
                <span>Niveau :</span>
                <div class="vu-meter-bar" title="Niveau sonore capté par le micro">
                    <div id="vu-meter-fill" class="vu-meter-fill"></div>
                </div>
            </div>
        </div>

        <!-- 2. Studio d'enregistrement -->
        <div class="studio-card">
            <div class="visualizer-container">
                <canvas id="oscilloscope"></canvas>
            </div>

            <!-- CONTROLS: LIVE STREAMING MODE -->
            <div id="controls-live" class="controls-row">
                <button class="btn-live-stream" id="btn-live-toggle">
                    <span id="live-icon">🔴</span>
                    <span id="live-text">Activer le Streaming Live</span>
                </button>
            </div>

            <!-- CONTROLS: SENTENCE MODE -->
            <div id="controls-sentence" class="controls-row" style="display: none;">
                <button class="btn-record" id="btn-record">
                    <span id="record-icon">🎙️</span>
                    <span id="record-text">Commencer à parler</span>
                </button>

                <label class="vad-toggle" title="Arrête et convertit dès que vous faites une pause">
                    <input type="checkbox" id="chk-vad" checked>
                    <span>Auto VAD (Mains libres)</span>
                </label>
            </div>

            <div class="status-message" id="status-message">Prêt. Cliquez sur le bouton pour démarrer le streaming live.</div>
        </div>

        <!-- 3. Résultats (Mode Phrase) -->
        <div class="results-section" id="results-section">
            <div class="section-title">
                <span>🔊 Résultat de la Conversion</span>
                <span class="perf-badge" id="latency-badge">⚡ 45 ms</span>
            </div>
            <div class="results-grid">
                <div class="audio-player-box">
                    <div class="box-title">
                        <span>Votre voix (Source)</span>
                    </div>
                    <audio id="audio-source" controls></audio>
                </div>
                <div class="audio-player-box target-box">
                    <div class="box-title">
                        <span id="target-title-label">Voix Convertie (Cible)</span>
                        <span style="color: var(--gold); font-size: 0.8rem;">★ Boosted LWT</span>
                    </div>
                    <audio id="audio-target" controls autoplay></audio>
                </div>
            </div>
        </div>

        <!-- 4. Paramètres avancés -->
        <details class="settings-details">
            <summary>⚙️ Paramètres Avancés (Algorithme & Boost)</summary>
            <div class="settings-content">
                <div class="setting-group">
                    <label for="sel-method">Algorithme de Transport :</label>
                    <select id="sel-method">
                        <option value="boosted_lwt" selected>Boosted LWT (alpha=1.5 - Recommandé)</option>
                        <option value="lwt_baseline">Baseline LWT (alpha=1.0)</option>
                        <option value="knn_vc">kNN-VC (k=4)</option>
                        <option value="wct">Classic WCT Global</option>
                    </select>
                </div>
                <div class="setting-group">
                    <label for="rng-alpha">Boost Factor &alpha; : <span id="lbl-alpha">1.5</span></label>
                    <input type="range" id="rng-alpha" min="0.5" max="2.0" step="0.1" value="1.5">
                </div>
                <div class="setting-group">
                    <label for="rng-beta">Température &beta; : <span id="lbl-beta">20.0</span></label>
                    <input type="range" id="rng-beta" min="5.0" max="40.0" step="5.0" value="20.0">
                </div>
            </div>
        </details>

        <footer>
            Projet Voice Conversion &bull; Transport de Bures-Wasserstein Local (LWT) &bull; Streaming WebSockets 16 kHz
        </footer>
    </div>

    <script>
        let currentMode = 'live';
        let selectedTarget = '1089';
        let isLiveActive = false;
        let isRecording = false;

        let audioCtx = null;
        let micStream = null;
        let micSource = null;
        let gainNode = null;
        let scriptNode = null;
        let analyser = null;
        let ws = null;
        let nextPlayTime = 0;
        let animationId = null;

        let mediaRecorder = null;
        let audioChunks = [];
        let vadInterval = null;
        let lastVoiceTime = 0;

        const targets = [
            { id: '1089', name: 'Peter Bobbe', gender: 'Homme', emoji: '👨', desc: 'Voix masculine grave, posée et claire' },
            { id: '121',  name: 'Nikolle Doolin', gender: 'Femme', emoji: '👩', desc: 'Voix féminine claire, nette et mélodieuse' },
            { id: '237',  name: 'Rachelellen', gender: 'Femme', emoji: '👩', desc: 'Voix féminine chaleureuse et expressive' },
            { id: '260',  name: 'Brad Bush', gender: 'Homme', emoji: '👨', desc: 'Voix masculine médium, articulée' }
        ];

        const container = document.getElementById('targets-container');
        targets.forEach(t => {
            const card = document.createElement('div');
            card.className = `target-card ${t.id === selectedTarget ? 'selected' : ''}`;
            card.dataset.id = t.id;
            card.innerHTML = `
                <div>
                    <div class="target-header">
                        <div class="avatar">${t.emoji}</div>
                        <div class="spk-info">
                            <h3>${t.name}</h3>
                            <span class="spk-id">Locuteur #${t.id} (${t.gender})</span>
                        </div>
                    </div>
                    <p class="spk-desc">${t.desc}</p>
                </div>
                <button class="btn-preview" onclick="playPreview('${t.id}', event)">
                    <span>🔊</span> Écouter extrait
                </button>
            `;
            card.onclick = () => selectTarget(t.id);
            container.appendChild(card);
        });

        function selectTarget(id) {
            selectedTarget = id;
            document.querySelectorAll('.target-card').forEach(c => {
                c.classList.toggle('selected', c.dataset.id === id);
            });
            const t = targets.find(x => x.id === id);
            document.getElementById('target-title-label').innerText = `Voix Convertie (${t.name} - ${t.gender})`;
            sendConfig();
        }

        function playPreview(spkId, evt) {
            evt.stopPropagation();
            new Audio(`/api/target_preview/${spkId}`).play();
        }

        function switchMode(mode) {
            if (isLiveActive) stopLiveStreaming();
            if (isRecording) stopRecordingAndConvert();

            currentMode = mode;
            document.getElementById('tab-live').classList.toggle('active', mode === 'live');
            document.getElementById('tab-sentence').classList.toggle('active', mode === 'sentence');
            document.getElementById('controls-live').style.display = mode === 'live' ? 'flex' : 'none';
            document.getElementById('controls-sentence').style.display = mode === 'sentence' ? 'flex' : 'none';
            document.getElementById('alert-headphones').style.display = mode === 'live' ? 'flex' : 'none';

            const statusMsg = document.getElementById('status-message');
            if (mode === 'live') {
                statusMsg.className = 'status-message';
                statusMsg.innerText = 'Prêt. Cliquez sur "Activer le Streaming Live" pour parler en temps réel.';
            } else {
                statusMsg.className = 'status-message';
                statusMsg.innerText = 'Prêt. Appuyez sur "Commencer à parler" ou la barre Espace.';
            }
        }

        // Device Enumeration
        async function populateAudioDevices() {
            try {
                const devices = await navigator.mediaDevices.enumerateDevices();
                const select = document.getElementById('sel-audio-device');
                const audioInputs = devices.filter(d => d.kind === 'audioinput');
                const currentVal = select.value;
                select.innerHTML = '';
                audioInputs.forEach((d, idx) => {
                    const opt = document.createElement('option');
                    opt.value = d.deviceId;
                    opt.innerText = d.label || `Microphone ${idx + 1}`;
                    if (d.deviceId === currentVal) opt.selected = true;
                    select.appendChild(opt);
                });
            } catch (e) {
                console.error('enumerateDevices:', e);
            }
        }
        navigator.mediaDevices.ondevicechange = populateAudioDevices;
        populateAudioDevices();

        function updateGain(val) {
            document.getElementById('lbl-gain').innerText = val + 'x';
            if (gainNode) gainNode.gain.value = parseFloat(val);
        }

        async function onDeviceChanged() {
            if (isLiveActive) {
                stopLiveStreaming();
                await startLiveStreaming();
            }
        }

        document.getElementById('rng-alpha').oninput = e => {
            document.getElementById('lbl-alpha').innerText = e.target.value;
            sendConfig();
        };
        document.getElementById('rng-beta').oninput = e => {
            document.getElementById('lbl-beta').innerText = e.target.value;
            sendConfig();
        };
        document.getElementById('sel-method').onchange = () => sendConfig();

        function sendConfig() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'config',
                    target_id: selectedTarget,
                    method: document.getElementById('sel-method').value,
                    alpha: parseFloat(document.getElementById('rng-alpha').value),
                    beta: parseFloat(document.getElementById('rng-beta').value)
                }));
            }
        }

        // Visualizer
        const canvas = document.getElementById('oscilloscope');
        const canvasCtx = canvas.getContext('2d');
        function drawVisualizer() {
            if (!analyser) return;
            const bufferLength = analyser.fftSize;
            const dataArray = new Uint8Array(bufferLength);
            analyser.getByteTimeDomainData(dataArray);

            canvasCtx.fillStyle = 'rgba(11, 15, 25, 0.35)';
            canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
            canvasCtx.lineWidth = 2.5;
            canvasCtx.strokeStyle = (isLiveActive || isRecording) ? '#38bdf8' : '#475569';
            canvasCtx.beginPath();

            const sliceWidth = canvas.width * 1.0 / bufferLength;
            let x = 0;
            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = v * canvas.height / 2;
                if (i === 0) canvasCtx.moveTo(x, y);
                else canvasCtx.lineTo(x, y);
                x += sliceWidth;
            }
            canvasCtx.lineTo(canvas.width, canvas.height / 2);
            canvasCtx.stroke();

            animationId = requestAnimationFrame(drawVisualizer);
        }

        // -------------------------------------------------------------
        // LIVE STREAMING
        // -------------------------------------------------------------
        const btnLive = document.getElementById('btn-live-toggle');
        btnLive.onclick = toggleLiveStreaming;

        async function toggleLiveStreaming() {
            if (!isLiveActive) await startLiveStreaming();
            else stopLiveStreaming();
        }

        async function startLiveStreaming() {
            try {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                if (audioCtx.state === 'suspended') await audioCtx.resume();

                const deviceId = document.getElementById('sel-audio-device').value;
                const audioConstraints = {
                    sampleRate: 16000,
                    channelCount: 1,
                    autoGainControl: true,
                    echoCancellation: false,
                    noiseSuppression: false
                };
                if (deviceId) audioConstraints.deviceId = { exact: deviceId };

                micStream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
                await populateAudioDevices();

                analyser = audioCtx.createAnalyser();
                analyser.fftSize = 1024;
                micSource = audioCtx.createMediaStreamSource(micStream);

                // Pre-amp gain node
                gainNode = audioCtx.createGain();
                gainNode.gain.value = parseFloat(document.getElementById('rng-gain').value);

                micSource.connect(gainNode);
                gainNode.connect(analyser);

                canvas.width = canvas.parentElement.clientWidth;
                canvas.height = canvas.parentElement.clientHeight;
                drawVisualizer();

                const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${proto}//${window.location.host}/ws/stream`);
                ws.binaryType = 'arraybuffer';

                ws.onopen = () => {
                    console.log('[WebSocket] Connected');
                    sendConfig();
                };

                ws.onmessage = (event) => {
                    if (event.data instanceof ArrayBuffer) {
                        const int16 = new Int16Array(event.data);
                        const float32 = new Float32Array(int16.length);
                        for (let i = 0; i < int16.length; i++) {
                            float32[i] = int16[i] / 32768.0;
                        }
                        queueAudioChunk(float32);
                    }
                };

                ws.onerror = (e) => console.error('[WebSocket] Error', e);
                ws.onclose = () => { if (isLiveActive) stopLiveStreaming(); };

                scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);
                gainNode.connect(scriptNode);
                scriptNode.connect(audioCtx.destination);

                scriptNode.onaudioprocess = (e) => {
                    if (!isLiveActive || !ws || ws.readyState !== WebSocket.OPEN) return;
                    const inputChannel = e.inputBuffer.getChannelData(0);

                    let sum = 0;
                    for (let i = 0; i < inputChannel.length; i++) sum += inputChannel[i] * inputChannel[i];
                    const rms = Math.sqrt(sum / inputChannel.length);

                    // Update live VU-Meter bar
                    const pct = Math.min(100, Math.round(rms * 500));
                    const vuBar = document.getElementById('vu-meter-fill');
                    if (vuBar) vuBar.style.width = pct + '%';

                    // Ultra-sensitive gate: even gentle voice passes through
                    if (rms > 0.001) {
                        const pcm16 = new Int16Array(inputChannel.length);
                        for (let i = 0; i < inputChannel.length; i++) {
                            pcm16[i] = Math.max(-32768, Math.min(32767, inputChannel[i] * 32767));
                        }
                        ws.send(pcm16.buffer);
                    }
                };

                isLiveActive = true;
                btnLive.classList.add('active');
                document.getElementById('live-icon').innerText = '⏹️';
                document.getElementById('live-text').innerText = 'Désactiver le Streaming';
                const statusMsg = document.getElementById('status-message');
                statusMsg.className = 'status-message live-on';
                statusMsg.innerText = '🔴 STREAMING LIVE EN COURS : Parlez, votre voix transformée sort en direct !';

            } catch (err) {
                console.error(err);
                alert("Erreur micro : " + err.message);
            }
        }

        function stopLiveStreaming() {
            if (!isLiveActive) return;
            isLiveActive = false;

            if (scriptNode) { scriptNode.disconnect(); scriptNode = null; }
            if (gainNode) { gainNode.disconnect(); gainNode = null; }
            if (micSource) { micSource.disconnect(); micSource = null; }
            if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
            if (ws) { ws.close(); ws = null; }

            const vuBar = document.getElementById('vu-meter-fill');
            if (vuBar) vuBar.style.width = '0%';

            btnLive.classList.remove('active');
            document.getElementById('live-icon').innerText = '🔴';
            document.getElementById('live-text').innerText = 'Activer le Streaming Live';
            const statusMsg = document.getElementById('status-message');
            statusMsg.className = 'status-message';
            statusMsg.innerText = 'Streaming arrêté. Prêt à reprendre.';
        }

        function queueAudioChunk(float32Data) {
            if (!audioCtx) return;
            const audioBuffer = audioCtx.createBuffer(1, float32Data.length, 16000);
            audioBuffer.getChannelData(0).set(float32Data);

            const source = audioCtx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioCtx.destination);

            const currentTime = audioCtx.currentTime;
            if (nextPlayTime < currentTime) {
                nextPlayTime = currentTime + 0.04;
            }
            source.start(nextPlayTime);
            nextPlayTime += audioBuffer.duration;
        }

        // -------------------------------------------------------------
        // SENTENCE MODE
        // -------------------------------------------------------------
        const btnRecord = document.getElementById('btn-record');
        btnRecord.onclick = toggleSentenceRecording;

        window.addEventListener('keydown', e => {
            if (e.code === 'Space' && e.target === document.body && currentMode === 'sentence') {
                e.preventDefault();
                toggleSentenceRecording();
            }
        });

        async function toggleSentenceRecording() {
            if (!isRecording) await startRecording();
            else stopRecordingAndConvert();
        }

        async function startRecording() {
            try {
                const deviceId = document.getElementById('sel-audio-device').value;
                const audioConstraints = { sampleRate: 16000, channelCount: 1, autoGainControl: true, echoCancellation: false };
                if (deviceId) audioConstraints.deviceId = { exact: deviceId };

                micStream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
                await populateAudioDevices();

                audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                analyser = audioCtx.createAnalyser();
                analyser.fftSize = 1024;
                micSource = audioCtx.createMediaStreamSource(micStream);

                gainNode = audioCtx.createGain();
                gainNode.gain.value = parseFloat(document.getElementById('rng-gain').value);
                micSource.connect(gainNode);
                gainNode.connect(analyser);

                canvas.width = canvas.parentElement.clientWidth;
                canvas.height = canvas.parentElement.clientHeight;
                drawVisualizer();

                mediaRecorder = new MediaRecorder(micStream);
                audioChunks = [];
                mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
                mediaRecorder.start(100);

                isRecording = true;
                btnRecord.classList.add('recording');
                document.getElementById('record-icon').innerText = '⏹️';
                document.getElementById('record-text').innerText = 'Arrêter & Convertir';
                const statusMsg = document.getElementById('status-message');
                statusMsg.className = 'status-message';
                statusMsg.innerText = '🔴 Enregistrement en cours... Parlez !';

                // VU Meter & VAD in sentence mode
                const pcmData = new Uint8Array(analyser.fftSize);
                lastVoiceTime = Date.now();
                vadInterval = setInterval(() => {
                    analyser.getByteTimeDomainData(pcmData);
                    let sum = 0;
                    for (let i = 0; i < pcmData.length; i++) {
                        const val = (pcmData[i] - 128) / 128;
                        sum += val * val;
                    }
                    const rms = Math.sqrt(sum / pcmData.length);
                    const pct = Math.min(100, Math.round(rms * 500));
                    const vuBar = document.getElementById('vu-meter-fill');
                    if (vuBar) vuBar.style.width = pct + '%';

                    if (document.getElementById('chk-vad').checked) {
                        if (rms > 0.005) {
                            lastVoiceTime = Date.now();
                        } else {
                            if (Date.now() - lastVoiceTime > 900 && audioChunks.length > 8) {
                                stopRecordingAndConvert();
                            }
                        }
                    }
                }, 80);

            } catch (err) {
                console.error(err);
                alert("Erreur micro : " + err.message);
            }
        }

        function stopRecordingAndConvert() {
            if (!isRecording) return;
            isRecording = false;
            clearInterval(vadInterval);

            const vuBar = document.getElementById('vu-meter-fill');
            if (vuBar) vuBar.style.width = '0%';

            btnRecord.classList.remove('recording');
            document.getElementById('record-icon').innerText = '🎙️';
            document.getElementById('record-text').innerText = 'Commencer à parler';
            const statusMsg = document.getElementById('status-message');
            statusMsg.className = 'status-message converting';
            statusMsg.innerText = '⚡ Conversion en cours sur le GPU...';

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                document.getElementById('audio-source').src = URL.createObjectURL(audioBlob);
                document.getElementById('results-section').classList.add('show');

                const formData = new FormData();
                formData.append('audio', audioBlob, 'mic.webm');
                formData.append('target_id', selectedTarget);
                formData.append('method', document.getElementById('sel-method').value);
                formData.append('alpha', document.getElementById('rng-alpha').value);
                formData.append('beta', document.getElementById('rng-beta').value);

                try {
                    const res = await fetch('/api/convert', { method: 'POST', body: formData });
                    if (!res.ok) throw new Error("Erreur serveur");

                    const latency = res.headers.get('X-Inference-Time-Ms') || '45';
                    const rtf = res.headers.get('X-RTF') || '0.02';
                    document.getElementById('latency-badge').innerText = `⚡ ${parseFloat(latency).toFixed(0)} ms (RTF: ${parseFloat(rtf).toFixed(3)}x)`;

                    const blob = await res.blob();
                    const targetAudio = document.getElementById('audio-target');
                    targetAudio.src = URL.createObjectURL(blob);
                    
                    statusMsg.className = 'status-message playing';
                    statusMsg.innerText = '🔊 Lecture automatique de la voix cible !';
                    targetAudio.play();

                } catch (e) {
                    console.error(e);
                    statusMsg.className = 'status-message';
                    statusMsg.innerText = '❌ Erreur : ' + e.message;
                }
            };

            mediaRecorder.stop();
        }
    </script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# HTTP & WebSocket Handlers
# ---------------------------------------------------------------------------

async def handle_index(request):
    return web.Response(text=INDEX_HTML, content_type='text/html')

async def handle_target_preview(request):
    spk_id = request.match_info['spk_id']
    engine = request.app['engine']
    if spk_id not in engine.target_models:
        return web.Response(status=404, text="Speaker not found")
        
    try:
        # Check if authentic FLAC exists on disk
        tgt_files = sorted(list((engine.root_librispeech / spk_id).rglob('*.flac')))
        if tgt_files:
            data, sr = sf.read(str(tgt_files[0]))
            buf = io.BytesIO()
            sf.write(buf, data, sr, format='WAV')
            return web.Response(body=buf.getvalue(), content_type='audio/wav')
    except Exception as e:
        print(f"[Preview] Disk read error: {e}")

    # Fallback to cached ref_wav tensor
    wav_tensor = engine.target_models[spk_id].get('ref_wav')
    if wav_tensor is not None:
        if hasattr(wav_tensor, 'detach'):
            wav_tensor = wav_tensor.detach()
        if hasattr(wav_tensor, 'cpu'):
            wav_tensor = wav_tensor.cpu()
        wav_np = wav_tensor.numpy()
        buf = io.BytesIO()
        sf.write(buf, wav_np, 16000, format='WAV')
        return web.Response(body=buf.getvalue(), content_type='audio/wav')

    return web.Response(status=404, text="No sample found for this speaker")

async def handle_convert(request):
    engine = request.app['engine']
    reader = await request.multipart()
    
    raw_audio = None
    target_id = '1089'
    method = 'boosted_lwt'
    alpha = 1.5
    beta = 20.0
    
    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == 'audio':
            raw_audio = await part.read()
        elif part.name == 'target_id':
            target_id = (await part.read()).decode('utf-8')
        elif part.name == 'method':
            method = (await part.read()).decode('utf-8')
        elif part.name == 'alpha':
            alpha = float((await part.read()).decode('utf-8'))
        elif part.name == 'beta':
            beta = float((await part.read()).decode('utf-8'))
            
    if raw_audio is None or len(raw_audio) < 100:
        return web.Response(status=400, text="No audio data received")
        
    pcm = decode_audio_bytes(raw_audio)
    if len(pcm) < 1600:
        return web.Response(status=400, text="Audio too short")
        
    converted_audio, elapsed_ms, rtf = engine.convert(pcm, target_id=target_id, method=method, alpha=alpha, beta=beta)
    
    out_buf = io.BytesIO()
    sf.write(out_buf, converted_audio, 16000, format='WAV')
    out_bytes = out_buf.getvalue()
    
    headers = {
        'X-Inference-Time-Ms': f"{elapsed_ms:.1f}",
        'X-RTF': f"{rtf:.4f}",
        'Access-Control-Allow-Origin': '*'
    }
    return web.Response(body=out_bytes, content_type='audio/wav', headers=headers)

async def handle_websocket_stream(request):
    """Handles real-time low-latency WebSocket audio streaming."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    engine = request.app['engine']
    session = {
        'target_id': '1089',
        'method': 'boosted_lwt',
        'alpha': 1.5,
        'beta': 20.0,
        'history': np.zeros(3200, dtype=np.float32),
        'prev_tail': np.zeros(320, dtype=np.float32)
    }
    
    async for msg in ws:
        if msg.type == WSMsgType.TEXT:
            try:
                data = json.loads(msg.data)
                if data.get('type') == 'config':
                    session['target_id'] = data.get('target_id', session['target_id'])
                    session['method'] = data.get('method', session['method'])
                    session['alpha'] = data.get('alpha', session['alpha'])
                    session['beta'] = data.get('beta', session['beta'])
            except Exception as e:
                print(f"[WS Config Error] {e}")
                
        elif msg.type == WSMsgType.BINARY:
            try:
                pcm16 = np.frombuffer(msg.data, dtype=np.int16)
                if len(pcm16) < 640:
                    continue
                new_chunk = (pcm16 / 32768.0).astype(np.float32)
                
                out_slice, session['history'], session['prev_tail'], _ = engine.convert_streaming_chunk(
                    new_chunk,
                    session['history'],
                    session['prev_tail'],
                    target_id=session['target_id'],
                    method=session['method'],
                    alpha=session['alpha'],
                    beta=session['beta']
                )
                
                out_pcm16 = (np.clip(out_slice, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                await ws.send_bytes(out_pcm16)
                
            except Exception as e:
                print(f"[WS Audio Stream Error] {e}")
                
        elif msg.type == WSMsgType.ERROR:
            print(f"[WS Closed with exception] {ws.exception()}")
            
    return ws

# ---------------------------------------------------------------------------
# Main Server Runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Real-Time & Streaming VC Studio")
    parser.add_argument("--port", type=int, default=7860, help="Web server port (default: 7860)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ssl", action="store_true", help="Enable HTTPS with self-signed certificate")
    args = parser.parse_args()

    engine = VoiceConversionEngine(device=args.device)
    
    app = web.Application(client_max_size=50*1024*1024)
    app['engine'] = engine
    
    app.router.add_get('/', handle_index)
    app.router.add_get('/api/target_preview/{spk_id}', handle_target_preview)
    app.router.add_post('/api/convert', handle_convert)
    app.router.add_get('/ws/stream', handle_websocket_stream)
    
    ssl_context = None
    proto = "http"
    if args.ssl:
        import ssl
        cert_file = Path("output/cache/cert.pem")
        key_file = Path("output/cache/key.pem")
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(str(cert_file), str(key_file))
        proto = "https"
    
    print("\n" + "="*65)
    print(f"🚀 VOICE CONVERSION LIVE STREAMING STUDIO IS READY!")
    print(f"👉 Local : {proto}://localhost:{args.port}")
    print(f"👉 Réseau LAN : {proto}://194.199.23.26:{args.port}")
    print("="*65 + "\n")
    
    web.run_app(app, host='0.0.0.0', port=args.port, ssl_context=ssl_context)

if __name__ == '__main__':
    main()
