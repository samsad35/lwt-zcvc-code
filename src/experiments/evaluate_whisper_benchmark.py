#!/usr/bin/env python3
"""
Evaluate Character Error Rate (CER) and Word Error Rate (WER) across all 11 Voice Conversion models
on the Grand Unified Benchmark (N=200 conversions) using openai/whisper-small with BasicTextNormalizer.

Evaluates against both:
  1. Ground-truth LibriSpeech test-clean text (.trans.txt)
  2. Source audio Whisper-small transcription

Updates:
  - output/tables/unified_200_conversions_detail.csv (adds CER_Whisper, WER_Whisper, CER_Whisper_src, WER_Whisper_src)
  - output/tables/unified_200_benchmark_summary.csv (adds mean and std for the 4 Whisper metrics)
  - output/tables/tab_overall_unified_200.tex
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
import librosa
from einops import rearrange
from sklearn.cluster import KMeans
import jiwer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# Paths
proj_root = Path(__file__).resolve().parent.parent.parent
freevc_dir = proj_root / 'src/baselines/freevc'
soundstorm_dir = proj_root / 'src/baselines/soundstorm'

sys.path.insert(0, str(proj_root))
sys.path.insert(0, str(freevc_dir))
sys.path.insert(0, str(soundstorm_dir))

from src.core.models import get_wct
import utils as freevc_utils
from models import SynthesizerTrn
from speaker_encoder.voice_encoder import SpeakerEncoder as FreeVCSpeakerEncoder
from mel_processing import mel_spectrogram_torch
from transformers import WavLMModel

from soundstorm_speechtokenizer import SoundStorm, ConformerWrapper
from speechtokenizer import SpeechTokenizer

device = torch.device('cuda:0')
print(f"Using device: {device}", flush=True)

# Helper functions
def project_psd(M, eps=1e-3):
    evals, evecs = torch.linalg.eigh(M)
    evals = torch.clamp(evals, min=eps)
    return evecs @ torch.diag(evals) @ evecs.t()

def compute_bures_map_torch(cov_X, cov_Y, eps=1e-2):
    D = cov_X.shape[0]
    I_D = torch.eye(D, device=cov_X.device, dtype=cov_X.dtype)
    cX = cov_X + eps * I_D
    cY = cov_Y + eps * I_D
    
    evals_X, evecs_X = torch.linalg.eigh(cX)
    evals_X = evals_X.clamp(min=eps)
    cX_half = evecs_X @ torch.diag(torch.sqrt(evals_X)) @ evecs_X.t()
    cX_inv_half = evecs_X @ torch.diag(1.0 / torch.sqrt(evals_X)) @ evecs_X.t()
    
    M = cX_half @ cY @ cX_half
    evals_M, evecs_M = torch.linalg.eigh(M)
    evals_M = evals_M.clamp(min=eps**2)
    M_half = evecs_M @ torch.diag(torch.sqrt(evals_M)) @ evecs_M.t()
    
    A = cX_inv_half @ M_half @ cX_inv_half
    return A

# 1. Load Whisper Model & Normalizer
print("\n[1/6] Loading openai/whisper-small and text normalizer...", flush=True)
whisper_processor = WhisperProcessor.from_pretrained('openai/whisper-small')
whisper_model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small').to(device).eval()
normalizer = BasicTextNormalizer()

def transcribe_whisper(w_1d):
    """Transcribe 16kHz audio tensor using Whisper-small."""
    if torch.is_tensor(w_1d):
        audio_np = w_1d.squeeze().cpu().numpy()
    else:
        audio_np = np.squeeze(w_1d)
    
    inp_feat = whisper_processor(audio_np, sampling_rate=16000, return_tensors='pt').input_features.to(device)
    with torch.no_grad():
        pred_ids = whisper_model.generate(inp_feat, language='english')
    raw_text = whisper_processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    return normalizer(raw_text)

# 2. Load Vocoder & Content Feature Extractor
print("\n[2/6] Loading HiFi-GAN and WavLM-Large for feature VC...", flush=True)
knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
vocoder = knn_vc.hifigan.to(device).eval()
wavlm_large = knn_vc.wavlm.to(device).eval()

def voc(f):
    with torch.inference_mode():
        if isinstance(f, np.ndarray):
            f = torch.tensor(f, dtype=torch.float32, device=device)
        return vocoder(f.unsqueeze(0)).squeeze().cpu()

def ext(p):
    w, sr = torchaudio.load(str(p))
    w = w.to(device)
    if sr != 16000: w = AF.resample(w, sr, 16000)
    if w.dim() == 1: w = w.unsqueeze(0)
    elif w.dim() == 2 and w.shape[0] > 1: w = w.mean(dim=0, keepdim=True)
    with torch.no_grad():
        feat, _ = wavlm_large.extract_features(w, output_layer=6)
    return feat.squeeze(0).cpu().numpy(), w.squeeze().cpu()

# 3. Load FreeVC & FreeVC-s
print("\n[3/6] Loading FreeVC models...", flush=True)
hps_freevc = freevc_utils.get_hparams_from_file(str(freevc_dir / 'configs/freevc.json'))
freevc_net = SynthesizerTrn(
    hps_freevc.data.filter_length // 2 + 1,
    hps_freevc.train.segment_size // hps_freevc.data.hop_length,
    **hps_freevc.model
).to(device).eval()
freevc_utils.load_checkpoint(str(freevc_dir / 'checkpoints/freevc.pth'), freevc_net, None)

hps_freevc_s = freevc_utils.get_hparams_from_file(str(freevc_dir / 'configs/freevc-s.json'))
freevc_s_net = SynthesizerTrn(
    hps_freevc_s.data.filter_length // 2 + 1,
    hps_freevc_s.train.segment_size // hps_freevc_s.data.hop_length,
    **hps_freevc_s.model
).to(device).eval()
freevc_utils.load_checkpoint(str(freevc_dir / 'checkpoints/freevc-s.pth'), freevc_s_net, None)

freevc_spk_enc = FreeVCSpeakerEncoder(str(freevc_dir / 'speaker_encoder/ckpt/pretrained_bak_5805000.pt'))
cmodel_freevc = WavLMModel.from_pretrained('microsoft/wavlm-large').to(device).eval()

# 4. Load SoundStorm & SpeechTokenizer
print("\n[4/6] Loading SoundStorm & SpeechTokenizer...", flush=True)
st_cfg = soundstorm_dir / 'ckpt/config.json'
st_ckpt = soundstorm_dir / 'ckpt/SpeechTokenizer.pt'
ss_ckpt = soundstorm_dir / 'ckpt/SoundStorm_best_dev.pt'

tokenizer = SpeechTokenizer.load_from_checkpoint(str(st_cfg), str(st_ckpt)).to(device).eval()
conformer = ConformerWrapper(
    codebook_size=1024,
    num_quantizers=7,
    conformer={'dim': 1024, 'depth': 12, 'heads': 8, 'dim_head': 128, 'attn_flash': False},
)
soundstorm = SoundStorm(net=conformer, num_semantic_token_ids=1024, semantic_pad_id=1024, pad_id=1024, schedule='cosine')
soundstorm.load(str(ss_ckpt))
soundstorm = soundstorm.to(device).eval()

# 5. Setup Dataset
root = Path('/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean')
target_speakers = [
    '121', '1089', '1221', '3570', '3729', # Female
    '237', '260', '1188', '4446', '4507'   # Male
]
src_spks = ['1284', '1320', '1580', '1995', '2094', '2300', '2830', '2961', '3575', '4077']

src_files = []
for s in src_spks:
    f_list = sorted(list((root / s).rglob('*.flac')))
    if len(f_list) >= 2: src_files.extend(f_list[:2])
    else: src_files.extend(f_list)
src_files = src_files[:20]

# Load human ground-truth transcriptions from .trans.txt
trans_map = {}
for s in src_spks:
    for tf in (root / s).rglob('*.trans.txt'):
        for line in tf.read_text().splitlines():
            if line.strip():
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    trans_map[parts[0]] = parts[1]

print(f"\n[5/6] Pre-extracting source utterances (N={len(src_files)})...", flush=True)
src_data = []
for idx, p in enumerate(src_files):
    feat_np, w_cpu = ext(p)
    w_gpu = w_cpu.unsqueeze(0).to(device)
    
    stem = p.stem
    gt_raw = trans_map.get(stem, '')
    ref_gt = normalizer(gt_raw)
    
    # Transcribe source with Whisper
    ref_src = transcribe_whisper(w_cpu)
    
    with torch.no_grad():
        src_tokens = rearrange(tokenizer.encode(w_gpu.unsqueeze(0)), 'q b n -> b n q')
        semantic_tokens = src_tokens[:, :, 0]
    
    mu_X, W_X, _ = get_wct(feat_np, min(128, len(feat_np)-1))
    xw = (feat_np - mu_X) @ W_X
    feat_norm = feat_np / (np.linalg.norm(feat_np, axis=1, keepdims=True) + 1e-8)
    
    src_data.append({
        'path': p,
        'stem': stem,
        'spk': p.parent.parent.name,
        'wav': w_cpu,
        'feat': feat_np,
        'feat_norm': feat_norm,
        'xw': xw,
        'mu_X': mu_X,
        'W_X': W_X,
        'ref_gt': ref_gt,
        'ref_src': ref_src,
        'semantic_tokens': semantic_tokens,
        'duration': w_cpu.shape[-1] / 16000.0
    })

# Extract shared codebook for transport methods (identical to unified 200 benchmark)
print("Extracting shared background pool for LWT (K=10 clusters)...", flush=True)
bg_spks = ['4970', '4992', '5142', '5639', '5683', '61']
bg_feats = []
for b_spk in bg_spks:
    b_files = sorted(list((root / b_spk).rglob('*.flac')))[:6]
    for bf in b_files:
        x, _ = ext(bf)
        bg_feats.append(x)
bg_Y = np.concatenate(bg_feats, axis=0)

K_clusters = 10
beta_val = 20.0
km_shared = KMeans(n_clusters=K_clusters, random_state=42, n_init=1).fit(bg_Y)
cents_shared = torch.tensor(km_shared.cluster_centers_, dtype=torch.float32, device=device)
cents_shared_norm = torch.nn.functional.normalize(cents_shared, dim=1)

cov_shared_k = []
for k in range(K_clusters):
    idx_k = np.where(km_shared.labels_ == k)[0]
    Yk = bg_Y[idx_k]
    cov_k = np.cov(Yk, rowvar=False) if len(Yk) > 1 else np.eye(1024)
    cov_shared_k.append(torch.tensor(cov_k, dtype=torch.float32, device=device))

print(f"Pre-extracting target speaker profiles (K={len(target_speakers)})...", flush=True)
target_models = {}
for spk in target_speakers:
    files = sorted(list((root / spk).rglob('*.flac')))[:20]
    X_list = []
    g_freevc_list = []
    soundstorm_prompt = None
    freevc_s_mel = None
    
    for idx, f in enumerate(files):
        ft, w_raw = ext(f)
        X_list.append(ft)
        w_t = w_raw.unsqueeze(0)
        
        # FreeVC speaker embedding
        w_np, _ = librosa.load(str(f), sr=16000)
        w_np, _ = librosa.effects.trim(w_np, top_db=20)
        g_freevc_list.append(freevc_spk_enc.embed_utterance(w_np))
        
        if idx == 0:
            with torch.no_grad():
                p_tokens = rearrange(tokenizer.encode(w_t.unsqueeze(0).to(device)), 'q b n -> b n q')
                prompt_len = min(200, p_tokens.shape[1])
                soundstorm_prompt = p_tokens[:, :prompt_len]
            w_tgt_t = torch.from_numpy(w_np).unsqueeze(0).to(device)
            freevc_s_mel = mel_spectrogram_torch(
                w_tgt_t,
                hps_freevc_s.data.filter_length,
                hps_freevc_s.data.n_mel_channels,
                hps_freevc_s.data.sampling_rate,
                hps_freevc_s.data.hop_length,
                hps_freevc_s.data.win_length,
                hps_freevc_s.data.mel_fmin,
                hps_freevc_s.data.mel_fmax
            )
            
    Y = np.concatenate(X_list, axis=0)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    Y_norm = torch.nn.functional.normalize(Y_t, dim=1)
    
    # Global WCT
    mu_Y, W_Y, C_Y = get_wct(Y, 128)
    
    # ICP Whitened
    H_Y = (Y - mu_Y) @ W_Y
    H_Y_norm = H_Y / (np.linalg.norm(H_Y, axis=1, keepdims=True) + 1e-8)
    
    # Soft Local WCT target clusters (K=10)
    km_tgt = KMeans(n_clusters=K_clusters, random_state=42, n_init=1).fit(Y)
    cents_tgt = torch.tensor(km_tgt.cluster_centers_, dtype=torch.float32, device=device)
    cents_tgt_norm = torch.nn.functional.normalize(cents_tgt, dim=1)
    
    cl_wct = []
    for k in range(K_clusters):
        Yk = Y[km_tgt.labels_ == k]
        muk = np.mean(Yk, axis=0)
        covk = np.cov(Yk - muk, rowvar=False) if len(Yk) > 1 else np.zeros((1024, 1024))
        evals, evecs = np.linalg.eigh(covk)
        evals = np.maximum(evals, 1e-5)
        Ck = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
        cl_wct.append((muk, Ck))
        
    # LWT & Boosted Covariances on Shared Clusters
    sim_Y = Y_norm @ cents_shared_norm.t()
    w_Y = torch.softmax(sim_Y * beta_val, dim=1)
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
        
    g_freevc = torch.from_numpy(np.mean(g_freevc_list, axis=0)).unsqueeze(0).to(device)
    
    target_models[spk] = {
        'Y': Y,
        'Y_t': Y_t,
        'Y_norm': Y_norm,
        'mu_Y': mu_Y,
        'W_Y': W_Y,
        'C_Y': C_Y,
        'H_Y': H_Y,
        'H_Y_norm': H_Y_norm,
        'cents_tgt_norm': cents_tgt_norm,
        'cl_wct': cl_wct,
        'cl_mu_Y_shared': cl_mu_Y_shared,
        'cl_cov_Y_shared': cl_cov_Y_shared,
        'delta_cov_speaker': delta_cov_speaker,
        'g_freevc': g_freevc,
        'freevc_s_mel': freevc_s_mel,
        'soundstorm_prompt': soundstorm_prompt
    }

print("All target models precomputed.", flush=True)

# 6. Evaluation Loop
# EXACT order as in output/tables/unified_200_conversions_detail.csv
methods = [
    'Classic WCT',
    'ICP-WCT (ours)',
    'kNN-VC (k=1)',
    'kNN-VC (k=4)',
    'LinearVC',
    'Soft Local WCT (Ours)',
    'Local Wasserstein (LWT - Ours)',
    'Boosted LWT (alpha=1.5 - Ours)',
    'FreeVC',
    'FreeVC-s',
    'SoundStorm'
]

eval_results = []
checkpoint_csv = proj_root / 'output/tables/whisper_200_conversions_checkpoint.csv'

print("\n[6/6] Running Whisper-Small Evaluation across all 11 models (11 x 200 = 2200 conversions)...", flush=True)
t_bench_start = time.time()

for m_idx, method_name in enumerate(methods):
    t_method_start = time.time()
    print(f"\n[{m_idx+1}/{len(methods)}] Evaluating: {method_name}...", flush=True)
    count = 0
    method_records = []
    
    for tgt_spk in target_speakers:
        tm = target_models[tgt_spk]
        
        for s_idx, s_item in enumerate(src_data):
            xs = s_item['feat']
            xw = s_item['xw']
            xs_norm = s_item['feat_norm']
            Ns = len(xs)
            ref_gt = s_item['ref_gt']
            ref_src = s_item['ref_src']
            w_src = s_item['wav']
            
            xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
            xs_n = torch.nn.functional.normalize(xs_t, dim=1)
            
            # Synthesize converted audio wc
            if method_name == 'Classic WCT':
                x_hat_np = xw @ tm['C_Y'].T + tm['mu_Y']
                wc = voc(x_hat_np)
                
            elif method_name == 'ICP-WCT (ours)':
                R = np.eye(1024, dtype=np.float32)
                for _ in range(3):
                    H_src_rot = xw @ R
                    H_rot_norm = H_src_rot / (np.linalg.norm(H_src_rot, axis=1, keepdims=True) + 1e-8)
                    sim_mat = H_rot_norm @ tm['H_Y_norm'].T
                    best_idx = np.argmax(sim_mat, axis=1)
                    H_Y_matched = tm['H_Y'][best_idx]
                    S = xw.T @ H_Y_matched
                    # Fast GPU SVD
                    U_t, _, Vt_t = torch.linalg.svd(torch.from_numpy(S).to(device))
                    R = (U_t @ Vt_t).cpu().numpy()
                x_hat_np = (xw @ R) @ tm['C_Y'].T + tm['mu_Y']
                wc = voc(x_hat_np)
                
            elif method_name == 'kNN-VC (k=1)':
                sim_mat = torch.mm(xs_n, tm['Y_norm'].t())
                best_idx = torch.argmax(sim_mat, dim=1)
                x_hat_np = tm['Y_t'][best_idx].cpu().numpy()
                wc = voc(x_hat_np)
                
            elif method_name == 'kNN-VC (k=4)':
                sim_mat = torch.mm(xs_n, tm['Y_norm'].t())
                top4_idx = torch.topk(sim_mat, k=4, dim=1).indices
                x_hat_np = torch.mean(tm['Y_t'][top4_idx], dim=1).cpu().numpy()
                wc = voc(x_hat_np)
                
            elif method_name == 'LinearVC':
                sim_mat = torch.mm(xs_n, tm['Y_norm'].t())
                best_idx = torch.argmax(sim_mat, dim=1)
                Y_matched = tm['Y_t'][best_idx].cpu().numpy()
                X_bias = np.hstack([xs, np.ones((Ns, 1), dtype=np.float32)])
                # Fast GPU lstsq
                W_lin = torch.linalg.lstsq(torch.from_numpy(X_bias).to(device), torch.from_numpy(Y_matched).to(device)).solution.cpu().numpy()
                x_hat_np = X_bias @ W_lin
                wc = voc(x_hat_np)
                
            elif method_name == 'FreeVC':
                w_src_t = w_src.unsqueeze(0).to(device)
                with torch.no_grad():
                    c = cmodel_freevc(w_src_t).last_hidden_state.transpose(1, 2)
                    audio_out = freevc_net.infer(c, g=tm['g_freevc'])
                wc = audio_out.squeeze().cpu()
                
            elif method_name == 'FreeVC-s':
                w_src_t = w_src.unsqueeze(0).to(device)
                with torch.no_grad():
                    c = cmodel_freevc(w_src_t).last_hidden_state.transpose(1, 2)
                    audio_out = freevc_s_net.infer(c, mel=tm['freevc_s_mel'])
                wc = audio_out.squeeze().cpu()
                
            elif method_name == 'SoundStorm':
                with torch.no_grad():
                    gen = soundstorm.generate(
                        semantic_tokens=s_item['semantic_tokens'],
                        prompt_tokens=tm['soundstorm_prompt'],
                        steps=16
                    )
                    out_tokens = rearrange(gen, 'b n q -> q b n')
                    audio_out = tokenizer.decode(out_tokens)
                wc = audio_out.squeeze().cpu()
                
            elif method_name == 'Soft Local WCT (Ours)':
                sim_mat = xs_n @ tm['cents_tgt_norm'].t()
                w_dyn = torch.softmax(sim_mat * beta_val, dim=1).cpu().numpy()
                x_hat_np = np.zeros_like(xs)
                for k in range(K_clusters):
                    muk, Ck = tm['cl_wct'][k]
                    x_hat_np += w_dyn[:, k:k+1] * (xw @ Ck.T + muk)
                wc = voc(x_hat_np)
                
            elif method_name == 'Local Wasserstein (LWT - Ours)':
                sim_mat = xs_n @ cents_shared_norm.t()
                w_dyn = torch.softmax(sim_mat * beta_val, dim=1)
                x_hat = torch.zeros_like(xs_t)
                for k in range(K_clusters):
                    wk = w_dyn[:, k:k+1]
                    mass_k = wk.sum()
                    if mass_k < 1e-4: continue
                    mu_X_k = (xs_t * wk).sum(dim=0) / mass_k
                    xc = xs_t - mu_X_k.unsqueeze(0)
                    cov_X_k = torch.mm(xc.t(), xc * wk) / mass_k
                    A_k = compute_bures_map_torch(cov_X_k, tm['cl_cov_Y_shared'][k], eps=1e-2)
                    T_k = torch.mm(xc, A_k) + tm['cl_mu_Y_shared'][k].unsqueeze(0)
                    x_hat += wk * T_k
                x_hat_np = x_hat.cpu().numpy()
                wc = voc(x_hat_np)
                
            elif method_name == 'Boosted LWT (alpha=1.5 - Ours)':
                sim_mat = xs_n @ cents_shared_norm.t()
                w_dyn = torch.softmax(sim_mat * beta_val, dim=1)
                x_hat = torch.zeros_like(xs_t)
                for k in range(K_clusters):
                    wk = w_dyn[:, k:k+1]
                    mass_k = wk.sum()
                    if mass_k < 1e-4: continue
                    mu_X_k = (xs_t * wk).sum(dim=0) / mass_k
                    xc = xs_t - mu_X_k.unsqueeze(0)
                    cov_X_k = torch.mm(xc.t(), xc * wk) / mass_k
                    cov_boosted = cov_shared_k[k] + 1.5 * tm['delta_cov_speaker'][k]
                    cov_boosted = project_psd(cov_boosted, eps=1e-3)
                    A_k = compute_bures_map_torch(cov_X_k, cov_boosted, eps=1e-2)
                    T_k = torch.mm(xc, A_k) + tm['cl_mu_Y_shared'][k].unsqueeze(0)
                    x_hat += wk * T_k
                x_hat_np = x_hat.cpu().numpy()
                wc = voc(x_hat_np)
                
            # Transcribe with Whisper-small
            hyp_whisper = transcribe_whisper(wc)
            
            # Metrics
            cer_wh = jiwer.cer(ref_gt, hyp_whisper) * 100.0
            wer_wh = jiwer.wer(ref_gt, hyp_whisper) * 100.0
            cer_wh_src = jiwer.cer(ref_src, hyp_whisper) * 100.0
            wer_wh_src = jiwer.wer(ref_src, hyp_whisper) * 100.0
            
            rec = {
                'Method': method_name,
                'Target_Speaker': tgt_spk,
                'Source_Speaker': s_item['spk'],
                'CER_Whisper': cer_wh,
                'WER_Whisper': wer_wh,
                'CER_Whisper_src': cer_wh_src,
                'WER_Whisper_src': wer_wh_src
            }
            method_records.append(rec)
            eval_results.append(rec)
            count += 1
            if count % 50 == 0:
                print(f"  Processed {count}/200 conversions...", flush=True)
                
    m_cer_gt = np.mean([r['CER_Whisper'] for r in method_records])
    m_wer_gt = np.mean([r['WER_Whisper'] for r in method_records])
    m_cer_src = np.mean([r['CER_Whisper_src'] for r in method_records])
    m_wer_src = np.mean([r['WER_Whisper_src'] for r in method_records])
    print(f"  -> {method_name} Finished in {time.time() - t_method_start:.1f}s | Whisper CER (GT): {m_cer_gt:.2f}%, WER: {m_wer_gt:.2f}% | (Src Ref) CER: {m_cer_src:.2f}%, WER: {m_wer_src:.2f}%", flush=True)
    
    # Save checkpoint CSV after each model
    pd.DataFrame(eval_results).to_csv(checkpoint_csv, index=False)

print(f"\nAll 11 methods evaluated with Whisper-small in {time.time() - t_bench_start:.1f}s.", flush=True)

# Update output CSVs
detail_csv_path = proj_root / 'output/tables/unified_200_conversions_detail.csv'
df_detail = pd.read_csv(detail_csv_path)

df_wh = pd.DataFrame(eval_results)

# Check order
for idx in range(len(df_detail)):
    row_d = df_detail.iloc[idx]
    row_w = df_wh.iloc[idx]
    assert row_d['Method'] == row_w['Method'], f"Method mismatch at {idx}: {row_d['Method']} vs {row_w['Method']}"
    assert str(row_d['Target_Speaker']) == str(row_w['Target_Speaker']), f"Target mismatch at {idx}"
    assert str(row_d['Source_Speaker']) == str(row_w['Source_Speaker']), f"Source mismatch at {idx}"

print("Merging Whisper metrics into detailed CSV...", flush=True)

for col in ['CER_Whisper', 'WER_Whisper', 'CER_Whisper_src', 'WER_Whisper_src']:
    df_detail[col] = df_wh[col]

df_detail.to_csv(detail_csv_path, index=False)
print(f"Updated {detail_csv_path} with Whisper metrics.", flush=True)

# Update Summary CSV
summary_csv_path = proj_root / 'output/tables/unified_200_benchmark_summary.csv'
df_summary = pd.read_csv(summary_csv_path)

for m in methods:
    m_rows = df_detail[df_detail['Method'] == m]
    df_summary.loc[df_summary['Method'] == m, 'CER_Whisper_mean'] = m_rows['CER_Whisper'].mean()
    df_summary.loc[df_summary['Method'] == m, 'CER_Whisper_std'] = m_rows['CER_Whisper'].std()
    df_summary.loc[df_summary['Method'] == m, 'WER_Whisper_mean'] = m_rows['WER_Whisper'].mean()
    df_summary.loc[df_summary['Method'] == m, 'WER_Whisper_std'] = m_rows['WER_Whisper'].std()
    df_summary.loc[df_summary['Method'] == m, 'CER_Whisper_src_mean'] = m_rows['CER_Whisper_src'].mean()
    df_summary.loc[df_summary['Method'] == m, 'CER_Whisper_src_std'] = m_rows['CER_Whisper_src'].std()
    df_summary.loc[df_summary['Method'] == m, 'WER_Whisper_src_mean'] = m_rows['WER_Whisper_src'].mean()
    df_summary.loc[df_summary['Method'] == m, 'WER_Whisper_src_std'] = m_rows['WER_Whisper_src'].std()

df_summary.to_csv(summary_csv_path, index=False)
print(f"Updated {summary_csv_path} with Whisper metrics summary.", flush=True)

# Update LaTeX table
tex_path = proj_root / 'output/tables/tab_overall_unified_200.tex'
tex_lines = [
    r"\begin{table*}[t]",
    r"    \centering",
    r"    \caption{Comprehensive performance comparison across voice conversion paradigms evaluated on a strictly unified benchmark of $N=200$ conversions (10 target speakers, 20 source utterances from 10 distinct speakers, $T=20$ target utterances). Intelligibility is assessed using Whisper-Small ASR (CER/WER normalized against LibriSpeech ground truth). Speaker similarity is verified across two complementary backends: ECAPA-TDNN and WavLM-SV.}",
    r"    \label{tab:overall_unified_200}",
    r"    \begin{tabular}{llccccccc}",
    r"        \toprule",
    r"        \textbf{Method} & \textbf{Approach Type} & \textbf{CER (Wh.)} $\downarrow$ & \textbf{WER (Wh.)} $\downarrow$ & \textbf{Sim (ECAPA)} $\uparrow$ & \textbf{Sim (WavLM-SV)} $\uparrow$ & \textbf{MOS} $\uparrow$ & \textbf{Rel. Jitter} $\downarrow$ & \textbf{RTF} $\downarrow$ \\",
    r"        \midrule"
]

for _, r in df_summary.iterrows():
    m = r['Method']
    app = r['Approach_Type']
    c_mean, c_std = r['CER_Whisper_mean'], r['CER_Whisper_std']
    w_mean, w_std = r['WER_Whisper_mean'], r['WER_Whisper_std']
    s_mean, s_std = r['Sim_mean'], r['Sim_std']
    sv_mean, sv_std = r['Sim_WavLM_mean'], r['Sim_WavLM_std']
    mos_m, mos_s = r['MOS_mean'], r['MOS_std']
    jit = r['Jitter_mean']
    rtf = r['RTF_mean']
    
    is_ours = 'Ours' in m or 'ours' in m or 'LWT' in m
    bold_start = r"\textbf{" if is_ours else ""
    bold_end = "}" if is_ours else ""
    
    line = f"        {bold_start}{m}{bold_end} & {app} & {bold_start}{c_mean:.2f} $\\pm$ {c_std:.2f}{bold_end} & {bold_start}{w_mean:.2f} $\\pm$ {w_std:.2f}{bold_end} & {bold_start}{s_mean:.3f} $\\pm$ {s_std:.3f}{bold_end} & {bold_start}{sv_mean:.3f} $\\pm$ {sv_std:.3f}{bold_end} & {bold_start}{mos_m:.2f} $\\pm$ {mos_s:.2f}{bold_end} & {bold_start}{jit:.2f}$\\times${bold_end} & {bold_start}{rtf:.4f}{bold_end} \\\\"
    tex_lines.append(line)

tex_lines.extend([
    r"        \bottomrule",
    r"    \end{tabular}",
    r"\end{table*}"
])

tex_path.write_text("\n".join(tex_lines))
print(f"Updated LaTeX table: {tex_path}", flush=True)

# Print comparative summary table
print("\n" + "="*115, flush=True)
print(f"{'Method':<30} | {'W2V2 CER':<9} {'W2V2 WER':<9} | {'Wh CER (GT)':<11} {'Wh WER (GT)':<11} | {'Wh CER (Src)':<12} {'Wh WER (Src)':<12}", flush=True)
print("="*115, flush=True)
for _, r in df_summary.iterrows():
    m = r['Method']
    w2v_c = f"{r['CER_mean']:.2f}%"
    w2v_w = f"{r['WER_mean']:.2f}%"
    wh_c = f"{r['CER_Whisper_mean']:.2f}%"
    wh_w = f"{r['WER_Whisper_mean']:.2f}%"
    wh_cs = f"{r['CER_Whisper_src_mean']:.2f}%"
    wh_ws = f"{r['WER_Whisper_src_mean']:.2f}%"
    print(f"{m:<30} | {w2v_c:<9} {w2v_w:<9} | {wh_c:<11} {wh_w:<11} | {wh_cs:<12} {wh_ws:<12}", flush=True)
print("="*115, flush=True)
