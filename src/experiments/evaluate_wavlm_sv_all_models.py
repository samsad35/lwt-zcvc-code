#!/usr/bin/env python3
"""
Evaluate Speaker Similarity across all 11 Voice Conversion models on the Grand Unified
Benchmark (N=200 conversions) using microsoft/wavlm-base-sv.

Produces:
  - Updates output/tables/unified_200_conversions_detail.csv with 'Sim_WavLM'
  - Updates output/tables/unified_200_benchmark_summary.csv with 'Sim_WavLM_mean', 'Sim_WavLM_std'
  - Updates output/tables/tab_overall_unified_200.tex with both ECAPA and WavLM-SV columns
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
from transformers import AutoFeatureExtractor, WavLMForXVector, WavLMModel
import librosa
from einops import rearrange

# Root and paths
proj_root = Path(__file__).resolve().parent.parent.parent
freevc_dir = proj_root / 'src/baselines/freevc'
soundstorm_dir = proj_root / 'src/baselines/soundstorm'

sys.path.append(str(freevc_dir))
sys.path.append(str(soundstorm_dir))

import utils as freevc_utils
from models import SynthesizerTrn
from speaker_encoder.voice_encoder import SpeakerEncoder as FreeVCSpeakerEncoder
from mel_processing import mel_spectrogram_torch

from soundstorm_speechtokenizer import SoundStorm, ConformerWrapper
from speechtokenizer import SpeechTokenizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load WavLM-SV Model
print("[1/6] Loading microsoft/wavlm-base-sv speaker verification model...")
wavlm_sv_id = 'microsoft/wavlm-base-sv'
extractor = AutoFeatureExtractor.from_pretrained(wavlm_sv_id)
wavlm_sv = WavLMForXVector.from_pretrained(wavlm_sv_id).to(device).eval()

def get_wavlm_sv_emb(wav_tensor):
    w_np = wav_tensor.squeeze().cpu().numpy()
    inp = extractor(w_np, sampling_rate=16000, return_tensors='pt').to(device)
    with torch.no_grad():
        emb = wavlm_sv(**inp).embeddings.squeeze().cpu()
    return F.normalize(emb, dim=0)

# 2. Load Vocoder & WavLM for Content
print("[2/6] Loading HiFi-GAN and WavLM-Large for feature VC...")
knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
vocoder = knn_vc.hifigan.to(device).eval()
wavlm_large = knn_vc.wavlm.to(device).eval()

# 3. Load FreeVC & FreeVC-s
print("[3/6] Loading FreeVC models...")
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
print("[4/6] Loading SoundStorm & SpeechTokenizer...")
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

print(f"[5/6] Pre-extracting source utterances (N={len(src_files)})...")
src_data = []
for p in src_files:
    w, sr = torchaudio.load(str(p))
    if sr != 16000: w = F.resample(w, sr, 16000)
    if w.dim() == 2 and w.shape[0] > 1: w = w.mean(dim=0, keepdim=True)
    elif w.dim() == 1: w = w.unsqueeze(0)
    
    with torch.no_grad():
        feat, _ = wavlm_large.extract_features(w.to(device), output_layer=6)
        src_tokens = rearrange(tokenizer.encode(w.unsqueeze(0).to(device)), 'q b n -> b n q')
        semantic_tokens = src_tokens[:, :, 0]
        
    src_data.append({
        'path': p,
        'spk': p.parent.parent.name,
        'wav': w.squeeze().cpu(),
        'feat': feat.squeeze(0).cpu().numpy(),
        'semantic_tokens': semantic_tokens,
        'duration': w.shape[-1] / 16000.0
    })

print("Extracting shared codebook for transport methods...")
all_feats = []
for s in target_speakers:
    f = sorted(list((root / s).rglob('*.flac')))[0]
    w_t, sr = torchaudio.load(str(f))
    if sr != 16000: w_t = F.resample(w_t, sr, 16000)
    with torch.no_grad():
        ft, _ = wavlm_large.extract_features(w_t.to(device), output_layer=6)
    all_feats.append(ft.squeeze(0))
all_feats = torch.cat(all_feats, dim=0)

K_clusters = 16
beta_val = 15.0
indices = torch.linspace(0, len(all_feats) - 1, K_clusters).long()
cents_shared = all_feats[indices].clone()
cents_shared_norm = F.normalize(cents_shared, dim=1)

sim_sh = all_feats @ cents_shared_norm.t()
w_sh = torch.softmax(sim_sh * beta_val, dim=1)
cov_shared_k = []
for k in range(K_clusters):
    wk = w_sh[:, k:k+1]
    mk = wk.sum()
    muk = (all_feats * wk).sum(dim=0) / (mk + 1e-8)
    zc = all_feats - muk.unsqueeze(0)
    cov_k = torch.mm(zc.t(), zc * wk) / (mk + 1e-8)
    cov_shared_k.append(cov_k)

print(f"Pre-extracting target profiles (K={len(target_speakers)})...")
target_models = {}
for spk in target_speakers:
    files = sorted(list((root / spk).rglob('*.flac')))[:20]
    wavs_raw = []
    g_freevc_list = []
    
    Y_list = []
    for idx, f in enumerate(files):
        w_t, sr = torchaudio.load(str(f))
        if sr != 16000: w_t = F.resample(w_t, sr, 16000)
        if w_t.dim() == 2 and w_t.shape[0] > 1: w_t = w_t.mean(dim=0, keepdim=True)
        elif w_t.dim() == 1: w_t = w_t.unsqueeze(0)
        wavs_raw.append(w_t.squeeze().cpu())
        
        with torch.no_grad():
            ft, _ = wavlm_large.extract_features(w_t.to(device), output_layer=6)
        Y_list.append(ft.squeeze(0).cpu())
        
        # FreeVC
        w_np, _ = librosa.load(str(f), sr=16000)
        w_np, _ = librosa.effects.trim(w_np, top_db=20)
        g_freevc_list.append(freevc_spk_enc.embed_utterance(w_np))
        
        if idx == 0:
            # SoundStorm prompt
            with torch.no_grad():
                p_tokens = rearrange(tokenizer.encode(w_t.unsqueeze(0).to(device)), 'q b n -> b n q')
                prompt_len = min(200, p_tokens.shape[1])
                soundstorm_prompt = p_tokens[:, :prompt_len]
            # FreeVC-s reference mel
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
            
    # Target WavLM-SV profile
    prof_sv = torch.stack([get_wavlm_sv_emb(w) for w in wavs_raw[:10]]).mean(dim=0)
    prof_sv = F.normalize(prof_sv, dim=0)
    
    # Feature matrices
    Y_all = torch.cat(Y_list, dim=0).to(device)
    Y_norm = F.normalize(Y_all, dim=1)
    
    mu_Y = torch.mean(Y_all, dim=0)
    cov_Y = torch.cov(Y_all.t())
    evals, evecs = torch.linalg.eigh(cov_Y)
    evals = torch.clamp(evals, min=1e-5)
    C_Y = evecs @ torch.diag(torch.sqrt(evals)) @ evecs.t()
    
    # FreeVC averaged g
    g_avg = np.mean(g_freevc_list, axis=0)
    g_avg = g_avg / np.linalg.norm(g_avg, 2)
    g_freevc = torch.from_numpy(g_avg).unsqueeze(0).to(device)
    
    # Local Wasserstein covariances
    sim_Y = Y_norm @ cents_shared_norm.t()
    w_Y = torch.softmax(sim_Y * beta_val, dim=1)
    cl_mu_Y_shared = []
    cl_cov_Y_shared = []
    delta_cov_speaker = []
    for k in range(K_clusters):
        wk = w_Y[:, k:k+1]
        mk = wk.sum()
        muk = (Y_all * wk).sum(dim=0) / (mk + 1e-8)
        yc = Y_all - muk.unsqueeze(0)
        cov_Y_k = torch.mm(yc.t(), yc * wk) / (mk + 1e-8)
        cl_mu_Y_shared.append(muk)
        cl_cov_Y_shared.append(cov_Y_k)
        delta_cov_speaker.append(cov_Y_k - cov_shared_k[k])
        
    target_models[spk] = {
        'prof_sv': prof_sv,
        'Y_all': Y_all,
        'Y_norm': Y_norm,
        'mu_Y': mu_Y,
        'C_Y': C_Y,
        'cl_mu_Y_shared': cl_mu_Y_shared,
        'cl_cov_Y_shared': cl_cov_Y_shared,
        'delta_cov_speaker': delta_cov_speaker,
        'g_freevc': g_freevc,
        'freevc_s_mel': freevc_s_mel,
        'soundstorm_prompt': soundstorm_prompt
    }

print("Target models prepared successfully.")

# 6. Run WavLM-SV Evaluation Across All 11 Methods
methods = [
    'Classic WCT',
    'ICP-WCT (ours)',
    'kNN-VC (k=1)',
    'kNN-VC (k=4)',
    'LinearVC',
    'FreeVC',
    'FreeVC-s',
    'SoundStorm',
    'Soft Local WCT (Ours)',
    'Local Wasserstein (LWT - Ours)',
    'Boosted LWT (alpha=1.5 - Ours)'
]

sim_records = {m: [] for m in methods}

print("\n[6/6] Computing WavLM-SV speaker similarity across all 200 conversions...")
t_all_start = time.time()

for tgt_idx, (tgt_spk, tm) in enumerate(target_models.items()):
    prof_sv = tm['prof_sv']
    Y_all = tm['Y_all']
    Y_norm = tm['Y_norm']
    mu_Y = tm['mu_Y']
    C_Y = tm['C_Y']
    g_freevc = tm['g_freevc']
    freevc_s_mel = tm['freevc_s_mel']
    soundstorm_prompt = tm['soundstorm_prompt']
    
    for s_item in src_data:
        xs_np = s_item['feat']
        xs_t = torch.tensor(xs_np, dtype=torch.float32, device=device)
        xs_n = F.normalize(xs_t, dim=1)
        w_src = s_item['wav'].to(device)
        semantic_tokens = s_item['semantic_tokens']
        
        # 1. Classic WCT
        with torch.no_grad():
            mu_X = torch.mean(xs_t, dim=0)
            cov_X = torch.cov(xs_t.t())
            evals, evecs = torch.linalg.eigh(cov_X)
            evals = torch.clamp(evals, min=1e-5)
            inv_C_X = evecs @ torch.diag(1.0 / torch.sqrt(evals)) @ evecs.t()
            y_wct = (xs_t - mu_X) @ inv_C_X @ C_Y + mu_Y
            w_wct = vocoder(y_wct.unsqueeze(0)).squeeze().cpu()
            sim_records['Classic WCT'].append(torch.dot(prof_sv, get_wavlm_sv_emb(w_wct)).item())
            
        # 2. kNN-VC (k=1)
        with torch.no_grad():
            cos_sim = xs_n @ Y_norm.t()
            top1_idx = torch.argmax(cos_sim, dim=1)
            y_k1 = Y_all[top1_idx]
            w_k1 = vocoder(y_k1.unsqueeze(0)).squeeze().cpu()
            sim_records['kNN-VC (k=1)'].append(torch.dot(prof_sv, get_wavlm_sv_emb(w_k1)).item())
            
        # 3. kNN-VC (k=4)
        with torch.no_grad():
            val, idx = torch.topk(cos_sim, k=4, dim=1)
            w_k4_w = F.softmax(val / 0.1, dim=1)
            y_k4 = (Y_all[idx] * w_k4_w.unsqueeze(-1)).sum(dim=1)
            w_k4 = vocoder(y_k4.unsqueeze(0)).squeeze().cpu()
            sim_records['kNN-VC (k=4)'].append(torch.dot(prof_sv, get_wavlm_sv_emb(w_k4)).item())
            
        # 4. LinearVC
        with torch.no_grad():
            val_lin, idx_lin = torch.topk(cos_sim, k=8, dim=1)
            y_top8 = Y_all[idx_lin]
            mean_y = y_top8.mean(dim=1, keepdim=True)
            yc = y_top8 - mean_y
            U, S, V = torch.pca_lowrank(yc.view(-1, yc.shape[-1]), q=4)
            diff = (xs_t.unsqueeze(1) - mean_y)
            proj = (diff @ V) @ V.t()
            y_lin = mean_y.squeeze(1) + proj.squeeze(1)
            w_lin = vocoder(y_lin.unsqueeze(0)).squeeze().cpu()
            sim_records['LinearVC'].append(torch.dot(prof_sv, get_wavlm_sv_emb(w_lin)).item())
            
        # 5. ICP-WCT
        with torch.no_grad():
            R = torch.eye(xs_t.shape[1], device=device)
            t = mu_Y - mu_X
            for _ in range(3):
                xt = (xs_t @ R.t()) + t
                xt_n = F.normalize(xt, dim=1)
                nn_idx = torch.argmax(xt_n @ Y_norm.t(), dim=1)
                y_match = Y_all[nn_idx]
                mx = xt.mean(dim=0)
                my = y_match.mean(dim=0)
                H = (xt - mx).t() @ (y_match - my)
                u, s, vh = torch.linalg.svd(H)
                R_step = vh.t() @ u.t()
                R = R_step @ R
                t = my - mx @ R.t()
            y_icp = (xs_t @ R.t()) + t
            w_icp = vocoder(y_icp.unsqueeze(0)).squeeze().cpu()
            sim_records['ICP-WCT (ours)'].append(torch.dot(prof_sv, get_wavlm_sv_emb(w_icp)).item())

        # 6. Soft Local WCT
        with torch.no_grad():
            sim_sh_x = xs_n @ cents_shared_norm.t()
            w_x = torch.softmax(sim_sh_x * beta_val, dim=1)
            y_soft = torch.zeros_like(xs_t)
            for k in range(K_clusters):
                wk = w_x[:, k:k+1]
                muk_Y = tm['cl_mu_Y_shared'][k]
                covk_Y = tm['cl_cov_Y_shared'][k]
                evals, evecs = torch.linalg.eigh(covk_Y)
                evals = torch.clamp(evals, min=1e-5)
                Ck_Y = evecs @ torch.diag(torch.sqrt(evals)) @ evecs.t()
                cov_sh = cov_shared_k[k]
                evals_s, evecs_s = torch.linalg.eigh(cov_sh)
                evals_s = torch.clamp(evals_s, min=1e-5)
                inv_Ck_sh = evecs_s @ torch.diag(1.0 / torch.sqrt(evals_s)) @ evecs_s.t()
                yk = (xs_t - cents_shared[k]) @ inv_Ck_sh @ Ck_Y + muk_Y
                y_soft += yk * wk
            w_soft = vocoder(y_soft.unsqueeze(0)).squeeze().cpu()
            sim_records['Soft Local WCT (Ours)'].append(torch.dot(prof_sv, get_wavlm_sv_emb(w_soft)).item())

        # 7. Local Wasserstein (LWT)
        with torch.no_grad():
            y_lwt = torch.zeros_like(xs_t)
            for k in range(K_clusters):
                wk = w_x[:, k:k+1]
                mass_k = wk.sum()
                if mass_k < 1.0:
                    y_lwt += (xs_t - cents_shared[k] + tm['cl_mu_Y_shared'][k]) * wk
                    continue
                muk_X = (xs_t * wk).sum(dim=0) / mass_k
                xc = xs_t - muk_X.unsqueeze(0)
                cov_X_k = torch.mm(xc.t(), xc * wk) / mass_k
                eps = 1e-4
                Sx = cov_X_k + eps * torch.eye(1024, device=device)
                Sy = tm['cl_cov_Y_shared'][k] + eps * torch.eye(1024, device=device)
                evals_x, evecs_x = torch.linalg.eigh(Sx)
                evals_x = torch.clamp(evals_x, min=1e-5)
                Sx_sqrt = evecs_x @ torch.diag(torch.sqrt(evals_x)) @ evecs_x.t()
                Sx_inv_sqrt = evecs_x @ torch.diag(1.0 / torch.sqrt(evals_x)) @ evecs_x.t()
                C = Sx_sqrt @ Sy @ Sx_sqrt
                evals_c, evecs_c = torch.linalg.eigh(C)
                evals_c = torch.clamp(evals_c, min=1e-5)
                C_sqrt = evecs_c @ torch.diag(torch.sqrt(evals_c)) @ evecs_c.t()
                M_k = Sx_inv_sqrt @ C_sqrt @ Sx_inv_sqrt
                yk = (xs_t - muk_X) @ M_k + tm['cl_mu_Y_shared'][k]
                y_lwt += yk * wk
            w_lwt = vocoder(y_lwt.unsqueeze(0)).squeeze().cpu()
            sim_records['Local Wasserstein (LWT - Ours)'].append(torch.dot(prof_sv, get_wavlm_sv_emb(w_lwt)).item())

        # 8. Boosted LWT (alpha=1.5)
        with torch.no_grad():
            y_boost = torch.zeros_like(xs_t)
            alpha = 1.5
            for k in range(K_clusters):
                wk = w_x[:, k:k+1]
                mass_k = wk.sum()
                if mass_k < 1.0:
                    y_boost += (xs_t - cents_shared[k] + tm['cl_mu_Y_shared'][k]) * wk
                    continue
                muk_X = (xs_t * wk).sum(dim=0) / mass_k
                xc = xs_t - muk_X.unsqueeze(0)
                cov_X_k = torch.mm(xc.t(), xc * wk) / mass_k
                eps = 1e-4
                Sx = cov_X_k + eps * torch.eye(1024, device=device)
                Sy_boost = tm['cl_cov_Y_shared'][k] + alpha * tm['delta_cov_speaker'][k] + eps * torch.eye(1024, device=device)
                evals_y, evecs_y = torch.linalg.eigh(Sy_boost)
                Sy_boost = evecs_y @ torch.diag(torch.clamp(evals_y, min=1e-5)) @ evecs_y.t()
                evals_x, evecs_x = torch.linalg.eigh(Sx)
                evals_x = torch.clamp(evals_x, min=1e-5)
                Sx_sqrt = evecs_x @ torch.diag(torch.sqrt(evals_x)) @ evecs_x.t()
                Sx_inv_sqrt = evecs_x @ torch.diag(1.0 / torch.sqrt(evals_x)) @ evecs_x.t()
                C = Sx_sqrt @ Sy_boost @ Sx_sqrt
                evals_c, evecs_c = torch.linalg.eigh(C)
                evals_c = torch.clamp(evals_c, min=1e-5)
                C_sqrt = evecs_c @ torch.diag(torch.sqrt(evals_c)) @ evecs_c.t()
                M_k = Sx_inv_sqrt @ C_sqrt @ Sx_inv_sqrt
                yk = (xs_t - muk_X) @ M_k + tm['cl_mu_Y_shared'][k]
                y_boost += yk * wk
            w_boost = vocoder(y_boost.unsqueeze(0)).squeeze().cpu()
            sim_records['Boosted LWT (alpha=1.5 - Ours)'].append(torch.dot(prof_sv, get_wavlm_sv_emb(w_boost)).item())

        # 9. FreeVC
        with torch.no_grad():
            c_f = cmodel_freevc(w_src.unsqueeze(0)).last_hidden_state.transpose(1, 2)
            w_fvc = freevc_net.infer(c_f, g=g_freevc).squeeze().cpu()
            sim_records['FreeVC'].append(torch.dot(prof_sv, get_wavlm_sv_emb(w_fvc)).item())
            
        # 10. FreeVC-s
        with torch.no_grad():
            w_fvcs = freevc_s_net.infer(c_f, mel=freevc_s_mel).squeeze().cpu()
            sim_records['FreeVC-s'].append(torch.dot(prof_sv, get_wavlm_sv_emb(w_fvcs)).item())
            
        # 11. SoundStorm
        with torch.no_grad():
            gen = soundstorm.generate(semantic_tokens=semantic_tokens, prompt_tokens=soundstorm_prompt, steps=8, greedy=True)
            out_tokens = rearrange(gen, 'b n q -> q b n')
            w_ss = tokenizer.decode(out_tokens).squeeze().cpu()
            sim_records['SoundStorm'].append(torch.dot(prof_sv, get_wavlm_sv_emb(w_ss)).item())
            
    print(f"Completed target speaker {tgt_idx+1}/10 ({tgt_spk}) - Elapsed: {time.time() - t_all_start:.1f}s")

print(f"\nAll conversions processed in {time.time() - t_all_start:.1f}s!")

# 7. Update conversions detail and summary
out_dir = proj_root / 'output/tables'
detail_path = out_dir / 'unified_200_conversions_detail.csv'
df_detail = pd.read_csv(detail_path)

# Add Sim_WavLM to df_detail
for m in methods:
    mask = df_detail['Method'] == m
    if mask.sum() == len(sim_records[m]):
        df_detail.loc[mask, 'Sim_WavLM'] = sim_records[m]
    else:
        print(f"Warning: Count mismatch for {m}: {mask.sum()} vs {len(sim_records[m])}")

df_detail.to_csv(detail_path, index=False)
print(f"Updated conversions detail with Sim_WavLM: {detail_path}")

# Aggregate Summary
summary_path = out_dir / 'unified_200_benchmark_summary.csv'
summary = pd.read_csv(summary_path)

# Compute mean & std of Sim_WavLM for each method
wsv_means = []
wsv_stds = []
for _, row in summary.iterrows():
    m = row['Method']
    scores = sim_records.get(m, [])
    wsv_means.append(np.mean(scores) if len(scores) > 0 else 0.0)
    wsv_stds.append(np.std(scores) if len(scores) > 0 else 0.0)

summary['Sim_WavLM_mean'] = wsv_means
summary['Sim_WavLM_std'] = wsv_stds
summary.to_csv(summary_path, index=False)
print(f"Updated benchmark summary with Sim_WavLM: {summary_path}")

print("\n" + "="*95)
print("=== DUAL SPEAKER SIMILARITY BENCHMARK SUMMARY (ECAPA vs WavLM-SV) ===")
print("="*95)
print(summary[['Method', 'Approach_Type', 'Sim_mean', 'Sim_std', 'Sim_WavLM_mean', 'Sim_WavLM_std', 'CER_mean', 'WER_mean', 'MOS_mean']].to_string())

# 8. Update Publication LaTeX Table with both metrics
tex_path = out_dir / 'tab_overall_unified_200.tex'
with open(tex_path, 'w') as f:
    f.write("\\begin{table*}[t]\n")
    f.write("    \\centering\n")
    f.write("    \\caption{Comprehensive performance comparison across voice conversion paradigms evaluated on a strictly unified benchmark of $N=200$ conversions (10 target speakers, 20 source utterances from 10 distinct speakers, $T=20$ target utterances). Speaker similarity is verified across two complementary backends: ECAPA-TDNN (time-delay neural network) and WavLM-SV (self-supervised transformer x-vector).}\n")
    f.write("    \\label{tab:overall_unified_200}\n")
    f.write("    \\begin{tabular}{llccccccc}\n")
    f.write("        \\toprule\n")
    f.write("        \\textbf{Method} & \\textbf{Approach Type} & \\textbf{CER (\\%)} $\\downarrow$ & \\textbf{WER (\\%)} $\\downarrow$ & \\textbf{Sim (ECAPA)} $\\uparrow$ & \\textbf{Sim (WavLM-SV)} $\\uparrow$ & \\textbf{MOS} $\\uparrow$ & \\textbf{Rel. Jitter} $\\downarrow$ & \\textbf{RTF} $\\downarrow$ \\\\\n")
    f.write("        \\midrule\n")
    for _, row in summary.iterrows():
        m = row['Method']
        app = row['Approach_Type']
        cer = f"{row['CER_mean']:.2f} $\\pm$ {row['CER_std']:.2f}"
        wer = f"{row['WER_mean']:.2f} $\\pm$ {row['WER_std']:.2f}"
        sim_ecapa = f"{row['Sim_mean']:.3f} $\\pm$ {row['Sim_std']:.3f}"
        sim_wsv = f"{row['Sim_WavLM_mean']:.3f} $\\pm$ {row['Sim_WavLM_std']:.3f}"
        mos = f"{row['MOS_mean']:.2f} $\\pm$ {row['MOS_std']:.2f}"
        jit = f"{row['Jitter_mean']:.2f}$\\times$"
        rtf = f"{row['RTF_mean']:.4f}"
        
        if 'Ours' in m:
            f.write(f"        \\textbf{{{m}}} & {app} & \\textbf{{{cer}}} & \\textbf{{{wer}}} & \\textbf{{{sim_ecapa}}} & \\textbf{{{sim_wsv}}} & \\textbf{{{mos}}} & \\textbf{{{jit}}} & \\textbf{{{rtf}}} \\\\\n")
        else:
            f.write(f"        {m} & {app} & {cer} & {wer} & {sim_ecapa} & {sim_wsv} & {mos} & {jit} & {rtf} \\\\\n")
    f.write("        \\bottomrule\n")
    f.write("    \\end{tabular}\n")
    f.write("\\end{table*}\n")

print(f"Saved updated LaTeX table with dual similarity to {tex_path}")
