#!/usr/bin/env python3
"""
Ablation Study: Evaluate Local WCT with Local Source Statistics across all N=200 conversions
on the Grand Unified Benchmark.

Compares:
  1. Classic WCT (Global Source, Global Target)
  2. Soft Local WCT (Global Source, Local Target)
  3. Local WCT (Local Source Stats, Local Target WCT: W_{X,k} @ C_{Y,k}) -> THIS ABLATION
  4. Local Wasserstein Transport (LWT: Local Bures-Wasserstein Monge Map A_k)
  5. Boosted LWT (alpha=1.5: Speaker-Covariance Amplified Monge Map)
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
from sklearn.cluster import KMeans
import jiwer

from transformers import (
    Wav2Vec2ForCTC, Wav2Vec2Processor,
    WhisperProcessor, WhisperForConditionalGeneration,
    AutoFeatureExtractor, WavLMForXVector
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from speechbrain.inference.speaker import EncoderClassifier

proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(proj_root))
from src.core.models import get_wct, compute_trajectory_jitter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

# 1. Load Pretrained Models
print("\n[1/5] Loading evaluation models and vocoders...", flush=True)
knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
vocoder = knn_vc.hifigan.to(device).eval()
wavlm_large = knn_vc.wavlm.to(device).eval()

# ASR Models
processor_w2v2 = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
asr_w2v2 = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h').to(device).eval()

whisper_processor = WhisperProcessor.from_pretrained('openai/whisper-small')
whisper_model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small').to(device).eval()
normalizer = BasicTextNormalizer()

# Speaker Verification Models
spk_model_ecapa = EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb',
    run_opts={'device': 'cuda:0'},
    savedir='/tmp/speechbrain'
)

wavlm_sv_id = 'microsoft/wavlm-base-sv'
extractor_wavlm_sv = AutoFeatureExtractor.from_pretrained(wavlm_sv_id)
wavlm_sv_model = WavLMForXVector.from_pretrained(wavlm_sv_id).to(device).eval()

# Speech Quality Models (SQUIM)
squim_subj = torchaudio.pipelines.SQUIM_SUBJECTIVE.get_model().to(device).eval()
squim_obj = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().to(device).eval()

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

def tr_w2v2(w):
    inp = processor_w2v2(w.squeeze().numpy(), sampling_rate=16000, return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        log = asr_w2v2(inp.input_values).logits
    pred_ids = torch.argmax(log, dim=-1)
    return processor_w2v2.batch_decode(pred_ids)[0].strip()

def tr_whisper(w):
    audio_np = w.squeeze().cpu().numpy() if torch.is_tensor(w) else np.squeeze(w)
    inp = whisper_processor(audio_np, sampling_rate=16000, return_tensors='pt').input_features.to(device)
    with torch.no_grad():
        pred_ids = whisper_model.generate(inp, language='english')
    raw = whisper_processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    return normalizer(raw)

def emb_ecapa(w):
    with torch.no_grad():
        return F.normalize(
            spk_model_ecapa.encode_batch(w.squeeze().float().unsqueeze(0).to(device)).squeeze().cpu(),
            dim=0
        )

def emb_wavlm_sv(w):
    audio_np = w.squeeze().cpu().numpy() if torch.is_tensor(w) else np.squeeze(w)
    inp = extractor_wavlm_sv(audio_np, sampling_rate=16000, return_tensors='pt').to(device)
    with torch.no_grad():
        emb = wavlm_sv_model(**inp).embeddings.squeeze().cpu()
    return F.normalize(emb, dim=0)

# 2. Setup Dataset
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

trans_map = {}
for s in src_spks:
    for tf in (root / s).rglob('*.trans.txt'):
        for line in tf.read_text().splitlines():
            if line.strip():
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    trans_map[parts[0]] = parts[1]

print(f"\n[2/5] Pre-extracting source utterances (N={len(src_files)})...", flush=True)
src_data = []
for idx, p in enumerate(src_files):
    feat_np, w_cpu = ext(p)
    stem = p.stem
    gt_raw = trans_map.get(stem, '')
    ref_gt = normalizer(gt_raw)
    ref_w2v2 = tr_w2v2(w_cpu)
    ref_wh_src = tr_whisper(w_cpu)
    
    src_data.append({
        'path': p,
        'stem': stem,
        'spk': p.parent.parent.name,
        'wav': w_cpu,
        'feat': feat_np,
        'ref_gt': ref_gt,
        'ref_w2v2': ref_w2v2,
        'ref_wh_src': ref_wh_src,
        'duration': w_cpu.shape[-1] / 16000.0
    })

# Extract shared background pool for K=10 clusters (identical to LWT)
print("Extracting shared background pool (K=10 clusters)...", flush=True)
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

print(f"\n[3/5] Precomputing target speaker models (K={len(target_speakers)})...", flush=True)
target_models = {}
for spk in target_speakers:
    files = sorted(list((root / spk).rglob('*.flac')))[:20]
    X_list, wavs_list = [], []
    for f in files:
        ft, w_raw = ext(f)
        X_list.append(ft)
        wavs_list.append(w_raw)
        
    Y = np.concatenate(X_list, axis=0)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    Y_norm = torch.nn.functional.normalize(Y_t, dim=1)
    
    # Speaker verification profiles
    prof_ecapa = torch.stack([emb_ecapa(w) for w in wavs_list[:10]]).mean(dim=0)
    prof_ecapa = F.normalize(prof_ecapa, dim=0)
    
    prof_wavlm_sv = torch.stack([emb_wavlm_sv(w) for w in wavs_list[:10]]).mean(dim=0)
    prof_wavlm_sv = F.normalize(prof_wavlm_sv, dim=0)
    
    # Shared cluster statistics on target
    sim_Y = Y_norm @ cents_shared_norm.t()
    w_Y = torch.softmax(sim_Y * beta_val, dim=1)
    cl_mu_Y_shared = []
    cl_cov_Y_shared = []
    cl_C_Y_shared = []  # Matrix square root (coloration matrix)
    
    eps = 1e-3
    I_D = torch.eye(1024, device=device)
    
    for k in range(K_clusters):
        wk = w_Y[:, k:k+1]
        mass_k = wk.sum()
        muk = (Y_t * wk).sum(dim=0) / (mass_k + 1e-8)
        yc = Y_t - muk.unsqueeze(0)
        cov_Y_k = torch.mm(yc.t(), yc * wk) / (mass_k + 1e-8)
        
        # Coloration matrix C_{Y,k} = Sigma_{Y,k}^{1/2}
        evals_Y, evecs_Y = torch.linalg.eigh(cov_Y_k + eps * I_D)
        evals_Y = torch.clamp(evals_Y, min=eps)
        Ck = evecs_Y @ torch.diag(torch.sqrt(evals_Y)) @ evecs_Y.t()
        
        cl_mu_Y_shared.append(muk)
        cl_cov_Y_shared.append(cov_Y_k)
        cl_C_Y_shared.append(Ck)
        
    target_models[spk] = {
        'prof_ecapa': prof_ecapa,
        'prof_wavlm_sv': prof_wavlm_sv,
        'cl_mu_Y_shared': cl_mu_Y_shared,
        'cl_cov_Y_shared': cl_cov_Y_shared,
        'cl_C_Y_shared': cl_C_Y_shared
    }

print("Target models precomputed.", flush=True)

# 3. Conversion and Evaluation Loop
print("\n[4/5] Running conversions for: Local WCT (Local Source Stats)...", flush=True)
records = []
t_start = time.time()
count = 0

eps = 1e-3
I_D = torch.eye(1024, device=device)

for tgt_spk in target_speakers:
    tm = target_models[tgt_spk]
    prof_ecapa = tm['prof_ecapa']
    prof_wavlm_sv = tm['prof_wavlm_sv']
    
    for s_item in src_data:
        xs = s_item['feat']
        duration = s_item['duration']
        w_src = s_item['wav']
        
        xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
        xs_n = torch.nn.functional.normalize(xs_t, dim=1)
        
        t0 = time.perf_counter()
        
        # Soft assignment to shared clusters
        sim_mat = xs_n @ cents_shared_norm.t()
        w_dyn = torch.softmax(sim_mat * beta_val, dim=1)
        
        x_hat = torch.zeros_like(xs_t)
        
        for k in range(K_clusters):
            wk = w_dyn[:, k:k+1]
            mass_k = wk.sum()
            if mass_k < 1e-4:
                continue
                
            # LOCAL SOURCE MEAN
            mu_X_k = (xs_t * wk).sum(dim=0) / mass_k
            xc = xs_t - mu_X_k.unsqueeze(0)
            
            # LOCAL SOURCE COVARIANCE
            cov_X_k = torch.mm(xc.t(), xc * wk) / mass_k
            
            # LOCAL SOURCE WHITENING MATRIX: W_{X,k} = Sigma_{X,k}^{-1/2}
            evals_X, evecs_X = torch.linalg.eigh(cov_X_k + eps * I_D)
            evals_X = torch.clamp(evals_X, min=eps)
            inv_C_X_k = evecs_X @ torch.diag(1.0 / torch.sqrt(evals_X)) @ evecs_X.t()
            
            # LOCAL TARGET COLORING MATRIX: C_{Y,k} = Sigma_{Y,k}^{1/2}
            C_Y_k = tm['cl_C_Y_shared'][k]
            
            # LOCAL WCT LINEAR MAP: M_k = W_{X,k} @ C_{Y,k}
            M_k = torch.mm(inv_C_X_k, C_Y_k)
            
            # LOCAL WCT TRANSPORT: T_k(x) = (x - mu_{X,k}) @ M_k + mu_{Y,k}
            T_k = torch.mm(xc, M_k) + tm['cl_mu_Y_shared'][k].unsqueeze(0)
            
            x_hat += wk * T_k
            
        t1 = time.perf_counter()
        rtf = (t1 - t0) / duration
        
        x_hat_np = x_hat.cpu().numpy()
        wc = voc(x_hat_np)
        
        # Metrics
        # 1. Wav2Vec2
        hyp_w2v2 = tr_w2v2(wc)
        cer_w2v2 = jiwer.cer(s_item['ref_w2v2'], hyp_w2v2) * 100.0
        wer_w2v2 = jiwer.wer(s_item['ref_w2v2'], hyp_w2v2) * 100.0
        
        # 2. Whisper-Small
        hyp_wh = tr_whisper(wc)
        cer_wh_gt = jiwer.cer(s_item['ref_gt'], hyp_wh) * 100.0
        wer_wh_gt = jiwer.wer(s_item['ref_gt'], hyp_wh) * 100.0
        cer_wh_src = jiwer.cer(s_item['ref_wh_src'], hyp_wh) * 100.0
        wer_wh_src = jiwer.wer(s_item['ref_wh_src'], hyp_wh) * 100.0
        
        # 3. Speaker Similarities
        sim_ecapa = torch.dot(prof_ecapa, emb_ecapa(wc)).item()
        sim_wavlm = torch.dot(prof_wavlm_sv, emb_wavlm_sv(wc)).item()
        
        # 4. SQUIM (MOS, PESQ, STOI)
        wc_2d = wc.view(1, -1).to(device)
        w_ref_2d = w_src.view(1, -1).to(device)
        with torch.no_grad():
            mos = squim_subj(wc_2d, w_ref_2d).item()
            stoi, pesq, _ = squim_obj(wc_2d)
            pesq_val = pesq.item()
            stoi_val = stoi.item()
            
        # 5. Trajectory Jitter
        _, rjit = compute_trajectory_jitter(x_hat_np, xs)
        
        records.append({
            'Method': 'Local WCT (Local Source Stats)',
            'Approach_Type': 'Piecewise WCT (Local Source & Target)',
            'Target_Speaker': tgt_spk,
            'Source_Speaker': s_item['spk'],
            'CER_W2V2': cer_w2v2,
            'WER_W2V2': wer_w2v2,
            'CER_Whisper': cer_wh_gt,
            'WER_Whisper': wer_wh_gt,
            'CER_Whisper_src': cer_wh_src,
            'WER_Whisper_src': wer_wh_src,
            'Sim_ECAPA': sim_ecapa,
            'Sim_WavLM': sim_wavlm,
            'MOS': mos,
            'PESQ': pesq_val,
            'STOI': stoi_val,
            'Jitter': rjit,
            'RTF': rtf
        })
        
        count += 1
        if count % 50 == 0:
            print(f"  Completed {count}/200 conversions...", flush=True)

df_res = pd.DataFrame(records)
out_csv = proj_root / 'output/tables/local_wct_local_stats_200_detail.csv'
df_res.to_csv(out_csv, index=False)
print(f"\nSaved detailed results to: {out_csv}", flush=True)

# 4. Summary Statistics
print("\n" + "="*120, flush=True)
print("=== ABLATION RESULTS: STATISTICAL PROGRESSION FROM WCT TO MONGE-KANTOROVICH (N=200) ===", flush=True)
print("="*120, flush=True)

# Load existing benchmark summary for direct comparison
df_sum = pd.read_csv(proj_root / 'output/tables/unified_200_benchmark_summary.csv')

comp_methods = [
    'Classic WCT',
    'Soft Local WCT (Ours)',
    'Local WCT (Local Source Stats)',
    'Local Wasserstein (LWT - Ours)',
    'Boosted LWT (alpha=1.5 - Ours)'
]

new_row = {
    'Method': 'Local WCT (Local Source Stats)',
    'Approach_Type': 'Piecewise WCT (Local Source & Target)',
    'CER_mean': df_res['CER_W2V2'].mean(),
    'CER_std': df_res['CER_W2V2'].std(),
    'WER_mean': df_res['WER_W2V2'].mean(),
    'WER_std': df_res['WER_W2V2'].std(),
    'CER_Whisper_mean': df_res['CER_Whisper'].mean(),
    'CER_Whisper_std': df_res['CER_Whisper'].std(),
    'WER_Whisper_mean': df_res['WER_Whisper'].mean(),
    'WER_Whisper_std': df_res['WER_Whisper'].std(),
    'CER_Whisper_src_mean': df_res['CER_Whisper_src'].mean(),
    'CER_Whisper_src_std': df_res['CER_Whisper_src'].std(),
    'WER_Whisper_src_mean': df_res['WER_Whisper_src'].mean(),
    'WER_Whisper_src_std': df_res['WER_Whisper_src'].std(),
    'Sim_mean': df_res['Sim_ECAPA'].mean(),
    'Sim_std': df_res['Sim_ECAPA'].std(),
    'Sim_WavLM_mean': df_res['Sim_WavLM'].mean(),
    'Sim_WavLM_std': df_res['Sim_WavLM'].std(),
    'MOS_mean': df_res['MOS'].mean(),
    'MOS_std': df_res['MOS'].std(),
    'PESQ_mean': df_res['PESQ'].mean(),
    'PESQ_std': df_res['PESQ'].std(),
    'STOI_mean': df_res['STOI'].mean(),
    'STOI_std': df_res['STOI'].std(),
    'Jitter_mean': df_res['Jitter'].mean(),
    'Jitter_std': df_res['Jitter'].std(),
    'RTF_mean': df_res['RTF'].mean(),
    'RTF_std': df_res['RTF'].std()
}

all_rows = []
for m in comp_methods:
    if m == 'Local WCT (Local Source Stats)':
        all_rows.append(new_row)
    else:
        matched = df_sum[df_sum['Method'] == m]
        if len(matched) > 0:
            all_rows.append(matched.iloc[0].to_dict())

df_ablation = pd.DataFrame(all_rows)
out_ablation_csv = proj_root / 'output/tables/ablation_wct_to_lwt_summary.csv'
df_ablation.to_csv(out_ablation_csv, index=False)
print(f"Saved ablation comparison summary to: {out_ablation_csv}\n", flush=True)

header = f"{'Method':<32} | {'Wh. CER (GT)':<12} {'Wh. WER (GT)':<12} | {'Sim (ECAPA)':<12} {'Sim (WavLM)':<12} | {'MOS':<8} {'Jitter':<8} {'RTF':<8}"
print(header, flush=True)
print("-" * len(header), flush=True)

for _, r in df_ablation.iterrows():
    m = r['Method']
    c_wh = f"{r['CER_Whisper_mean']:.2f}%"
    w_wh = f"{r['WER_Whisper_mean']:.2f}%"
    s_ecapa = f"{r['Sim_mean']:.3f}"
    s_wavlm = f"{r['Sim_WavLM_mean']:.3f}"
    mos = f"{r['MOS_mean']:.2f}"
    jit = f"{r['Jitter_mean']:.2f}x"
    rtf = f"{r['RTF_mean']:.4f}"
    print(f"{m:<32} | {c_wh:<12} {w_wh:<12} | {s_ecapa:<12} {s_wavlm:<12} | {mos:<8} {jit:<8} {rtf:<8}", flush=True)

print("="*120, flush=True)
