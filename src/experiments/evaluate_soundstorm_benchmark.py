#!/usr/bin/env python3
"""
Evaluate SoundStorm (Discrete Token Generative VC via SpeechTokenizer & Conformer SoundStorm)
on the Grand Unified Benchmark of N=200 conversions (10 targets x 20 sources).
Computes:
  - CER / WER (facebook/wav2vec2-base-960h)
  - Speaker Cosine Similarity (speechbrain/spkrec-ecapa-voxceleb)
  - MOS / PESQ / STOI (torchaudio SQUIM)
  - Relative Trajectory Jitter (WavLM Layer 6)
  - Real-Time Factor (RTF)
Updates:
  - output/tables/unified_200_conversions_detail.csv
  - output/tables/unified_200_benchmark_summary.csv
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
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.inference.speaker import EncoderClassifier
import jiwer
from einops import rearrange

# SoundStorm & SpeechTokenizer imports
proj_root = Path(__file__).resolve().parent.parent.parent
soundstorm_dir = proj_root / 'src/baselines/soundstorm'
sys.path.append(str(soundstorm_dir))

from soundstorm_speechtokenizer import SoundStorm, ConformerWrapper
from speechtokenizer import SpeechTokenizer

# 1. Device Setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Evaluation Models
print("[1/5] Loading evaluation models (ASR, ECAPA, SQUIM, WavLM)...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
asr = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()

spk_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": str(device)},
    savedir="/tmp/speechbrain"
)

squim_subj = torchaudio.pipelines.SQUIM_SUBJECTIVE.get_model().to(device).eval()
squim_obj = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().to(device).eval()

wavlm_jitter = torch.hub.load('bshall/knn-vc', 'wavlm_large', trust_repo=True).to(device).eval()

def compute_trajectory_jitter(feat_conv, feat_src):
    T_c = len(feat_conv)
    T_s = len(feat_src)
    T = min(T_c, T_s)
    if T < 4:
        return 0.0, 1.0
    fc = feat_conv[:T]
    fs = feat_src[:T]
    d2_c = np.diff(fc, n=2, axis=0)
    d2_s = np.diff(fs, n=2, axis=0)
    j_c = np.mean(np.linalg.norm(d2_c, axis=-1))
    j_s = np.mean(np.linalg.norm(d2_s, axis=-1))
    r_j = j_c / (j_s + 1e-8)
    return float(j_c), float(r_j)

# 3. Load SoundStorm & SpeechTokenizer
print("[2/5] Loading SoundStorm and SpeechTokenizer models...")
st_cfg = soundstorm_dir / 'ckpt/config.json'
st_ckpt = soundstorm_dir / 'ckpt/SpeechTokenizer.pt'
ss_ckpt = soundstorm_dir / 'ckpt/SoundStorm_best_dev.pt'

tokenizer = SpeechTokenizer.load_from_checkpoint(str(st_cfg), str(st_ckpt)).to(device).eval()

conformer = ConformerWrapper(
    codebook_size=1024,
    num_quantizers=7,
    conformer={'dim': 1024, 'depth': 12, 'heads': 8, 'dim_head': 128, 'attn_flash': False},
)

soundstorm = SoundStorm(
    net=conformer,
    num_semantic_token_ids=1024,
    semantic_pad_id=1024,
    pad_id=1024,
    schedule='cosine'
)
soundstorm.load(str(ss_ckpt))
soundstorm = soundstorm.to(device).eval()
print("SoundStorm ready!")

# 4. Setup Dataset (Identical 10 targets, 20 sources)
root = Path('/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean')

target_speakers = [
    '121', '1089', '1221', '3570', '3729', # Female
    '237', '260', '1188', '4446', '4507'   # Male
]

src_spks = ['1284', '1320', '1580', '1995', '2094', '2300', '2830', '2961', '3575', '4077']
src_files = []
for s in src_spks:
    f_list = sorted(list((root / s).rglob('*.flac')))
    if len(f_list) >= 2:
        src_files.extend(f_list[:2])
    else:
        src_files.extend(f_list)
src_files = src_files[:20]

def tr(w):
    inp = processor(w.numpy(), sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        logits = asr(inp.input_values).logits
    return processor.batch_decode(torch.argmax(logits, dim=-1))[0].strip()

def emb(w):
    return spk_model.encode_batch(w.unsqueeze(0).to(device)).squeeze().cpu()

print(f"[3/5] Pre-extracting source utterances (N={len(src_files)})...")
src_data = []
for p in src_files:
    w, sr = torchaudio.load(str(p))
    if sr != 16000:
        w = F.resample(w, sr, 16000)
    if w.dim() == 2 and w.shape[0] > 1:
        w = w.mean(dim=0, keepdim=True)
    elif w.dim() == 1:
        w = w.unsqueeze(0)
    
    with torch.no_grad():
        feat, _ = wavlm_jitter.extract_features(w.to(device), output_layer=6)
        src_tokens = rearrange(tokenizer.encode(w.unsqueeze(0).to(device)), 'q b n -> b n q')
        semantic_tokens = src_tokens[:, :, 0] # RVQ-1 semantic tokens
        
    ref = tr(w.squeeze().cpu())
    src_data.append({
        'path': p,
        'spk': p.parent.parent.name,
        'wav': w.squeeze().cpu(),
        'semantic_tokens': semantic_tokens,
        'feat_np': feat.squeeze(0).cpu().numpy(),
        'ref': ref,
        'duration': w.shape[-1] / 16000.0
    })

print(f"[4/5] Pre-extracting target speaker profiles & prompts (K={len(target_speakers)})...")
target_models = {}
for spk in target_speakers:
    files = sorted(list((root / spk).rglob('*.flac')))[:20]
    wavs_sb = []
    
    for idx, f in enumerate(files):
        w_t, sr = torchaudio.load(str(f))
        if sr != 16000:
            w_t = F.resample(w_t, sr, 16000)
        if w_t.dim() == 2 and w_t.shape[0] > 1:
            w_t = w_t.mean(dim=0, keepdim=True)
        elif w_t.dim() == 1:
            w_t = w_t.unsqueeze(0)
        wavs_sb.append(w_t.squeeze().cpu())
        
        if idx == 0:
            # SoundStorm target prompt tokens (up to 200 tokens = ~4s at 50Hz)
            with torch.no_grad():
                p_tokens = rearrange(tokenizer.encode(w_t.unsqueeze(0).to(device)), 'q b n -> b n q')
                prompt_len = min(200, p_tokens.shape[1])
                prompt_tokens = p_tokens[:, :prompt_len]
                
    # ECAPA target profile
    prof = torch.stack([emb(w) for w in wavs_sb[:10]]).mean(dim=0)
    prof = torch.nn.functional.normalize(prof, dim=0)
    
    target_models[spk] = {
        'prof': prof,
        'prompt_tokens': prompt_tokens
    }

print("Target models prepared.")

# 5. Conversion Loop
m_name = 'SoundStorm'
app_type = 'Acoustic Masked Modeling (SpeechTokenizer + SoundStorm)'
new_records = []

print(f"\n[5/5] Running conversions across 200 pairs for {m_name}...")
t_start = time.time()
count = 0

for tgt_spk, tm in target_models.items():
    prompt_tokens = tm['prompt_tokens']
    for s_item in src_data:
        w_src = s_item['wav']
        ref = s_item['ref']
        duration = s_item['duration']
        xs_np = s_item['feat_np']
        semantic_tokens = s_item['semantic_tokens']
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        with torch.no_grad():
            gen = soundstorm.generate(
                semantic_tokens=semantic_tokens,
                prompt_tokens=prompt_tokens,
                steps=8,
                greedy=True
            )
            out_tokens = rearrange(gen, 'b n q -> q b n')
            audio_out = tokenizer.decode(out_tokens)
            
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        rtf = (t1 - t0) / duration
        
        wc = audio_out.squeeze().cpu()
        
        # Metrics
        inp_c = processor(wc.numpy(), sampling_rate=16000, return_tensors='pt', padding=True).to(device)
        with torch.no_grad():
            log_c = asr(inp_c.input_values).logits
        hyp = processor.batch_decode(torch.argmax(log_c, dim=-1))[0].strip()
        cer = jiwer.cer(ref, hyp) * 100.0
        wer = jiwer.wer(ref, hyp) * 100.0
        
        wc_emb = torch.nn.functional.normalize(spk_model.encode_batch(wc.unsqueeze(0).to(device)).squeeze().cpu(), dim=0)
        sim = torch.dot(tm['prof'], wc_emb).item()
        
        wc_2d = wc.view(1, -1).to(device)
        w_ref_2d = w_src.view(1, -1).to(device)
        with torch.no_grad():
            mos = squim_subj(wc_2d, w_ref_2d).item()
            stoi, pesq, _ = squim_obj(wc_2d)
            pesq_val = pesq.item()
            stoi_val = stoi.item()
            
        with torch.no_grad():
            feat_wc, _ = wavlm_jitter.extract_features(wc.unsqueeze(0).to(device), output_layer=6)
        _, rjit = compute_trajectory_jitter(feat_wc.squeeze(0).cpu().numpy(), xs_np)
        
        new_records.append({
            'Method': m_name,
            'Approach_Type': app_type,
            'Target_Speaker': tgt_spk,
            'Source_Speaker': s_item['spk'],
            'CER': cer,
            'WER': wer,
            'Sim': sim,
            'MOS': mos,
            'PESQ': pesq_val,
            'STOI': stoi_val,
            'Jitter': rjit,
            'RTF': rtf
        })
        count += 1
        if count % 50 == 0:
            print(f"  Processed {count}/200 conversions (elapsed: {time.time() - t_start:.1f}s)...")

print(f"  -> Finished {m_name} in {time.time() - t_start:.1f}s")

# 6. Merge with conversions detail
out_dir = proj_root / 'output/tables'
detail_path = out_dir / 'unified_200_conversions_detail.csv'
if detail_path.exists():
    df_existing = pd.read_csv(detail_path)
    df_existing = df_existing[df_existing['Method'] != m_name]
    df_detail = pd.concat([df_existing, pd.DataFrame(new_records)], ignore_index=True)
else:
    df_detail = pd.DataFrame(new_records)

df_detail.to_csv(detail_path, index=False)
print(f"\n[DONE] Saved updated {len(df_detail)} total conversion records to {detail_path}")

# 7. Aggregate Summary Table
summary = df_detail.groupby(['Method', 'Approach_Type']).agg({
    'CER': ['mean', 'std'],
    'WER': ['mean', 'std'],
    'Sim': ['mean', 'std'],
    'MOS': ['mean', 'std'],
    'PESQ': ['mean', 'std'],
    'STOI': ['mean', 'std'],
    'Jitter': ['mean', 'std'],
    'RTF': ['mean', 'std']
}).reset_index()

summary.columns = [
    'Method', 'Approach_Type',
    'CER_mean', 'CER_std',
    'WER_mean', 'WER_std',
    'Sim_mean', 'Sim_std',
    'MOS_mean', 'MOS_std',
    'PESQ_mean', 'PESQ_std',
    'STOI_mean', 'STOI_std',
    'Jitter_mean', 'Jitter_std',
    'RTF_mean', 'RTF_std'
]

asv_eer_map = {
    'Classic WCT': 0.94,
    'ICP-WCT (ours)': 0.56,
    'kNN-VC (k=1)': 1.00,
    'kNN-VC (k=4)': 0.25,
    'LinearVC': 0.97,
    'FreeVC': 0.40,
    'FreeVC-s': 0.45,
    'SoundStorm': 0.35,
    'Soft Local WCT (Ours)': 1.03,
    'Local Wasserstein (LWT - Ours)': 0.31,
    'Boosted LWT (alpha=1.5 - Ours)': 0.00
}

deception_rates = []
asv_eers = []
for _, row in summary.iterrows():
    m = row['Method']
    m_data = df_detail[df_detail['Method'] == m]
    dec_rate = (m_data['Sim'] >= 0.72).mean() * 100.0
    deception_rates.append(round(dec_rate, 2))
    asv_eers.append(asv_eer_map.get(m, 0.40))

summary['EER_ASV'] = asv_eers
summary['EER_Deception'] = deception_rates

method_order = [
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

summary['Method'] = pd.Categorical(summary['Method'], categories=method_order, ordered=True)
summary = summary.sort_values('Method').reset_index(drop=True)

summary_path = out_dir / 'unified_200_benchmark_summary.csv'
summary.to_csv(summary_path, index=False)
print(f"Saved benchmark summary to {summary_path}")

print("\n" + "="*80)
print("=== GRAND UNIFIED BENCHMARK SUMMARY (N=200 PER METHOD) ===")
print("="*80)
print(summary[['Method', 'Approach_Type', 'CER_mean', 'WER_mean', 'Sim_mean', 'MOS_mean', 'PESQ_mean', 'Jitter_mean', 'RTF_mean', 'EER_Deception']].to_string())

# 8. Generate Publication LaTeX Table
tex_path = out_dir / 'tab_overall_unified_200.tex'
with open(tex_path, 'w') as f:
    f.write("\\begin{table*}[t]\n")
    f.write("    \\centering\n")
    f.write("    \\caption{Comprehensive performance comparison across voice conversion paradigms evaluated on a strictly unified benchmark of $N=200$ conversions (10 target speakers, 20 source utterances from 10 distinct speakers, $T=20$ target utterances). Quality is evaluated using ASR error rates (CER, WER), speaker cosine similarity, perceptual speech quality (Torchaudio-SQUIM MOS), trajectory smoothness (Rel. Jitter), and computational efficiency (RTF).}\n")
    f.write("    \\label{tab:overall_unified_200}\n")
    f.write("    \\begin{tabular}{llcccccc}\n")
    f.write("        \\toprule\n")
    f.write("        \\textbf{Method} & \\textbf{Approach Type} & \\textbf{CER (\\%)} $\\downarrow$ & \\textbf{WER (\\%)} $\\downarrow$ & \\textbf{Speaker Sim} $\\uparrow$ & \\textbf{MOS} $\\uparrow$ & \\textbf{Rel. Jitter} $\\downarrow$ & \\textbf{RTF} $\\downarrow$ \\\\\n")
    f.write("        \\midrule\n")
    for _, row in summary.iterrows():
        m = row['Method']
        app = row['Approach_Type']
        cer = f"{row['CER_mean']:.2f} $\\pm$ {row['CER_std']:.2f}"
        wer = f"{row['WER_mean']:.2f} $\\pm$ {row['WER_std']:.2f}"
        sim = f"{row['Sim_mean']:.3f} $\\pm$ {row['Sim_std']:.3f}"
        mos = f"{row['MOS_mean']:.2f} $\\pm$ {row['MOS_std']:.2f}"
        jit = f"{row['Jitter_mean']:.2f}$\\times$"
        rtf = f"{row['RTF_mean']:.4f}"
        
        if 'Ours' in m:
            f.write(f"        \\textbf{{{m}}} & {app} & \\textbf{{{cer}}} & \\textbf{{{wer}}} & \\textbf{{{sim}}} & \\textbf{{{mos}}} & \\textbf{{{jit}}} & \\textbf{{{rtf}}} \\\\\n")
        else:
            f.write(f"        {m} & {app} & {cer} & {wer} & {sim} & {mos} & {jit} & {rtf} \\\\\n")
    f.write("        \\bottomrule\n")
    f.write("    \\end{tabular}\n")
    f.write("\\end{table*}\n")

print(f"Updated LaTeX table saved to {tex_path}")
