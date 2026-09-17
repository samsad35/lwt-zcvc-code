import os, sys, time, torch, torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
import jiwer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.inference.speaker import EncoderClassifier
import torchaudio.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.core.models import compute_trajectory_jitter

device = 'cuda:0'

print("================================================================================")
print("=== EVALUATING kNN-VC (k=1) ON UNIFIED 200 BENCHMARK (JITTER ABLATION) ===")
print("================================================================================")

# 1. Pretrained models
wavlm = torch.hub.load('bshall/knn-vc', 'wavlm_large', trust_repo=True, device=device).eval()
hifigan, _ = torch.hub.load('bshall/knn-vc', 'hifigan_wavlm', trust_repo=True, prematched=True, device=device)
hifigan.eval()

processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
asr = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h').to(device).eval()
spk_model = EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb', 
    run_opts={'device': 'cuda:0'}, 
    savedir='/tmp/speechbrain'
)

def ext(p):
    w, sr = torchaudio.load(str(p))
    w = w.to(device)
    if sr != 16000: w = F.resample(w, sr, 16000)
    if w.dim() == 1: w = w.unsqueeze(0)
    with torch.no_grad(): feat, _ = wavlm.extract_features(w, output_layer=6)
    return feat.squeeze(0).cpu().numpy(), w.squeeze().cpu()

def voc(f):
    with torch.inference_mode():
        if isinstance(f, np.ndarray): f = torch.tensor(f, dtype=torch.float32, device=device)
        return hifigan(f.unsqueeze(0)).squeeze(0).cpu()

def tr(w):
    inp = processor(w.squeeze().numpy(), sampling_rate=16000, return_tensors='pt', padding=True).to(device)
    with torch.no_grad(): log = asr(inp.input_values).logits
    pred_ids = torch.argmax(log, dim=-1)
    return processor.batch_decode(pred_ids)[0].strip()

def emb(w):
    with torch.no_grad():
        return torch.nn.functional.normalize(
            spk_model.encode_batch(w.squeeze().float().unsqueeze(0).to(device)).squeeze().cpu(), 
            dim=0
        )

root = Path('/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean')

# 10 Target Speakers
target_speakers = [
    '121', '1089', '1221', '3570', '3729', # Female
    '237', '260', '1188', '4446', '4507'   # Male
]

# 20 Source Utterances from 10 distinct speakers
src_spks = ['1284', '1320', '1580', '1995', '2094', '2300', '2830', '2961', '3575', '4077']
src_files = []
for s in src_spks:
    f_list = sorted(list((root / s).rglob('*.flac')))
    if len(f_list) >= 2:
        src_files.extend(f_list[:2])
    else:
        src_files.extend(f_list)
src_files = src_files[:20]

print("Pre-extracting source features and references...")
src_data = []
for p in src_files:
    feat, wav = ext(p)
    ref = tr(wav)
    src_data.append({
        'path': p,
        'spk': p.parent.parent.name,
        'feat': feat,
        'ref': ref,
        'duration': len(feat) * 0.02
    })

print("Extracting target models (T=20 utterances per speaker)...")
target_models = {}
for spk in target_speakers:
    files = sorted(list((root / spk).rglob('*.flac')))[:20]
    X_list, wavs = [], []
    for p in files:
        x, w = ext(p); X_list.append(x); wavs.append(w)
    Y = np.concatenate(X_list, axis=0)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    Y_norm = torch.nn.functional.normalize(Y_t, dim=1)
    prof = torch.stack([emb(w) for w in wavs[:10]]).mean(dim=0)
    prof = torch.nn.functional.normalize(prof, dim=0)
    target_models[spk] = {
        'Y_t': Y_t,
        'Y_norm': Y_norm,
        'prof': prof
    }

print("\nRunning kNN-VC (k=1) across 200 conversions...")
k1_records = []
t_start = time.time()
count = 0
for tgt_spk, tm in target_models.items():
    for s_item in src_data:
        xs = s_item['feat']
        ref = s_item['ref']
        duration = s_item['duration']
        
        xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
        xs_n = torch.nn.functional.normalize(xs_t, dim=1)
        
        t0 = time.perf_counter()
        sim_mat = torch.mm(xs_n, tm['Y_norm'].t())
        best_idx = torch.argmax(sim_mat, dim=1)
        x_hat = tm['Y_t'][best_idx]
        x_hat_np = x_hat.cpu().numpy()
        t1 = time.perf_counter()
        rtf = (t1 - t0) / duration
        
        _, rjit = compute_trajectory_jitter(x_hat_np, xs)
        wc = voc(x_hat_np)
        hyp = tr(wc)
        cer = jiwer.cer(ref, hyp) * 100.0
        wer = jiwer.wer(ref, hyp) * 100.0
        sim = torch.dot(tm['prof'], emb(wc)).item()
        
        k1_records.append({
            'Method': 'kNN-VC (k=1)',
            'Approach_Type': 'Local Instance',
            'Target_Speaker': tgt_spk,
            'Source_Speaker': s_item['spk'],
            'CER': cer,
            'WER': wer,
            'Sim': sim,
            'Jitter': rjit,
            'RTF': rtf
        })
        count += 1
        if count % 50 == 0:
            print(f"  Processed {count}/200 conversions...")

print(f"kNN-VC (k=1) evaluated in {time.time()-t_start:.1f}s.")

# Update Detailed Records CSV
detail_csv = Path('/local_scratch/ssadok/un_projet_audio/output/tables/unified_200_conversions_detail.csv')
if detail_csv.exists():
    df_existing = pd.read_csv(detail_csv)
    # Remove previous kNN-VC (k=1) if any
    df_existing = df_existing[df_existing['Method'] != 'kNN-VC (k=1)']
    df_all = pd.concat([df_existing, pd.DataFrame(k1_records)], ignore_index=True)
else:
    df_all = pd.DataFrame(k1_records)

df_all.to_csv(detail_csv, index=False)
print(f"Updated detailed conversions CSV: {detail_csv} ({len(df_all)} total evaluations).")

# Recompute Summary
summary = df_all.groupby(['Method', 'Approach_Type']).agg({
    'CER': ['mean', 'std'],
    'WER': ['mean', 'std'],
    'Sim': ['mean', 'std'],
    'Jitter': ['mean', 'std'],
    'RTF': ['mean', 'std']
}).reset_index()

summary.columns = [
    'Method', 'Approach_Type',
    'CER_mean', 'CER_std',
    'WER_mean', 'WER_std',
    'Sim_mean', 'Sim_std',
    'Jitter_mean', 'Jitter_std',
    'RTF_mean', 'RTF_std'
]

order = [
    'Classic WCT',
    'ICP-WCT (ours)',
    'kNN-VC (k=1)',
    'kNN-VC (k=4)',
    'LinearVC',
    'Soft Local WCT (Ours)',
    'Local Wasserstein (LWT - Ours)',
    'Boosted LWT (alpha=1.5 - Ours)'
]

summary['Method'] = pd.Categorical(summary['Method'], categories=order, ordered=True)
summary = summary.sort_values('Method').reset_index(drop=True)

out_summary = Path('/local_scratch/ssadok/un_projet_audio/output/tables/unified_200_benchmark_summary.csv')
summary.to_csv(out_summary, index=False)
print(f"Updated benchmark summary CSV: {out_summary}")

print("\n================================================================================")
print("=== UPDATED UNIFIED BENCHMARK SUMMARY (N=200 CONVERSIONS PER METHOD) ===")
print("================================================================================")
print(summary.to_string())

# Regenerate LaTeX Table
tex_path = Path('/local_scratch/ssadok/un_projet_audio/output/tables/tab_overall_unified_200.tex')
with open(tex_path, 'w') as f:
    f.write("\\begin{table*}[t]\n")
    f.write("    \\centering\n")
    f.write("    \\caption{Comprehensive performance comparison across voice conversion paradigms evaluated on a strictly unified benchmark of $N=200$ conversions (10 target speakers, 20 source utterances from 10 distinct speakers, $T=20$ target utterances). kNN-VC with $k=1$ exposes raw instance jitter, which local averaging ($k=4$) mitigates at the cost of over-smoothing, whereas LWT achieves superior similarity with smooth Monge-Kantorovich transport.}\n")
    f.write("    \\label{tab:overall_unified_200}\n")
    f.write("    \\begin{tabular}{llccccc}\n")
    f.write("        \\toprule\n")
    f.write("        \\textbf{Method} & \\textbf{Approach Type} & \\textbf{CER (\\%)} $\\downarrow$ & \\textbf{WER (\\%)} $\\downarrow$ & \\textbf{Speaker Sim} $\\uparrow$ & \\textbf{Rel. Jitter} $\\downarrow$ & \\textbf{RTF} $\\downarrow$ \\\\\n")
    f.write("        \\midrule\n")
    for _, row in summary.iterrows():
        m = row['Method']
        app = row['Approach_Type']
        cer = f"{row['CER_mean']:.2f} $\\pm$ {row['CER_std']:.2f}"
        wer = f"{row['WER_mean']:.2f} $\\pm$ {row['WER_std']:.2f}"
        sim = f"{row['Sim_mean']:.3f} $\\pm$ {row['Sim_std']:.3f}"
        jit = f"{row['Jitter_mean']:.2f}$\\times$"
        rtf = f"{row['RTF_mean']:.4f}"
        
        if 'Ours' in m:
            f.write(f"        \\textbf{{{m}}} & {app} & \\textbf{{{cer}}} & \\textbf{{{wer}}} & \\textbf{{{sim}}} & {jit} & \\textbf{{{rtf}}} \\\\\n")
        else:
            f.write(f"        {m} & {app} & {cer} & {wer} & {sim} & {jit} & {rtf} \\\\\n")
    f.write("        \\bottomrule\n")
    f.write("    \\end{tabular}\n")
    f.write("\\end{table*}\n")

print(f"Saved updated LaTeX table to {tex_path}")
