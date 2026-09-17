#!/usr/bin/env python
"""
Evaluate the official LinearVC (Kamper et al., Interspeech 2025)
on the unified 200 benchmark (10 target speakers x 20 source utterances from 10 distinct speakers).

Official LinearVC learns a speaker-to-speaker projection matrix W in R^{1024 x 1024}
by 1-NN matching of source reference speech (up to 8,192 frames) to target reference speech (up to 8,192 frames),
and solving: min_W || X_src W - Y_matched ||_F^2 via least squares.
At test time, the unseen source utterance X_test is projected via: X_conv = X_test @ W.
"""

import os
import sys
import time
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import jiwer
from tqdm import tqdm

from transformers import (
    Wav2Vec2ForCTC, Wav2Vec2Processor,
    WhisperForConditionalGeneration, WhisperProcessor,
    AutoFeatureExtractor, AutoModel
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from speechbrain.inference.speaker import EncoderClassifier

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

proj_root = Path(__file__).resolve().parent.parent.parent
data_root = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")

# 1. Target and Source speakers definition (Strict match with Unified 200 benchmark)
target_speakers = [
    '121', '1089', '1221', '3570', '3729', # Female
    '237', '260', '1188', '4446', '4507'   # Male
]

src_spks = [
    '1284', '1320', '1580', '1995', '2094',
    '2300', '2830', '2961', '3575', '4077'
]

# 2. Text Normalizer for Whisper
whisper_normalizer = BasicTextNormalizer()

def normalize_text_w2v2(text):
    text = text.upper()
    text = re.sub(r'[^A-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Fast cosine distance matching (from official utils.py of LinearVC)
def fast_cosine_dist(source_feats: torch.Tensor, matching_pool: torch.Tensor) -> torch.Tensor:
    """
    Cosine distance between all pairs in source_feats and matching_pool.
    dist = 1 - cos_sim
    """
    s_norm = F.normalize(source_feats, p=2, dim=-1)
    m_norm = F.normalize(matching_pool, p=2, dim=-1)
    return 1.0 - torch.mm(s_norm, m_norm.t())

def compute_trajectory_jitter(x_conv, x_src):
    diffs = np.linalg.norm(x_conv[1:] - x_conv[:-1], axis=1)
    src_diffs = np.linalg.norm(x_src[1:] - x_src[:-1], axis=1)
    return float(np.mean(diffs) / (np.mean(src_diffs) + 1e-8))

def main():
    print("--- [1/6] Loading Models (WavLM, HiFi-GAN, ASR, SV) ---")
    
    # Feature Extractor & Vocoder
    wavlm = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True).to(device).eval()
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, prematched=True, device=device)
    hifigan.eval()

    def ext(path):
        w, sr = torchaudio.load(path)
        if sr != 16000:
            w = torchaudio.functional.resample(w, sr, 16000)
        w = w.to(device)
        if w.dim() == 1: w = w.unsqueeze(0)
        elif w.dim() == 2 and w.shape[0] > 1: w = w.mean(dim=0, keepdim=True)
        with torch.no_grad():
            feat, _ = wavlm.extract_features(w, output_layer=6)
        return feat.squeeze(0), w.squeeze().cpu()

    def voc(feat):
        with torch.inference_mode():
            if isinstance(feat, np.ndarray):
                feat = torch.from_numpy(feat).float()
            feat = feat.to(device)
            if feat.dim() == 2:
                feat = feat.unsqueeze(0)
            return hifigan(feat).squeeze().cpu()

    # Whisper-Small
    wh_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    wh_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device).eval()

    def tr_whisper(wav):
        wav_np = wav.numpy() if isinstance(wav, torch.Tensor) else wav
        inputs = wh_processor(wav_np, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        with torch.no_grad():
            gen_ids = wh_model.generate(input_features, language="en", task="transcribe")
        transcription = wh_processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        return whisper_normalizer(transcription)

    # Wav2Vec2-base
    w2v2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    w2v2_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()

    def tr_w2v2(wav):
        wav_t = wav.to(device) if isinstance(wav, torch.Tensor) else torch.tensor(wav, device=device)
        inputs = w2v2_processor(wav_t.cpu().numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            logits = w2v2_model(inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = w2v2_processor.batch_decode(pred_ids)[0]
        return normalize_text_w2v2(transcription)

    # ECAPA-TDNN Speaker Verification
    ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device}, savedir="/tmp/speechbrain_ecapa")
    def emb_ecapa(wav):
        w = wav.to(device) if isinstance(wav, torch.Tensor) else torch.tensor(wav, device=device)
        if w.dim() == 1: w = w.unsqueeze(0)
        with torch.no_grad():
            e = ecapa.encode_batch(w).squeeze()
        return F.normalize(e, dim=0)

    # WavLM-SV Speaker Verification
    wavlm_sv_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    wavlm_sv_model = AutoModel.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()
    def emb_wavlm_sv(wav):
        w = wav.cpu().numpy() if isinstance(wav, torch.Tensor) else wav
        inputs = wavlm_sv_extractor(w, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = wavlm_sv_model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze()
        return F.normalize(emb, dim=0)

    print("\n--- [2/6] Loading Target Speaker Reference Features (M=8,192 max frames) ---")
    target_models = {}
    for spk in target_speakers:
        files = sorted(list((data_root / spk).rglob("*.flac")))[:20]
        feats_list, wavs_list = [], []
        for f in files:
            ft, w = ext(f)
            feats_list.append(ft)
            wavs_list.append(w)
        Y_all = torch.cat(feats_list, dim=0)
        # Cap at n_frames_max = 8192 exactly as in official LinearVC
        Y_ref = Y_all[:8192]
        
        prof_ecapa = F.normalize(torch.stack([emb_ecapa(w) for w in wavs_list[:10]]).mean(dim=0), dim=0)
        prof_wavlm_sv = F.normalize(torch.stack([emb_wavlm_sv(w) for w in wavs_list[:10]]).mean(dim=0), dim=0)
        
        target_models[spk] = {
            'Y_ref': Y_ref,
            'prof_ecapa': prof_ecapa,
            'prof_wavlm_sv': prof_wavlm_sv
        }
    print(f"Target speakers loaded: {len(target_models)}")

    print("\n--- [3/6] Loading Source Speaker Reference & Test Utterances ---")
    # Load LibriSpeech human ground truth transcriptions from .trans.txt
    trans_map = {}
    for s in src_spks:
        for tf in (data_root / s).rglob('*.trans.txt'):
            for line in tf.read_text().splitlines():
                if line.strip():
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        trans_map[parts[0]] = parts[1]

    # For each source speaker: 2 test utterances + reference utterances (files[2:22] capped at 8192 frames)
    source_speakers_data = {}
    src_test_data = []

    for spk in src_spks:
        files = sorted(list((data_root / spk).rglob("*.flac")))
        test_files = files[:2]
        ref_files = files[2:22]
        
        # Test items
        for tf in test_files:
            ft, w = ext(tf)
            gt_text = trans_map.get(tf.stem, "")
            ref_gt_wh = whisper_normalizer(gt_text)
            ref_gt_w2v2 = normalize_text_w2v2(gt_text)
            ref_src_wh = tr_whisper(w)
            ref_src_w2v2 = tr_w2v2(w)
            src_test_data.append({
                'path': tf,
                'spk': spk,
                'stem': tf.stem,
                'feat': ft,
                'wav': w,
                'ref_gt_wh': ref_gt_wh,
                'ref_gt_w2v2': ref_gt_w2v2,
                'ref_src_wh': ref_src_wh,
                'ref_src_w2v2': ref_src_w2v2,
                'duration': len(ft) * 0.02
            })
            
        # Reference features for training W
        ref_feats = []
        for rf in ref_files:
            ft, _ = ext(rf)
            ref_feats.append(ft)
        X_all = torch.cat(ref_feats, dim=0)
        # Cap at n_frames_max = 8192
        X_ref = X_all[:8192]
        
        source_speakers_data[spk] = {
            'X_ref': X_ref
        }
    print(f"Loaded {len(src_spks)} source speakers with reference pools and {len(src_test_data)} test utterances.")

    print("\n--- [4/6] Training Official LinearVC Projection Matrices (100 pairs) ---")
    # For each (src_spk, tgt_spk), learn W in R^{1024 x 1024}
    # min_W || X_ref W - Y_matched ||_F^2
    W_dict = {}
    t_train_start = time.time()
    for s_spk in src_spks:
        X_ref = source_speakers_data[s_spk]['X_ref']
        for t_spk in target_speakers:
            Y_ref = target_models[t_spk]['Y_ref']
            
            # Fast 1-NN cosine distance matching on GPU
            dists = fast_cosine_dist(X_ref, Y_ref)
            best_idx = torch.argmin(dists, dim=-1)
            Y_matched = Y_ref[best_idx]
            
            # Solve least squares: X_ref @ W = Y_matched directly on GPU
            # Overdetermined system: N ~ 8192 rows, D = 1024 cols
            W = torch.linalg.lstsq(X_ref, Y_matched).solution
            W_dict[(s_spk, t_spk)] = W

    print(f"100 projection matrices W trained in {time.time() - t_train_start:.2f}s!")

    print("\n--- [5/6] Running Official LinearVC Benchmark on 200 Conversions ---")
    results = []
    t_conv_start = time.time()
    
    for t_spk in target_speakers:
        tm = target_models[t_spk]
        prof_ecapa = tm['prof_ecapa']
        prof_wavlm_sv = tm['prof_wavlm_sv']
        
        for s_item in src_test_data:
            s_spk = s_item['spk']
            W = W_dict[(s_spk, t_spk)]
            xs = s_item['feat'] # (T, 1024)
            dur = s_item['duration']
            
            # In official LinearVC: X_conv = X_test @ W
            t0 = time.perf_counter()
            x_conv = torch.mm(xs, W)
            t1 = time.perf_counter()
            rtf = (t1 - t0) / dur
            
            # Vocode
            wc = voc(x_conv)
            
            # ASR Intelligibility
            hyp_wh = tr_whisper(wc)
            cer_wh_gt = jiwer.cer(s_item['ref_gt_wh'], hyp_wh) * 100.0
            wer_wh_gt = jiwer.wer(s_item['ref_gt_wh'], hyp_wh) * 100.0
            cer_wh_src = jiwer.cer(s_item['ref_src_wh'], hyp_wh) * 100.0
            wer_wh_src = jiwer.wer(s_item['ref_src_wh'], hyp_wh) * 100.0
            
            hyp_w2v2 = tr_w2v2(wc)
            cer_w2v2_src = jiwer.cer(s_item['ref_src_w2v2'], hyp_w2v2) * 100.0
            wer_w2v2_src = jiwer.wer(s_item['ref_src_w2v2'], hyp_w2v2) * 100.0
            cer_w2v2_gt = jiwer.cer(s_item['ref_gt_w2v2'], hyp_w2v2) * 100.0
            wer_w2v2_gt = jiwer.wer(s_item['ref_gt_w2v2'], hyp_w2v2) * 100.0
            
            # Speaker Similarity
            emb_e = emb_ecapa(wc)
            sim_ecapa = torch.dot(prof_ecapa, emb_e).item()
            
            emb_w = emb_wavlm_sv(wc)
            sim_wavlm = torch.dot(prof_wavlm_sv, emb_w).item()
            
            # Trajectory Jitter / Path Length Ratio
            jitter = compute_trajectory_jitter(x_conv.cpu().numpy(), xs.cpu().numpy())
            
            results.append({
                'Method': 'LinearVC',
                'Approach_Type': 'Global Linear Projection',
                'Target_Speaker': t_spk,
                'Source_Speaker': s_spk,
                'Source_Stem': s_item['stem'],
                'CER_Wav2Vec2': cer_w2v2_src,
                'WER_Wav2Vec2': wer_w2v2_src,
                'CER_Wav2Vec2_gt': cer_w2v2_gt,
                'WER_Wav2Vec2_gt': wer_w2v2_gt,
                'CER_Whisper': cer_wh_gt,
                'WER_Whisper': wer_wh_gt,
                'CER_Whisper_src': cer_wh_src,
                'WER_Whisper_src': wer_wh_src,
                'Sim_ECAPA': sim_ecapa,
                'Sim_WavLM': sim_wavlm,
                'Rel_Jitter': jitter,
                'RTF': rtf
            })
            
            if len(results) % 50 == 0:
                print(f"  Processed {len(results)}/200 conversions...", flush=True)

    print(f"\nAll 200 conversions finished in {time.time() - t_conv_start:.2f}s!")
    df_new_linear = pd.DataFrame(results)
    
    # Print summary of official LinearVC
    print("\n=======================================================")
    print("      OFFICIAL LINEARVC (KAMPER 2025) RESULTS        ")
    print("=======================================================")
    print(f"Wav2Vec2 CER (Source Audio)   : {df_new_linear['CER_Wav2Vec2'].mean():.2f}% ± {df_new_linear['CER_Wav2Vec2'].std():.2f}%")
    print(f"Wav2Vec2 WER (Source Audio)   : {df_new_linear['WER_Wav2Vec2'].mean():.2f}% ± {df_new_linear['WER_Wav2Vec2'].std():.2f}%")
    print(f"Wav2Vec2 CER (Human GT Text)  : {df_new_linear['CER_Wav2Vec2_gt'].mean():.2f}% ± {df_new_linear['CER_Wav2Vec2_gt'].std():.2f}%")
    print(f"Wav2Vec2 WER (Human GT Text)  : {df_new_linear['WER_Wav2Vec2_gt'].mean():.2f}% ± {df_new_linear['WER_Wav2Vec2_gt'].std():.2f}%")
    print(f"Whisper CER (Human GT Text)   : {df_new_linear['CER_Whisper'].mean():.2f}% ± {df_new_linear['CER_Whisper'].std():.2f}%")
    print(f"Whisper WER (Human GT Text)   : {df_new_linear['WER_Whisper'].mean():.2f}% ± {df_new_linear['WER_Whisper'].std():.2f}%")
    print(f"Whisper CER (Source Audio)    : {df_new_linear['CER_Whisper_src'].mean():.2f}% ± {df_new_linear['CER_Whisper_src'].std():.2f}%")
    print(f"Whisper WER (Source Audio)    : {df_new_linear['WER_Whisper_src'].mean():.2f}% ± {df_new_linear['WER_Whisper_src'].std():.2f}%")
    print(f"Sim (ECAPA-TDNN)              : {df_new_linear['Sim_ECAPA'].mean():.4f} ± {df_new_linear['Sim_ECAPA'].std():.4f}")
    print(f"Sim (WavLM-SV)                : {df_new_linear['Sim_WavLM'].mean():.4f} ± {df_new_linear['Sim_WavLM'].std():.4f}")
    print(f"Rel. Jitter                   : {df_new_linear['Rel_Jitter'].mean():.2f}x ± {df_new_linear['Rel_Jitter'].std():.2f}x")
    print(f"RTF                           : {df_new_linear['RTF'].mean():.6f}")
    print("=======================================================")
    
    # Save results to CSV
    out_linear_csv = proj_root / "output/tables/official_linearvc_unified_200_results.csv"
    df_new_linear.to_csv(out_linear_csv, index=False)
    print(f"Saved results to {out_linear_csv}")

if __name__ == "__main__":
    main()
