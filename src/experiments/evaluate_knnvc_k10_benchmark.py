import os, sys, time, torch, torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
import jiwer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from speechbrain.inference.speaker import EncoderClassifier
from transformers import AutoFeatureExtractor, WavLMForXVector
import torchaudio.functional as AF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.core.models import compute_trajectory_jitter

device = 'cuda:0'

print("================================================================================")
print("=== EVALUATING kNN-VC (k=10) ON UNIFIED 200 BENCHMARK ===")
print("================================================================================")

# 1. Models
print("Loading feature extractor (WavLM-Large) and vocoder (HiFi-GAN)...")
wavlm = torch.hub.load('bshall/knn-vc', 'wavlm_large', trust_repo=True, device=device).eval()
hifigan, _ = torch.hub.load('bshall/knn-vc', 'hifigan_wavlm', trust_repo=True, prematched=True, device=device)
hifigan.eval()

print("Loading evaluation models...")
processor_w2v2 = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
asr_w2v2 = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h').to(device).eval()

whisper_processor = WhisperProcessor.from_pretrained('openai/whisper-small')
whisper_model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small').to(device).eval()
normalizer = BasicTextNormalizer()

spk_model_ecapa = EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb', 
    run_opts={'device': device}, 
    savedir='/tmp/speechbrain'
)

wavlm_sv_id = 'microsoft/wavlm-base-sv'
extractor_wavlm_sv = AutoFeatureExtractor.from_pretrained(wavlm_sv_id)
wavlm_sv_model = WavLMForXVector.from_pretrained(wavlm_sv_id).to(device).eval()

squim_subj = torchaudio.pipelines.SQUIM_SUBJECTIVE.get_model().to(device).eval()

def ext(p):
    w, sr = torchaudio.load(str(p))
    w = w.to(device)
    if sr != 16000: w = AF.resample(w, sr, 16000)
    if w.dim() == 1: w = w.unsqueeze(0)
    elif w.dim() == 2 and w.shape[0] > 1: w = w.mean(dim=0, keepdim=True)
    with torch.no_grad(): feat, _ = wavlm.extract_features(w, output_layer=6)
    return feat.squeeze(0).cpu().numpy(), w.squeeze().cpu()

def voc(f):
    with torch.inference_mode():
        if isinstance(f, np.ndarray): f = torch.tensor(f, dtype=torch.float32, device=device)
        return hifigan(f.unsqueeze(0)).squeeze().cpu()

def tr_w2v2(w):
    inp = processor_w2v2(w.squeeze().numpy(), sampling_rate=16000, return_tensors='pt', padding=True).to(device)
    with torch.no_grad(): log = asr_w2v2(inp.input_values).logits
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
        return torch.nn.functional.normalize(
            spk_model_ecapa.encode_batch(w.squeeze().float().unsqueeze(0).to(device)).squeeze().cpu(), 
            dim=0
        )

def emb_wavlm_sv(w):
    audio_np = w.squeeze().cpu().numpy() if torch.is_tensor(w) else np.squeeze(w)
    inp = extractor_wavlm_sv(audio_np, sampling_rate=16000, return_tensors='pt').to(device)
    with torch.no_grad():
        emb = wavlm_sv_model(**inp).embeddings.squeeze().cpu()
    return torch.nn.functional.normalize(emb, dim=0)

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
    if len(f_list) >= 2: src_files.extend(f_list[:2])
    else: src_files.extend(f_list)
src_files = src_files[:20]

# Ground-truth transcriptions
trans_map = {}
for s in src_spks:
    for tf in (root / s).rglob('*.trans.txt'):
        for line in tf.read_text().splitlines():
            if line.strip():
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    trans_map[parts[0]] = parts[1]

print("Pre-extracting source features and references...")
src_data = []
for p in src_files:
    feat, wav = ext(p)
    stem = p.stem
    gt_text = trans_map.get(stem, "")
    gt_norm = normalizer(gt_text)
    ref_w2v2 = tr_w2v2(wav)
    ref_wh = tr_whisper(wav)
    src_data.append({
        'path': p,
        'stem': stem,
        'spk': p.parent.parent.name,
        'feat': feat,
        'wav': wav,
        'gt_text': gt_text,
        'gt_norm': gt_norm,
        'ref_w2v2': ref_w2v2,
        'ref_wh': ref_wh,
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
    prof_ecapa = torch.stack([emb_ecapa(w) for w in wavs[:10]]).mean(dim=0)
    prof_ecapa = torch.nn.functional.normalize(prof_ecapa, dim=0)
    prof_wavlm_sv = torch.stack([emb_wavlm_sv(w) for w in wavs[:10]]).mean(dim=0)
    prof_wavlm_sv = torch.nn.functional.normalize(prof_wavlm_sv, dim=0)
    target_models[spk] = {
        'Y_t': Y_t,
        'Y_norm': Y_norm,
        'prof_ecapa': prof_ecapa,
        'prof_wavlm_sv': prof_wavlm_sv,
        'ref_wav': wavs[0]
    }

print("\nRunning kNN-VC (k=10) across 200 conversions...")
k10_records = []
t_start = time.time()
count = 0

for tgt_spk, tm in target_models.items():
    prof_ecapa = tm['prof_ecapa']
    prof_wavlm_sv = tm['prof_wavlm_sv']
    tgt_ref_wav = tm['ref_wav']
    
    for s_item in src_data:
        xs = s_item['feat']
        duration = s_item['duration']
        
        xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
        xs_n = torch.nn.functional.normalize(xs_t, dim=1)
        
        t0 = time.perf_counter()
        sim_mat = torch.mm(xs_n, tm['Y_norm'].t())
        top10_idx = torch.topk(sim_mat, k=10, dim=1).indices
        x_hat = torch.mean(tm['Y_t'][top10_idx], dim=1)
        x_hat_np = x_hat.cpu().numpy()
        t1 = time.perf_counter()
        rtf = (t1 - t0) / duration
        
        _, rjit = compute_trajectory_jitter(x_hat_np, xs)
        wc = voc(x_hat_np)
        
        # ASR Wav2Vec2
        hyp_w2v2 = tr_w2v2(wc)
        cer_w2v2_src = jiwer.cer(s_item['ref_w2v2'], hyp_w2v2) * 100.0
        wer_w2v2_src = jiwer.wer(s_item['ref_w2v2'], hyp_w2v2) * 100.0
        cer_w2v2_gt = jiwer.cer(s_item['gt_text'], hyp_w2v2) * 100.0 if s_item['gt_text'] else None
        wer_w2v2_gt = jiwer.wer(s_item['gt_text'], hyp_w2v2) * 100.0 if s_item['gt_text'] else None
        
        # ASR Whisper
        hyp_wh = tr_whisper(wc)
        cer_wh_gt = jiwer.cer(s_item['gt_norm'], hyp_wh) * 100.0 if s_item['gt_norm'] else None
        wer_wh_gt = jiwer.wer(s_item['gt_norm'], hyp_wh) * 100.0 if s_item['gt_norm'] else None
        cer_wh_src = jiwer.cer(s_item['ref_wh'], hyp_wh) * 100.0
        wer_wh_src = jiwer.wer(s_item['ref_wh'], hyp_wh) * 100.0
        
        # Speaker Similarity
        emb_e = emb_ecapa(wc)
        sim_ecapa = torch.dot(prof_ecapa, emb_e).item()
        emb_w = emb_wavlm_sv(wc)
        sim_wavlm = torch.dot(prof_wavlm_sv, emb_w).item()
        
        # MOS SQUIM
        with torch.no_grad():
            wc_dev = wc.unsqueeze(0).to(device)
            ref_dev = tgt_ref_wav.unsqueeze(0).to(device)
            mos_val = squim_subj(wc_dev, ref_dev).item()
            
        k10_records.append({
            'Method': 'kNN-VC (k=10)',
            'Approach_Type': 'Local Instance',
            'Target_Speaker': tgt_spk,
            'Source_Speaker': s_item['spk'],
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
            'MOS': mos_val,
            'Rel_Jitter': rjit,
            'RTF': rtf
        })
        count += 1
        if count % 50 == 0:
            print(f"  Processed {count}/200 conversions in {time.time()-t_start:.1f}s...", flush=True)

df_k10 = pd.DataFrame(k10_records)
out_csv = Path('output/tables/knnvc_k10_unified_200_results.csv')
df_k10.to_csv(out_csv, index=False)

print("\n=======================================================")
print("             kNN-VC (k=10) BENCHMARK RESULTS           ")
print("=======================================================")
print(f"Wav2Vec2 CER (Source Audio)   : {df_k10['CER_Wav2Vec2'].mean():.2f}% ± {df_k10['CER_Wav2Vec2'].std():.2f}%")
print(f"Wav2Vec2 WER (Source Audio)   : {df_k10['WER_Wav2Vec2'].mean():.2f}% ± {df_k10['WER_Wav2Vec2'].std():.2f}%")
print(f"Wav2Vec2 CER (Human GT Text)  : {df_k10['CER_Wav2Vec2_gt'].mean():.2f}% ± {df_k10['CER_Wav2Vec2_gt'].std():.2f}%")
print(f"Wav2Vec2 WER (Human GT Text)  : {df_k10['WER_Wav2Vec2_gt'].mean():.2f}% ± {df_k10['WER_Wav2Vec2_gt'].std():.2f}%")
print(f"Whisper CER (Human GT Text)   : {df_k10['CER_Whisper'].mean():.2f}% ± {df_k10['CER_Whisper'].std():.2f}%")
print(f"Whisper WER (Human GT Text)   : {df_k10['WER_Whisper'].mean():.2f}% ± {df_k10['WER_Whisper'].std():.2f}%")
print(f"Sim (ECAPA-TDNN)              : {df_k10['Sim_ECAPA'].mean():.4f} ± {df_k10['Sim_ECAPA'].std():.4f}")
print(f"Sim (WavLM-SV)                : {df_k10['Sim_WavLM'].mean():.4f} ± {df_k10['Sim_WavLM'].std():.4f}")
print(f"MOS (TorchAudio SQUIM)        : {df_k10['MOS'].mean():.2f} ± {df_k10['MOS'].std():.2f}")
print(f"Rel. Jitter (RLA)             : {df_k10['Rel_Jitter'].mean():.2f}x ± {df_k10['Rel_Jitter'].std():.2f}x")
print(f"RTF                           : {df_k10['RTF'].mean():.6f}")
print("=======================================================")
