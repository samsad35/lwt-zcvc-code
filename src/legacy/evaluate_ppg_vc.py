import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torchaudio.functional as F
import sys
import warnings
import jiwer
import logging

logging.getLogger("speechbrain").setLevel(logging.ERROR)

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.inference.speaker import EncoderClassifier

sys.path.append(str(Path(__file__).parent.parent))
from une import LibriSpeech

device = "cpu"

def extract_features(wav_path, wavlm_large, asr_model, processor):
    wav, sr = torchaudio.load(str(wav_path))
    wav = wav.to(device)
    if sr != 16000:
        wav = F.resample(wav, sr, 16000)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    with torch.no_grad():
        features_large, _ = wavlm_large.extract_features(wav, output_layer=6)
        x = features_large.squeeze(0).cpu().numpy()
        
        inputs = processor(wav.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt").to(device)
        y = asr_model(inputs.input_values).logits.squeeze(0).cpu().numpy()
        
    min_T = min(x.shape[0], y.shape[0])
    return x[:min_T], y[:min_T], wav.squeeze()

def learn_cca_full(X_list, Y_list, device):
    X_frames = np.concatenate(X_list, axis=0)
    Y_frames = np.concatenate(Y_list, axis=0)
    X_mean = np.mean(X_frames, axis=0)
    Y_mean = np.mean(Y_frames, axis=0)
    Xt = torch.tensor(X_frames - X_mean, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y_frames - Y_mean, dtype=torch.float32, device=device)
    Q_x, R_x = torch.linalg.qr(Xt)
    Q_y, R_y = torch.linalg.qr(Yt)
    C = torch.matmul(Q_x.T, Q_y)
    U, S, Vh = torch.linalg.svd(C)
    Wy = torch.linalg.solve(R_y, Vh.T)
    return Y_mean, Wy.cpu().numpy()

def get_universal_content(Y_features, Y_mean, Wy):
    Y_c = Y_features - Y_mean
    return np.matmul(Y_c, Wy)

def vocode(features, hifigan, device):
    with torch.inference_mode():
        feats_tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        return hifigan(feats_tensor).squeeze(0).cpu()

def main():
    warnings.filterwarnings("ignore")
    print("1. Loading Models (WavLM-Large, Wav2Vec2-PPG, HiFi-GAN, ASR, Speaker Verification)...")
    
    wavlm_large = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device).eval()
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, prematched=True, device=device)
    hifigan.eval()
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()
    speaker_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device}, savedir="/tmp/speechbrain")

    def transcribe(wav):
        inputs = processor(wav.numpy(), sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = asr_model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(predicted_ids)[0]

    def get_embedding(wav):
        with torch.no_grad():
            return speaker_model.encode_batch(wav.squeeze().unsqueeze(0)).squeeze()

    print("\n2. Loading Dataset & Computing CCA Space (PPGs)...")
    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    librispeech = LibriSpeech(root=dataset_path, ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    
    df_cca = df_all.sample(n=250, random_state=42)
    X_list, Y_list = [], []
    for p in tqdm(df_cca['path'], desc="CCA Data"):
        x, y, _ = extract_features(p, wavlm_large, asr_model, processor)
        X_list.append(x)
        Y_list.append(y)
    Y_mean, Wy_full = learn_cca_full(X_list, Y_list, device)
    
    tgt_spk = "121"
    tgt_paths = df_all[df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")]['path'].tolist()[:25]
    
    X_tgt_list, Y_tgt_list, tgt_wavs = [], [], []
    for p in tgt_paths:
        x, y, w = extract_features(p, wavlm_large, asr_model, processor)
        X_tgt_list.append(x)
        Y_tgt_list.append(y)
        tgt_wavs.append(w)
        
    X_tgt_all = np.concatenate(X_tgt_list, axis=0)
    Y_tgt_all = np.concatenate(Y_tgt_list, axis=0)
    
    tgt_embs = [get_embedding(w) for w in tgt_wavs[:10]]
    tgt_profile = torch.stack(tgt_embs).mean(dim=0)
    tgt_profile = torch.nn.functional.normalize(tgt_profile, dim=0)

    df_src = df_all[~df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")].sample(n=25, random_state=42)
    src_paths = df_src['path'].tolist()
    print(f"   Collected {len(src_paths)} source utterances.")
    
    source_data = []
    for p in src_paths:
        x, y, w = extract_features(p, wavlm_large, asr_model, processor)
        ref_text = transcribe(w)
        source_data.append({"x": x, "y": y, "wav": w, "text": ref_text})
        
    print("\n3. Learning LINEAR Decoder (W_speaker) for Target...")
    # Wy_full has max 32 dims because PPG is 32 dims
    dim = Wy_full.shape[1]
    Wy_sub = Wy_full
    
    Z_c_tgt = get_universal_content(Y_tgt_all, Y_mean, Wy_sub)
    
    # Solve ||X - Z_c W_speaker||^2
    W_speaker, residuals, rank, s = np.linalg.lstsq(Z_c_tgt, X_tgt_all, rcond=None)
    
    print("\n4. Running Conversion Pipeline...")
    cers = []
    sims = []
    
    for src in tqdm(source_data, desc="Evaluating PPG-Linear"):
        Z_c_src = get_universal_content(src["y"], Y_mean, Wy_sub)
        
        # Factorised projection
        X_converted = np.matmul(Z_c_src, W_speaker)
        
        wav_conv = vocode(X_converted, hifigan, device)
        
        conv_text = transcribe(wav_conv)
        cer = jiwer.cer(src["text"], conv_text)
        cers.append(cer)
        
        conv_emb = get_embedding(wav_conv)
        conv_emb = torch.nn.functional.normalize(conv_emb, dim=0)
        sim = torch.dot(tgt_profile, conv_emb).item()
        sims.append(sim)
            
    avg_cer = np.mean(cers) * 100
    avg_sim = np.mean(sims)
    
    print(f"\n================ CCA-PPG RESULTS ({dim}D) ================")
    print(f" CER (%):             {avg_cer:.2f}%")
    print(f" Similarity (Cosine): {avg_sim:.4f}")
    print("========================================================")
    
if __name__ == "__main__":
    main()
