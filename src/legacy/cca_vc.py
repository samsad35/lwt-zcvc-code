import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torchaudio.functional as F
import sys
import os
import warnings
import random

sys.path.append(str(Path(__file__).parent.parent))
from une import LibriSpeech, WavLM

device = "cpu"

def extract_features(wav_path, wavlm_large, wavlm_asr, vad=False):
    wav, sr = torchaudio.load(str(wav_path))
    wav = wav.to(device)
    if sr != 16000:
        wav = F.resample(wav, sr, 16000)
    
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
        
    with torch.no_grad():
        features_large, _ = wavlm_large.extract_features(wav, output_layer=6)
        h_x = features_large.squeeze(0).cpu().numpy() # [T, 1024]
        
        h_y = wavlm_asr(str(wav_path))[12].squeeze(0).cpu().numpy() # [T, 768]
        
    min_T = min(h_x.shape[0], h_y.shape[0])
    return h_x[:min_T], h_y[:min_T]

def learn_cca_mapping(X_list, Y_list, device, dim=64):
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
    
    Wx = torch.linalg.solve(R_x, U)
    Wy = torch.linalg.solve(R_y, Vh.T)
    
    Wy_sub = Wy[:, :dim]
    
    C_y_train = torch.matmul(Yt, Wy_sub)
    M = torch.linalg.lstsq(C_y_train, Xt).solution
    
    return X_mean, Y_mean, Wy_sub, M

def project_content(Y_features, X_mean, Y_mean, Wy_sub, M, device):
    Y_c = torch.tensor(Y_features - Y_mean, dtype=torch.float32, device=device)
    C_y = torch.matmul(Y_c, Wy_sub)
    X_pred_c = torch.matmul(C_y, M)
    return X_pred_c.cpu().numpy() + X_mean

def vocode(features, hifigan, device):
    with torch.inference_mode():
        feats_tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        wav_hat = hifigan(feats_tensor).squeeze(0)
        return wav_hat.cpu().squeeze()

def main():
    warnings.filterwarnings("ignore")
    print("1. Loading Models (WavLM Large + HiFi-GAN + WavLM ASR)...")
    
    wavlm_large = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device).eval()
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, prematched=True, device=device)
    hifigan.eval()
    
    wavlm_asr = WavLM(model_name="patrickvonplaten/wavlm-libri-clean-100h-base-plus", device=device)
    
    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    librispeech = LibriSpeech(root=dataset_path, ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    
    print("\n2. Extracting features for CCA space learning...")
    df_cca = df_all.sample(n=50, random_state=42)
    X_list, Y_list = [], []
    for p in tqdm(df_cca['path'], desc="CCA Data"):
        x, y = extract_features(p, wavlm_large, wavlm_asr)
        X_list.append(x)
        Y_list.append(y)
        
    print("\n3. Computing CCA Matrix (Dim=128)...")
    xM, yM, Wy, M = learn_cca_mapping(X_list, Y_list, device, dim=128)
    
    print("\n4. Performing 4 Random Voice Conversions via CCA...")
    speakers = list(df_all['path'].apply(lambda x: Path(x).parts[-3]).unique())
    
    out_dir = Path("voice_conversion")
    out_dir.mkdir(exist_ok=True)
    
    for i in range(1, 5):
        src_spk, tgt_spk = random.sample(speakers, 2)
        
        # Safe strict matching for speaker ID
        src_path = next(p for p in df_all['path'] if Path(p).parts[-3] == str(src_spk))
        tgt_path = next(p for p in df_all['path'] if Path(p).parts[-3] == str(tgt_spk))
        
        print(f"\n--- Pair {i} ---")
        print(f"Source (Content): Locuteur {src_spk}")
        print(f"Target (Voice)  : Locuteur {tgt_spk}")
        
        x_src, y_src = extract_features(src_path, wavlm_large, wavlm_asr)
        x_tgt, y_tgt = extract_features(tgt_path, wavlm_large, wavlm_asr)
        
        Z_content_src = project_content(y_src, xM, yM, Wy, M, device)
        Z_content_tgt = project_content(y_tgt, xM, yM, Wy, M, device)
        Z_speaker_tgt = np.mean(x_tgt - Z_content_tgt, axis=0)
        
        X_converted = Z_content_src + Z_speaker_tgt
        
        wav_src_orig = vocode(x_src, hifigan, device)
        wav_tgt_orig = vocode(x_tgt, hifigan, device)
        wav_converted = vocode(X_converted, hifigan, device)
        
        torchaudio.save(out_dir / f"pair{i}_source_{src_spk}.wav", wav_src_orig.unsqueeze(0), 16000)
        torchaudio.save(out_dir / f"pair{i}_target_{tgt_spk}.wav", wav_tgt_orig.unsqueeze(0), 16000)
        torchaudio.save(out_dir / f"pair{i}_converted_{src_spk}_to_{tgt_spk}.wav", wav_converted.unsqueeze(0), 16000)
    
    print(f"\n🎉 Done! Files saved in {out_dir.absolute()}")

if __name__ == "__main__":
    main()
