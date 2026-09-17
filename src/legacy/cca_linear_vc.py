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

def extract_features(wav_path, wavlm_large, wavlm_asr):
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

def learn_cca_mapping(X_list, Y_list, device, dim=128):
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
    
    Wy_sub = Wy[:, :dim].cpu().numpy()
    
    return Y_mean, Wy_sub

def get_universal_content(Y_features, Y_mean, Wy_sub):
    """ Project ASR features into the pure CCA 128-D canonical space """
    Y_c = Y_features - Y_mean
    Z_content = np.matmul(Y_c, Wy_sub) # [T, 128]
    return Z_content

def vocode(features, hifigan, device):
    with torch.inference_mode():
        feats_tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        wav_hat = hifigan(feats_tensor).squeeze(0)
        return wav_hat.cpu().squeeze()

def main():
    warnings.filterwarnings("ignore")
    print("1. Loading Models...")
    
    wavlm_large = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device).eval()
    wavlm_asr = WavLM(model_name="patrickvonplaten/wavlm-libri-clean-100h-base-plus", device=device)
    
    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    librispeech = LibriSpeech(root=dataset_path, ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    
    print("\n2. Computing Universal CCA Space...")
    # Utiliser 300 fichiers pour avoir un dictionnaire phonétique universel beaucoup plus riche
    df_cca = df_all.sample(n=300, random_state=42)
    X_list, Y_list = [], []
    for p in tqdm(df_cca['path'], desc="CCA Data"):
        x, y = extract_features(p, wavlm_large, wavlm_asr)
        X_list.append(x)
        Y_list.append(y)
        
    Y_mean, Wy_sub = learn_cca_mapping(X_list, Y_list, device, dim=768)
    
    print("\n3. Learning Target Decoder Matrix (W_tgt) for 2 Speakers...")
    speakers = list(df_all['path'].apply(lambda x: Path(x).parts[-3]).unique())
    targets = random.sample(speakers, 2)
        
    W_target_dict = {}
    
    for tgt_spk in targets:
        print(f"  -> Training Decoder for Speaker {tgt_spk}...")
        # Get ALL utterances for the target speaker to learn a robust W (thousands of frames instead of hundreds)
        tgt_paths = df_all[df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")]['path'].tolist()
        
        Z_c_list = []
        X_tgt_list = []
        for p in tgt_paths:
            x, y = extract_features(p, wavlm_large, wavlm_asr)
            Z_c = get_universal_content(y, Y_mean, Wy_sub)
            Z_c_list.append(Z_c)
            X_tgt_list.append(x)
            
        Z_c_all = np.concatenate(Z_c_list, axis=0)
        X_tgt_all = np.concatenate(X_tgt_list, axis=0)
        
        # Add bias column for linear regression
        Z_c_bias = np.hstack([Z_c_all, np.ones((Z_c_all.shape[0], 1))])
        
        # Learn W_tgt: (769, 1024) using standard Least Squares (OLS)
        W_tgt, _, _, _ = np.linalg.lstsq(Z_c_bias, X_tgt_all, rcond=None)
        W_target_dict[tgt_spk] = W_tgt
        print(f"     [+] W_tgt learned! Shape: {W_tgt.shape}")
        
    print("\n4. Performing Universal CCA-LinearVC (Zero-Shot Conversion)...")
    out_dir = Path("voice_conversion_ultimate")
    out_dir.mkdir(exist_ok=True)
    
    # We will pick 2 random sources and cross-convert them to our 2 trained targets
    sources = random.sample([s for s in speakers if s not in targets], 2)
    
    generated = []
    
    for src_spk in sources:
        src_path = df_all[df_all['path'].astype(str).str.contains(f"/{src_spk}/")]['path'].tolist()[0]
        x_src, y_src = extract_features(src_path, wavlm_large, wavlm_asr)
        Z_content_src = get_universal_content(y_src, Y_mean, Wy_sub)
        
        # Save original source for reference
        generated.append({
            'name': f"source_{src_spk}.wav",
            'features': x_src
        })
        
        for tgt_spk in targets:
            W_tgt = W_target_dict[tgt_spk]
            
            # Linear Conversion ! Perfect Temporal Continuity !
            Z_c_src_bias = np.hstack([Z_content_src, np.ones((Z_content_src.shape[0], 1))])
            X_converted = np.matmul(Z_c_src_bias, W_tgt)
            
            generated.append({
                'name': f"converted_{src_spk}_to_{tgt_spk}.wav",
                'features': X_converted
            })
            
            # Save original target reference (1st utterance used for training)
            tgt_ref_path = df_all[df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")]['path'].tolist()[0]
            x_tgt, _ = extract_features(tgt_ref_path, wavlm_large, wavlm_asr)
            generated.append({
                'name': f"target_ref_{tgt_spk}.wav",
                'features': x_tgt
            })

    print("\n5. Vocoding outputs to audio (loading HiFi-GAN)...")
    del wavlm_large, wavlm_asr
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, prematched=True, device=device)
    hifigan.eval()
    
    # Remove duplicates from references
    seen = set()
    for item in generated:
        if item['name'] not in seen:
            seen.add(item['name'])
            wav = vocode(item['features'], hifigan, device)
            torchaudio.save(out_dir / item['name'], wav.unsqueeze(0), 16000)

    print(f"\n🎉 Done! The ULTIMATE CCA-LinearVC files are saved in {out_dir.absolute()}")

if __name__ == "__main__":
    main()
