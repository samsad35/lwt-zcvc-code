import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torchaudio.functional as F
import sys
import warnings
import librosa
from sklearn.neighbors import NearestNeighbors

import logging
logging.getLogger("speechbrain").setLevel(logging.ERROR)

sys.path.append(str(Path(__file__).parent.parent))
from une import LibriSpeech

device = "cpu"

def extract_features_and_pitch(wav_path, wavlm_large):
    wav, sr = torchaudio.load(str(wav_path))
    if sr != 16000:
        wav = F.resample(wav, sr, 16000)
    wav_np = wav.squeeze().numpy()
    
    f0, _, _ = librosa.pyin(wav_np, fmin=50, fmax=500, sr=16000, frame_length=1024, hop_length=320)
    f0 = np.nan_to_num(f0, nan=0.0)
    
    wav_t = wav.to(device)
    if wav_t.dim() == 1:
        wav_t = wav_t.unsqueeze(0)
    with torch.no_grad():
        features_large, _ = wavlm_large.extract_features(wav_t, output_layer=6)
        x = features_large.squeeze(0).cpu().numpy()
        
    min_len = min(len(f0), x.shape[0])
    return x[:min_len], f0[:min_len], wav_np

def vocode(features, hifigan, device):
    with torch.inference_mode():
        feats_tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        return hifigan(feats_tensor).squeeze(0).cpu()

def apply_custom_pitch(original_f0, mode="flat"):
    new_f0 = np.zeros_like(original_f0)
    voiced = original_f0 > 0
    if not np.any(voiced):
        return new_f0
        
    mean_pitch = 180.0
    N = len(original_f0)
    
    if mode == "flat":
        new_f0[voiced] = mean_pitch
    elif mode == "high":
        new_f0[voiced] = mean_pitch * 1.5
    elif mode == "low":
        new_f0[voiced] = mean_pitch * 0.7
    elif mode == "wave":
        t = np.linspace(0, 4 * np.pi, N)
        wave = mean_pitch + np.sin(t) * (mean_pitch * 0.4)
        new_f0[voiced] = wave[voiced]
        
    return new_f0

def main():
    warnings.filterwarnings("ignore")
    print("1. Loading Models...")
    wavlm_large = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device).eval()
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, prematched=True, device=device)
    hifigan.eval()
    
    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    librispeech = LibriSpeech(root=dataset_path, ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    
    tgt_spk = "121"
    tgt_paths = df_all[df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")]['path'].tolist()[:20]
    
    print("\n2. Extracting Target Pool & Pitch...")
    X_tgt_list, F0_tgt_list = [], []
    for p in tqdm(tgt_paths, desc="Target Data"):
        x, f0, _ = extract_features_and_pitch(p, wavlm_large)
        X_tgt_list.append(x)
        F0_tgt_list.append(f0)
        
    X_tgt_pool = np.concatenate(X_tgt_list, axis=0)
    F0_tgt_pool = np.concatenate(F0_tgt_list, axis=0)
    
    knn = NearestNeighbors(n_neighbors=1, metric="cosine", n_jobs=-1)
    knn.fit(X_tgt_pool)

    # Pick exactly ONE source utterance
    df_src = df_all[~df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")].sample(n=1, random_state=42)
    src_path = df_src['path'].tolist()[0]
    
    print(f"\n3. Extracting Source: {src_path}")
    X_1, f0_src, wav_src = extract_features_and_pitch(src_path, wavlm_large)
    P_1 = f0_src.reshape(-1, 1)
    
    # kNN Alignment
    _, indices = knn.kneighbors(X_1)
    X_2 = X_tgt_pool[indices.squeeze()]
    P_2 = F0_tgt_pool[indices.squeeze()].reshape(-1, 1)
    
    # Orthogonalization
    W_p1, _, _, _ = np.linalg.lstsq(P_1, X_1, rcond=None)
    W_p2, _, _, _ = np.linalg.lstsq(P_2, X_2, rcond=None)
    
    X_1_perp = X_1 - np.matmul(P_1, W_p1)
    X_2_perp = X_2 - np.matmul(P_2, W_p2)
    
    # SVD
    X_block = np.concatenate([X_1_perp, X_2_perp], axis=1)
    U, S_vals, Vh = np.linalg.svd(X_block, full_matrices=False)
    
    RANK_r = 32
    C = np.matmul(U[:, :RANK_r], np.diag(S_vals[:RANK_r]))
    S_2 = Vh[:RANK_r, 1024:] 
    
    print("\n4. Generating Pitch-Controlled Samples...")
    out_dir = Path("pitch_control_demo")
    out_dir.mkdir(exist_ok=True)
    
    # Save original source
    torchaudio.save(out_dir / "0_original_source.wav", torch.tensor(wav_src).unsqueeze(0), 16000)
    
    modes = ["flat", "high", "low", "wave"]
    for i, mode in enumerate(modes, 1):
        # Create artificial pitch
        f0_custom = apply_custom_pitch(f0_src, mode=mode)
        P_custom = f0_custom.reshape(-1, 1)
        
        # Generation: Content * Speaker + Pitch
        X_conv = np.matmul(C, S_2) + np.matmul(P_custom, W_p2)
        
        # Vocode and save
        wav_conv = vocode(X_conv, hifigan, device)
        torchaudio.save(out_dir / f"{i}_{mode}_pitch.wav", wav_conv.view(1, -1), 16000)
        print(f"   -> Saved {mode} pitch sample.")
        
    print("\nAll done! Check the 'pitch_control_demo' directory.")

if __name__ == "__main__":
    main()
