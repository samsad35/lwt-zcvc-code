import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torchaudio.functional as F
import torch.nn.functional as F_nn
import sys
import os
import warnings

sys.path.append(str(Path(__file__).parent.parent))
from une import LibriSpeech, WavLM

device = "cpu"

def fast_cosine_dist(source_feats, target_feats):
    source_feats = F_nn.normalize(source_feats, dim=1)
    target_feats = F_nn.normalize(target_feats, dim=1)
    dist = 1 - torch.matmul(source_feats, target_feats.T)
    return dist

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
    df_cca = df_all.sample(n=50, random_state=42)
    X_list, Y_list = [], []
    for p in tqdm(df_cca['path'], desc="CCA Data"):
        x, y = extract_features(p, wavlm_large, wavlm_asr)
        X_list.append(x)
        Y_list.append(y)
        
    Y_mean, Wy_sub = learn_cca_mapping(X_list, Y_list, device, dim=128)
    
    print("\n3. Preparing Source & Target Data...")
    src_spk = "8224" # Male
    tgt_spk = "121"  # Female
    
    # Target pool (Use 5 utterances for a good diverse pool of matching target sounds)
    tgt_paths = df_all[df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")]['path'].tolist()[:5]
    
    X_tgt_pool, Y_tgt_pool = [], []
    for p in tgt_paths:
        x, y = extract_features(p, wavlm_large, wavlm_asr)
        X_tgt_pool.append(x)
        Y_tgt_pool.append(y)
    
    X_tgt_all = np.concatenate(X_tgt_pool, axis=0) # [N_tgt, 1024]
    Y_tgt_all = np.concatenate(Y_tgt_pool, axis=0)
    Z_c_tgt = get_universal_content(Y_tgt_all, Y_mean, Wy_sub) # [N_tgt, 128]
    
    # Source utterance
    src_path = df_all[df_all['path'].astype(str).str.contains(f"/{src_spk}/")]['path'].tolist()[0]
    X_src, Y_src = extract_features(src_path, wavlm_large, wavlm_asr) # [N_src, 1024]
    Z_c_src = get_universal_content(Y_src, Y_mean, Wy_sub) # [N_src, 128]
    
    print("\n4. Phase 1: kNN Alignment in Pure CCA Space...")
    Z_src_tensor = torch.tensor(Z_c_src, dtype=torch.float32, device=device)
    Z_tgt_tensor = torch.tensor(Z_c_tgt, dtype=torch.float32, device=device)
    
    dists = fast_cosine_dist(Z_src_tensor, Z_tgt_tensor) # [N_src, N_tgt]
    best_indices = dists.argmin(dim=1).cpu().numpy()
    
    # The corresponding Target sounds in the rich 1024D acoustic space
    X_tgt_matched = X_tgt_all[best_indices] # [N_src, 1024]
    
    print("\n5. Phase 2: Learning the 1024x1024 LinearVC Matrix (W)...")
    X_src_bias = np.hstack([X_src, np.ones((X_src.shape[0], 1))]) # [N_src, 1025]
    
    # We solve: X_src_bias * W = X_tgt_matched
    W, _, _, _ = np.linalg.lstsq(X_src_bias, X_tgt_matched, rcond=None)
    
    print(f"   [+] Matrix W learned! Shape: {W.shape}")
    
    print("\n6. Conversion and Vocoding...")
    # Multiply the continuous Source audio by W to transpose it perfectly to the target!
    X_converted = np.matmul(X_src_bias, W)
    
    out_dir = Path("voice_conversion_final")
    out_dir.mkdir(exist_ok=True)
    
    del wavlm_large, wavlm_asr
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, prematched=True, device=device)
    hifigan.eval()
    
    wav_src = vocode(X_src, hifigan, device)
    wav_tgt = vocode(X_tgt_pool[0], hifigan, device) # ref
    wav_conv = vocode(X_converted, hifigan, device)
    
    torchaudio.save(out_dir / "1_source.wav", wav_src.unsqueeze(0), 16000)
    torchaudio.save(out_dir / "2_target_ref.wav", wav_tgt.unsqueeze(0), 16000)
    torchaudio.save(out_dir / "3_converted_CCA_Aligned_LinearVC.wav", wav_conv.unsqueeze(0), 16000)

    print(f"\n🎉 Done! The final CCA-Aligned LinearVC files are in {out_dir.absolute()}")

if __name__ == "__main__":
    main()
