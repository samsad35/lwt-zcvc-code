import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torchaudio.functional as F
import sys
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
        h_x = features_large.squeeze(0).cpu().numpy()
        h_y = wavlm_asr(str(wav_path))[12].squeeze(0).cpu().numpy()
    min_T = min(h_x.shape[0], h_y.shape[0])
    return h_x[:min_T], h_y[:min_T]

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
    
    # We don't need Wx for conversion, just Wy
    Wy = torch.linalg.solve(R_y, Vh.T)
    return Y_mean, Wy.cpu().numpy()

def get_universal_content(Y_features, Y_mean, Wy_sub):
    Y_c = Y_features - Y_mean
    Z_content = np.matmul(Y_c, Wy_sub)
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
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, prematched=True, device=device)
    hifigan.eval()
    
    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    librispeech = LibriSpeech(root=dataset_path, ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    
    print("\n2. Computing Universal CCA Space (Full 768 Dimensions)...")
    df_cca = df_all.sample(n=300, random_state=42)
    X_list, Y_list = [], []
    for p in tqdm(df_cca['path'], desc="CCA Data"):
        x, y = extract_features(p, wavlm_large, wavlm_asr)
        X_list.append(x)
        Y_list.append(y)
        
    Y_mean, Wy_full = learn_cca_full(X_list, Y_list, device)
    
    print("\n3. Preparing Source & Target data...")
    speakers = list(df_all['path'].apply(lambda x: Path(x).parts[-3]).unique())
    targets = [spk for spk in speakers if spk == "121"]
    sources = [spk for spk in speakers if spk == "8224"] # A male source
    if not targets: targets = [speakers[0]]
    if not sources: sources = [speakers[1]]
    tgt_spk = targets[0]
    src_spk = sources[0]
    
    print(f"   Source (Male): {src_spk}  -> Target (Female): {tgt_spk}")
    
    tgt_paths = df_all[df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")]['path'].tolist()
    X_tgt_list, Y_tgt_list = [], []
    for p in tgt_paths:
        x, y = extract_features(p, wavlm_large, wavlm_asr)
        X_tgt_list.append(x)
        Y_tgt_list.append(y)
    X_tgt_all = np.concatenate(X_tgt_list, axis=0)
    Y_tgt_all = np.concatenate(Y_tgt_list, axis=0)
    
    src_path = df_all[df_all['path'].astype(str).str.contains(f"/{src_spk}/")]['path'].tolist()[0]
    X_src, Y_src = extract_features(src_path, wavlm_large, wavlm_asr)
    
    out_dir = Path("voice_conversion_dim_experiment")
    out_dir.mkdir(exist_ok=True)
    
    # Save references
    torchaudio.save(out_dir / "0_source_ref.wav", vocode(X_src, hifigan, device).unsqueeze(0), 16000)
    torchaudio.save(out_dir / "0_target_ref.wav", vocode(X_tgt_list[0], hifigan, device).unsqueeze(0), 16000)
    
    dims_to_test = [64, 128, 256, 512, 768]
    print("\n4. Running The Great Bottleneck Experiment...")
    
    for dim in dims_to_test:
        print(f"   -> Testing Dimension = {dim}")
        
        # 1. Truncate CCA space
        Wy_sub = Wy_full[:, :dim]
        
        # 2. Get content for target and learn W
        Z_c_tgt = get_universal_content(Y_tgt_all, Y_mean, Wy_sub)
        Z_c_tgt_bias = np.hstack([Z_c_tgt, np.ones((Z_c_tgt.shape[0], 1))])
        
        W_tgt, _, _, _ = np.linalg.lstsq(Z_c_tgt_bias, X_tgt_all, rcond=None)
        
        # 3. Convert Source
        Z_c_src = get_universal_content(Y_src, Y_mean, Wy_sub)
        Z_c_src_bias = np.hstack([Z_c_src, np.ones((Z_c_src.shape[0], 1))])
        
        X_converted = np.matmul(Z_c_src_bias, W_tgt)
        
        # 4. Vocode and save
        wav_conv = vocode(X_converted, hifigan, device)
        torchaudio.save(out_dir / f"converted_dim_{dim}.wav", wav_conv.unsqueeze(0), 16000)

    print(f"\n🎉 Experiment finished! Listen to the files in {out_dir.absolute()}")

if __name__ == "__main__":
    main()
