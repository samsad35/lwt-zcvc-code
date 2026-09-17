import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torchaudio.functional as F
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).parent.parent))
from une import LibriSpeech

device = "cpu"
output_dir = Path("wct_samples_target_2")
output_dir.mkdir(exist_ok=True)

def extract_features(wav_path, wavlm_large):
    wav, sr = torchaudio.load(str(wav_path))
    wav = wav.to(device)
    if sr != 16000:
        wav = F.resample(wav, sr, 16000)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    with torch.no_grad():
        features_large, _ = wavlm_large.extract_features(wav, output_layer=6)
        x = features_large.squeeze(0).cpu().numpy()
    return x, wav.squeeze()

def vocode(features, hifigan, device):
    with torch.inference_mode():
        feats_tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        return hifigan(feats_tensor).squeeze(0).cpu()

def get_wct_matrices(X, k=128):
    mu = np.mean(X, axis=0)
    X_c = X - mu
    cov = np.cov(X_c, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    eigenvalues = eigenvalues[:k]
    eigenvectors = eigenvectors[:, :k]
    
    eps = 1e-5
    eigenvalues = np.maximum(eigenvalues, eps)
    
    D_inv_half = np.diag(1.0 / np.sqrt(eigenvalues))
    D_half = np.diag(np.sqrt(eigenvalues))
    
    W = eigenvectors @ D_inv_half @ eigenvectors.T
    C = eigenvectors @ D_half @ eigenvectors.T
    return mu, W, C

def main():
    print("Loading Models...")
    wavlm_large = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device).eval()
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, prematched=True, device=device)
    hifigan.eval()
    
    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    librispeech = LibriSpeech(root=dataset_path, ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    
    df_all['spk_id'] = df_all['path'].apply(lambda x: Path(x).parts[-3])
    unique_all_spks = df_all['spk_id'].unique()
    
    # Pick a completely different target speaker
    tgt_spk = unique_all_spks[1] if unique_all_spks[0] == "121" else unique_all_spks[0]
    print(f"New Target Speaker Selected: {tgt_spk}")
    
    tgt_paths = df_all[df_all['spk_id'] == tgt_spk]['path'].tolist()[:20]
    
    X_tgt_list, tgt_wavs = [], []
    for p in tgt_paths:
        x, w = extract_features(p, wavlm_large)
        X_tgt_list.append(x)
        tgt_wavs.append(w)
        
    X_tgt_pool = np.concatenate(X_tgt_list, axis=0)
    
    N_COMPONENTS = 128
    print("Computing Target WCT Matrices...")
    mu_tgt, _, C_tgt = get_wct_matrices(X_tgt_pool, k=N_COMPONENTS)
    
    # Save a target sample for reference
    torchaudio.save(output_dir / "target_reference.wav", tgt_wavs[0].unsqueeze(0), 16000)

    # Extract speaker ID to pick 5 different speakers (male and female mix)
    df_src = df_all[~df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")]
    df_src['spk_id'] = df_src['path'].apply(lambda x: Path(x).parts[-3])
    unique_spks = df_src['spk_id'].unique()
    
    selected_spks = unique_spks[:5]
    src_paths = []
    for spk in selected_spks:
        src_paths.append(df_src[df_src['spk_id'] == spk]['path'].iloc[0])
    
    for i, p in enumerate(src_paths):
        print(f"Converting sample {i+1}...")
        X_src, w_src = extract_features(p, wavlm_large)
        
        torchaudio.save(output_dir / f"source_{i+1}.wav", w_src.unsqueeze(0), 16000)
        
        k_src = min(N_COMPONENTS, X_src.shape[0] - 1)
        mu_src, W_src, _ = get_wct_matrices(X_src, k=k_src)
        
        X_whitened = np.matmul(X_src - mu_src, W_src)
        X_colored = np.matmul(X_whitened, C_tgt) + mu_tgt
        
        wav_conv = vocode(X_colored, hifigan, device)
        
        torchaudio.save(output_dir / f"converted_wct_{i+1}.wav", wav_conv, 16000)
        
    print(f"Done! Samples saved to {output_dir.absolute()}")

if __name__ == "__main__":
    main()
