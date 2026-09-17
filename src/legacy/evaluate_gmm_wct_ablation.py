import torch
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
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

logging.getLogger("speechbrain").setLevel(logging.ERROR)
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.inference.speaker import EncoderClassifier

sys.path.append(str(Path(__file__).parent.parent))
from une import LibriSpeech

device = "cpu"

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
    
    k = min(k, len(eigenvalues))
    eigenvalues = np.maximum(eigenvalues[:k], 1e-5)
    eigenvectors = eigenvectors[:, :k]
    
    D_inv_half = np.diag(1.0 / np.sqrt(eigenvalues))
    D_half = np.diag(np.sqrt(eigenvalues))
    
    W = eigenvectors @ D_inv_half @ eigenvectors.T
    C = eigenvectors @ D_half @ eigenvectors.T
    return mu, W, C

def main():
    warnings.filterwarnings("ignore")
    print("1. Loading Models...")
    wavlm_large = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device).eval()
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, prematched=True, device=device)
    hifigan.eval()
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()
    speaker_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device}, savedir="/tmp/speechbrain")

    def transcribe(wav):
        if torch.is_tensor(wav):
            wav = wav.numpy()
        inputs = processor(wav, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = asr_model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(predicted_ids)[0]
    
    def get_embedding(wav):
        with torch.no_grad():
            wav_t = torch.tensor(wav).float().unsqueeze(0)
            return speaker_model.encode_batch(wav_t).squeeze()

    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    librispeech = LibriSpeech(root=dataset_path, ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    
    tgt_spk = "121"
    
    N_COMPONENTS = 128
    BETA = 2.0
    
    df_src = df_all[~df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")].sample(n=15, random_state=42)
    src_paths = df_src['path'].tolist()
    
    # Pre-extract sources to save time
    print("\n2. Pre-extracting Source Features...")
    source_data = []
    for p in tqdm(src_paths, desc="Extracting"):
        X_src, w_src = extract_features(p, wavlm_large)
        ref_text = transcribe(w_src)
        source_data.append((X_src, ref_text, w_src))
        
    K_VALUES = [5]
    TGT_SIZES = [30]
    
    print("\n3. Running Ablation Study...")
    
    results = []
    
    for T_size in TGT_SIZES:
        print(f"\n================ TARGET SIZE: {T_size} ==================")
        tgt_paths = df_all[df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")]['path'].tolist()[:T_size]
        X_tgt_list, tgt_wavs = [], []
        for p in tgt_paths:
            x, w = extract_features(p, wavlm_large)
            X_tgt_list.append(x)
            tgt_wavs.append(w)
            
        X_tgt_pool = np.concatenate(X_tgt_list, axis=0)
        
        tgt_embs = [get_embedding(w.numpy()) for w in tgt_wavs[:min(10, T_size)]]
        tgt_profile = torch.stack(tgt_embs).mean(dim=0)
        tgt_profile = torch.nn.functional.normalize(tgt_profile, dim=0)
        
        for K in K_VALUES:
            print(f"  -> Testing K = {K}")
            kmeans = KMeans(n_clusters=K, random_state=42, n_init=5)
            tgt_labels = kmeans.fit_predict(X_tgt_pool)
            tgt_centroids = kmeans.cluster_centers_
            
            cluster_matrices = []
            for k in range(K):
                X_k = X_tgt_pool[tgt_labels == k]
                if len(X_k) > 10:
                    mu_k, _, C_k = get_wct_matrices(X_k, k=N_COMPONENTS)
                else:
                    mu_k, _, C_k = get_wct_matrices(X_tgt_pool, k=N_COMPONENTS) # fallback
                cluster_matrices.append((mu_k, C_k))
                
            cers = []
            sims = []
            
            for X_src, ref_text, _ in source_data:
                k_src = min(N_COMPONENTS, X_src.shape[0] - 1)
                mu_src, W_src, _ = get_wct_matrices(X_src, k=k_src)
                X_whitened = np.matmul(X_src - mu_src, W_src)
                
                X_colored = np.zeros_like(X_src)
                dists = pairwise_distances(X_src, tgt_centroids, metric='euclidean')
                
                # Numerically stable softmax
                logits = -BETA * dists
                logits -= np.max(logits, axis=1, keepdims=True)
                weights = np.exp(logits)
                weights = weights / weights.sum(axis=1, keepdims=True)
                
                for i in range(X_src.shape[0]):
                    x_w = X_whitened[i]
                    x_c = np.zeros_like(x_w)
                    for k in range(K):
                        mu_k, C_k = cluster_matrices[k]
                        w = weights[i, k]
                        x_k = np.matmul(C_k, x_w) + mu_k
                        x_c += w * x_k
                    X_colored[i] = x_c
                    
                wav_conv = vocode(X_colored, hifigan, device)
                conv_text = transcribe(wav_conv.squeeze().numpy())
                cer = jiwer.cer(ref_text, conv_text)
                cers.append(cer)
                
                conv_emb = get_embedding(wav_conv.squeeze().numpy())
                conv_emb = torch.nn.functional.normalize(conv_emb, dim=0)
                sim = torch.dot(tgt_profile, conv_emb).item()
                sims.append(sim)
                
            avg_cer = np.mean(cers) * 100
            avg_sim = np.mean(sims)
            print(f"     [T={T_size}, K={K}] CER: {avg_cer:.2f}% | Sim: {avg_sim:.4f}")
            results.append({"T_size": T_size, "K": K, "CER": avg_cer, "Sim": avg_sim})
            
    print("\n\n================ ABLATION SUMMARY ================")
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()
