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

logging.getLogger("speechbrain").setLevel(logging.ERROR)
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.inference.speaker import EncoderClassifier

sys.path.append(str(Path(__file__).parent))
from une import LibriSpeech

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    return x, wav.squeeze().cpu()

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
    print(f"1. Loading Models on {device}...")
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
            wav_t = torch.tensor(wav).float().unsqueeze(0).to(device)
            return speaker_model.encode_batch(wav_t).squeeze().cpu()

    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    librispeech = LibriSpeech(root=dataset_path, ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    
    tgt_spk = "121"
    T_SIZE = 20
    N_COMPONENTS = 128
    
    # Target Pool
    print(f"\n2. Extracting Target Pool (T={T_SIZE})...")
    tgt_paths = df_all[df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")]['path'].tolist()[:T_SIZE]
    X_tgt_list, tgt_wavs = [], []
    for p in tgt_paths:
        x, w = extract_features(p, wavlm_large)
        X_tgt_list.append(x)
        tgt_wavs.append(w)
    X_tgt_pool = np.concatenate(X_tgt_list, axis=0) # [Nt, 1024]
    
    tgt_embs = [get_embedding(w.numpy()) for w in tgt_wavs[:10]]
    tgt_profile = torch.stack(tgt_embs).mean(dim=0)
    tgt_profile = torch.nn.functional.normalize(tgt_profile, dim=0)

    # Source samples (15 utterances)
    df_src = df_all[~df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")].sample(n=15, random_state=42)
    src_paths = df_src['path'].tolist()
    
    print("\n3. Pre-extracting Source Features...")
    source_data = []
    for p in tqdm(src_paths, desc="Extracting Sources"):
        X_src, w_src = extract_features(p, wavlm_large)
        ref_text = transcribe(w_src.numpy())
        source_data.append((X_src, ref_text, w_src))

    results = []

    # ----------------------------------------------------
    # Point 1 on Continuum: Global Classic WCT (K=1)
    # ----------------------------------------------------
    print("\n[Point 1] K = 1 : Classic Global WCT")
    mu_tgt, _, C_tgt = get_wct_matrices(X_tgt_pool, k=N_COMPONENTS)
    cers, sims = [], []
    for X_src, ref_text, _ in source_data:
        k_src = min(N_COMPONENTS, X_src.shape[0] - 1)
        mu_src, W_src, _ = get_wct_matrices(X_src, k=k_src)
        X_w = np.matmul(X_src - mu_src, W_src)
        X_colored = np.matmul(X_w, C_tgt.T) + mu_tgt
        
        wav_conv = vocode(X_colored, hifigan, device)
        conv_text = transcribe(wav_conv.squeeze().numpy())
        cers.append(jiwer.cer(ref_text, conv_text))
        
        conv_emb = get_embedding(wav_conv.squeeze().numpy())
        conv_emb = torch.nn.functional.normalize(conv_emb, dim=0)
        sims.append(torch.dot(tgt_profile, conv_emb).item())
        
    results.append({
        "Configuration": "K = 1 (Classic WCT)",
        "Continuum Regime": "Global Statistical Alignment",
        "CER (%)": np.mean(cers) * 100,
        "Cosine Sim": np.mean(sims)
    })
    print(f"  -> CER: {np.mean(cers)*100:.2f}% | Sim: {np.mean(sims):.4f}")

    # ----------------------------------------------------
    # Points 2 to 4 on Continuum: Soft Local WCT (K = 3, 5, 8)
    # ----------------------------------------------------
    for K in [3, 5, 8]:
        print(f"\n[Point] K = {K} : Soft Local WCT (beta=2.0)")
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=5)
        tgt_labels = kmeans.fit_predict(X_tgt_pool)
        tgt_centroids = kmeans.cluster_centers_
        
        cluster_matrices = []
        for k in range(K):
            X_k = X_tgt_pool[tgt_labels == k]
            mu_k, _, C_k = get_wct_matrices(X_k, k=N_COMPONENTS)
            cluster_matrices.append((mu_k, C_k))
            
        cers, sims = [], []
        for X_src, ref_text, _ in source_data:
            k_src = min(N_COMPONENTS, X_src.shape[0] - 1)
            mu_src, W_src, _ = get_wct_matrices(X_src, k=k_src)
            X_whitened = np.matmul(X_src - mu_src, W_src)
            
            dists = pairwise_distances(X_src, tgt_centroids, metric='euclidean')
            logits = -2.0 * dists
            logits -= np.max(logits, axis=1, keepdims=True)
            weights = np.exp(logits)
            weights = weights / weights.sum(axis=1, keepdims=True)
            
            X_colored = np.zeros_like(X_src)
            for i in range(X_src.shape[0]):
                x_w = X_whitened[i]
                x_c = np.zeros_like(x_w)
                for k in range(K):
                    w = weights[i, k]
                    if w > 1e-4:
                        mu_k, C_k = cluster_matrices[k]
                        x_k = np.matmul(C_k, x_w) + mu_k
                        x_c += w * x_k
                X_colored[i] = x_c
                
            wav_conv = vocode(X_colored, hifigan, device)
            conv_text = transcribe(wav_conv.squeeze().numpy())
            cers.append(jiwer.cer(ref_text, conv_text))
            
            conv_emb = get_embedding(wav_conv.squeeze().numpy())
            conv_emb = torch.nn.functional.normalize(conv_emb, dim=0)
            sims.append(torch.dot(tgt_profile, conv_emb).item())
            
        regime_label = "Intermediate Sweet Spot" if K == 5 else f"Piecewise Local (K={K})"
        results.append({
            "Configuration": f"K = {K} (Soft WCT)",
            "Continuum Regime": regime_label,
            "CER (%)": np.mean(cers) * 100,
            "Cosine Sim": np.mean(sims)
        })
        print(f"  -> CER: {np.mean(cers)*100:.2f}% | Sim: {np.mean(sims):.4f}")

    # ----------------------------------------------------
    # Points 5 & 6 on Continuum: Discrete Piecewise Codebook (K = 16, K = 32, Hard Argmax)
    # ----------------------------------------------------
    for K in [16, 32]:
        print(f"\n[Point] K = {K} : Discrete Piecewise Mapping (Hard Argmax)")
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=5)
        tgt_labels = kmeans.fit_predict(X_tgt_pool)
        tgt_centroids = kmeans.cluster_centers_
        
        cers, sims = [], []
        for X_src, ref_text, _ in source_data:
            # Match each frame directly to closest cluster centroid (vector quantization limit)
            dists = pairwise_distances(X_src, tgt_centroids, metric='cosine')
            best_idx = np.argmin(dists, axis=1)
            X_colored = tgt_centroids[best_idx]
            
            wav_conv = vocode(X_colored, hifigan, device)
            conv_text = transcribe(wav_conv.squeeze().numpy())
            cers.append(jiwer.cer(ref_text, conv_text))
            
            conv_emb = get_embedding(wav_conv.squeeze().numpy())
            conv_emb = torch.nn.functional.normalize(conv_emb, dim=0)
            sims.append(torch.dot(tgt_profile, conv_emb).item())
            
        results.append({
            "Configuration": f"K = {K} (Discrete Centroids)",
            "Continuum Regime": f"Discrete Vector Quantization (K={K})",
            "CER (%)": np.mean(cers) * 100,
            "Cosine Sim": np.mean(sims)
        })
        print(f"  -> CER: {np.mean(cers)*100:.2f}% | Sim: {np.mean(sims):.4f}")

    # ----------------------------------------------------
    # Terminal Point on Continuum: kNN-VC (1-NN and 4-NN Instance-based substitution)
    # ----------------------------------------------------
    print("\n[Point 7] K = Nt, beta -> inf : 1-NN (Pure Instance-based Substitution)")
    Y_tensor = torch.tensor(X_tgt_pool, dtype=torch.float32, device=device)
    Y_norm = torch.nn.functional.normalize(Y_tensor, dim=1)
    
    cers_1nn, sims_1nn = [], []
    cers_4nn, sims_4nn = [], []
    
    for X_src, ref_text, _ in source_data:
        X_tensor = torch.tensor(X_src, dtype=torch.float32, device=device)
        X_norm = torch.nn.functional.normalize(X_tensor, dim=1)
        
        dists = 1 - torch.matmul(X_norm, Y_norm.T) # [Ns, Nt]
        
        # 1-NN (Exact K=Nt, beta->inf limit)
        idx_1nn = dists.argmin(dim=1)
        X_1nn = Y_tensor[idx_1nn].unsqueeze(0)
        with torch.inference_mode():
            wav_1nn = hifigan(X_1nn).squeeze().cpu()
        conv_text = transcribe(wav_1nn.numpy())
        cers_1nn.append(jiwer.cer(ref_text, conv_text))
        emb_1nn = torch.nn.functional.normalize(get_embedding(wav_1nn.numpy()), dim=0)
        sims_1nn.append(torch.dot(tgt_profile, emb_1nn).item())
        
        # 4-NN (Standard kNN-VC)
        _, topk_idx = torch.topk(dists, k=4, dim=1, largest=False)
        X_4nn = Y_tensor[topk_idx].mean(dim=1).unsqueeze(0)
        with torch.inference_mode():
            wav_4nn = hifigan(X_4nn).squeeze().cpu()
        conv_text = transcribe(wav_4nn.numpy())
        cers_4nn.append(jiwer.cer(ref_text, conv_text))
        emb_4nn = torch.nn.functional.normalize(get_embedding(wav_4nn.numpy()), dim=0)
        sims_4nn.append(torch.dot(tgt_profile, emb_4nn).item())
        
    results.append({
        "Configuration": "1-NN (K = Nt, beta -> inf)",
        "Continuum Regime": "Extreme Local Instance Substitution",
        "CER (%)": np.mean(cers_1nn) * 100,
        "Cosine Sim": np.mean(sims_1nn)
    })
    results.append({
        "Configuration": "4-NN (kNN-VC standard)",
        "Continuum Regime": "Instance-based kNN Baseline",
        "CER (%)": np.mean(cers_4nn) * 100,
        "Cosine Sim": np.mean(sims_4nn)
    })
    print(f"  -> 1-NN : CER = {np.mean(cers_1nn)*100:.2f}% | Sim = {np.mean(sims_1nn):.4f}")
    print(f"  -> 4-NN : CER = {np.mean(cers_4nn)*100:.2f}% | Sim = {np.mean(sims_4nn):.4f}")

    print("\n======================= THE FULL ALIGNMENT CONTINUUM =======================")
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    df_res.to_csv("full_continuum_table.csv", index=False)
    print("\nSaved full continuum to full_continuum_table.csv")

if __name__ == "__main__":
    main()
