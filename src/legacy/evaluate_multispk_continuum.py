import os
import sys
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import jiwer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.inference.speaker import EncoderClassifier

import torchaudio.functional as F

# Ensure HuggingFace Hub token is set to prevent rate limits
device = "cuda:0" if torch.cuda.is_available() else "cpu"

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
    print(f"=== Multi-Speaker Alignment Continuum Evaluation ===")
    print(f"Using device: {device}")
    
    # 1. Models
    print("Loading models (WavLM, HiFi-GAN, Wav2Vec2, ECAPA)...")
    wavlm_large = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device).eval()
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, prematched=True, device=device)
    hifigan.eval()
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()
    speaker_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        run_opts={"device": device}, 
        savedir="/tmp/speechbrain"
    )

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

    # 2. Dataset Setup
    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    
    # 5 Target speakers (2 Female, 3 Male)
    target_speakers = ["121", "237", "260", "1089", "1188"]
    T_SIZE = 15 # 15 utterances per target pool (~4,000-5,000 target frames)
    N_COMPONENTS = 128
    
    # Select fixed pool of 10 source utterances outside target speakers
    all_spks = [d.name for d in dataset_path.iterdir() if d.is_dir()]
    src_spks = [s for s in all_spks if s not in target_speakers]
    
    src_paths = []
    for s in src_spks[:10]:
        f = list((dataset_path / s).rglob("*.flac"))
        if len(f) > 0:
            src_paths.append(f[0])
    src_paths = src_paths[:10]
    print(f"Pre-extracting {len(src_paths)} source test utterances...")
    source_data = []
    for p in src_paths:
        X_src, w_src = extract_features(p, wavlm_large)
        ref_text = transcribe(w_src.numpy())
        source_data.append((X_src, ref_text))

    methods = [
        ("Global WCT (K=1)", "wct_k1"),
        ("Soft WCT (K=3)", "wct_k3"),
        ("Soft WCT* (K=5)", "wct_k5"),
        ("Soft WCT (K=8)", "wct_k8"),
        ("1-NN (K=Nt)", "knn_1"),
        ("4-NN (kNN-VC)", "knn_4")
    ]

    all_records = []

    for tgt_spk in target_speakers:
        print(f"\n========================================================")
        print(f"--> Evaluating Target Speaker {tgt_spk}...")
        print(f"========================================================")
        tgt_files = sorted(list((dataset_path / tgt_spk).rglob("*.flac")))[:T_SIZE]
        
        X_tgt_list, tgt_wavs = [], []
        for p in tgt_files:
            x, w = extract_features(p, wavlm_large)
            X_tgt_list.append(x)
            tgt_wavs.append(w)
        X_tgt_pool = np.concatenate(X_tgt_list, axis=0) # [Nt, 1024]
        
        # Target speaker reference profile
        tgt_embs = [get_embedding(w.numpy()) for w in tgt_wavs[:8]]
        tgt_profile = torch.stack(tgt_embs).mean(dim=0)
        tgt_profile = torch.nn.functional.normalize(tgt_profile, dim=0)

        # -----------------------------------------------------------------
        # 1. Global Classic WCT (K=1)
        # -----------------------------------------------------------------
        mu_tgt, _, C_tgt = get_wct_matrices(X_tgt_pool, k=N_COMPONENTS)
        cers_k1, sims_k1 = [], []
        for X_src, ref_text in source_data:
            k_src = min(N_COMPONENTS, X_src.shape[0] - 1)
            mu_src, W_src, _ = get_wct_matrices(X_src, k=k_src)
            X_w = np.matmul(X_src - mu_src, W_src)
            X_c = np.matmul(X_w, C_tgt.T) + mu_tgt
            
            wav_conv = vocode(X_c, hifigan, device)
            cers_k1.append(jiwer.cer(ref_text, transcribe(wav_conv.squeeze().numpy())))
            conv_emb = torch.nn.functional.normalize(get_embedding(wav_conv.squeeze().numpy()), dim=0)
            sims_k1.append(torch.dot(tgt_profile, conv_emb).item())

        all_records.append({
            "Speaker": tgt_spk, "Method": "Global WCT (K=1)", "K": 1,
            "CER (%)": np.mean(cers_k1) * 100, "Cosine Sim": np.mean(sims_k1)
        })
        print(f"  [K=1] CER: {np.mean(cers_k1)*100:.2f}% | Sim: {np.mean(sims_k1):.4f}")

        # -----------------------------------------------------------------
        # 2. Soft Local WCT (K = 3, 5, 8)
        # -----------------------------------------------------------------
        for K in [3, 5, 8]:
            kmeans = KMeans(n_clusters=K, random_state=42, n_init=5)
            tgt_labels = kmeans.fit_predict(X_tgt_pool)
            tgt_centroids = kmeans.cluster_centers_
            
            cluster_matrices = []
            for k_idx in range(K):
                X_k = X_tgt_pool[tgt_labels == k_idx]
                if len(X_k) > N_COMPONENTS:
                    mu_k, _, C_k = get_wct_matrices(X_k, k=N_COMPONENTS)
                else:
                    mu_k = np.mean(X_k, axis=0) if len(X_k) > 0 else tgt_centroids[k_idx]
                    C_k = np.eye(1024)
                cluster_matrices.append((mu_k, C_k))
            
            cers_k, sims_k = [], []
            for X_src, ref_text in source_data:
                k_src = min(N_COMPONENTS, X_src.shape[0] - 1)
                mu_src, W_src, _ = get_wct_matrices(X_src, k=k_src)
                X_w = np.matmul(X_src - mu_src, W_src)
                
                dists = pairwise_distances(X_src, tgt_centroids, metric='euclidean')
                logits = -2.0 * dists
                logits -= np.max(logits, axis=1, keepdims=True)
                weights = np.exp(logits)
                weights = weights / weights.sum(axis=1, keepdims=True)
                
                X_c = np.zeros_like(X_src)
                for k_idx in range(K):
                    mu_k, C_k = cluster_matrices[k_idx]
                    X_ck = np.matmul(X_w, C_k.T) + mu_k
                    X_c += weights[:, k_idx:k_idx+1] * X_ck
                    
                wav_conv = vocode(X_c, hifigan, device)
                cers_k.append(jiwer.cer(ref_text, transcribe(wav_conv.squeeze().numpy())))
                conv_emb = torch.nn.functional.normalize(get_embedding(wav_conv.squeeze().numpy()), dim=0)
                sims_k.append(torch.dot(tgt_profile, conv_emb).item())

            all_records.append({
                "Speaker": tgt_spk, "Method": f"Soft WCT (K={K})", "K": K,
                "CER (%)": np.mean(cers_k) * 100, "Cosine Sim": np.mean(sims_k)
            })
            print(f"  [K={K}] CER: {np.mean(cers_k)*100:.2f}% | Sim: {np.mean(sims_k):.4f}")

        # -----------------------------------------------------------------
        # 3. 1-NN and 4-NN (Instance-based kNN-VC)
        # -----------------------------------------------------------------
        X_tgt_torch = torch.tensor(X_tgt_pool).float().to(device)
        
        for k_knn in [1, 4]:
            cers_knn, sims_knn = [], []
            for X_src, ref_text in source_data:
                X_src_t = torch.tensor(X_src).float().to(device)
                
                # Cosine distance
                src_norm = torch.nn.functional.normalize(X_src_t, dim=1)
                tgt_norm = torch.nn.functional.normalize(X_tgt_torch, dim=1)
                sim_matrix = torch.matmul(src_norm, tgt_norm.T) # [Ns, Nt]
                
                topk_sims, topk_indices = torch.topk(sim_matrix, k=k_knn, dim=1)
                if k_knn == 1:
                    X_out = X_tgt_torch[topk_indices.squeeze(1)]
                else:
                    X_out = torch.stack([X_tgt_torch[idx].mean(dim=0) for idx in topk_indices])
                    
                wav_conv = vocode(X_out.cpu().numpy(), hifigan, device)
                cers_knn.append(jiwer.cer(ref_text, transcribe(wav_conv.squeeze().numpy())))
                conv_emb = torch.nn.functional.normalize(get_embedding(wav_conv.squeeze().numpy()), dim=0)
                sims_knn.append(torch.dot(tgt_profile, conv_emb).item())

            method_name = "1-NN (K=Nt)" if k_knn == 1 else "4-NN (kNN-VC)"
            all_records.append({
                "Speaker": tgt_spk, "Method": method_name, "K": 9999 if k_knn==1 else 99999,
                "CER (%)": np.mean(cers_knn) * 100, "Cosine Sim": np.mean(sims_knn)
            })
            print(f"  [{method_name}] CER: {np.mean(cers_knn)*100:.2f}% | Sim: {np.mean(sims_knn):.4f}")

    # Save detailed CSV
    df_results = pd.DataFrame(all_records)
    csv_path = Path("/local_scratch/ssadok/un_projet_audio/multispk_continuum_table.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved detailed records to {csv_path}")

    # Compute Macro-Averaged Summary
    summary = df_results.groupby("Method", sort=False).agg({
        "CER (%)": ["mean", "std"],
        "Cosine Sim": ["mean", "std"]
    })
    print("\n========================================================")
    print("=== SUMMARY ACROSS ALL 5 SPEAKERS (Mean ± Std) ===")
    print("========================================================")
    print(summary)

if __name__ == "__main__":
    main()
