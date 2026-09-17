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
    evals, evecs = np.linalg.eigh(cov)
    
    k = min(k, len(evals))
    idx = np.argsort(evals)[::-1][:k]
    evals, evecs = evals[idx], evecs[:, idx]
    evals = np.maximum(evals, 1e-5)
    
    # Whitening and Coloring matrices
    W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    C = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
    return mu, W, C

def fast_cosine_dist(source_feats, target_feats):
    import torch.nn.functional as F_nn
    s = F_nn.normalize(torch.tensor(source_feats), dim=1)
    t = F_nn.normalize(torch.tensor(target_feats), dim=1)
    dist = 1 - torch.matmul(s, t.T)
    return dist.numpy()

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
        inputs = processor(wav.numpy(), sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = asr_model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(predicted_ids)[0]
    
    def get_embedding(wav):
        with torch.no_grad():
            wav_t = torch.tensor(wav).float().unsqueeze(0)
            return speaker_model.encode_batch(wav_t).squeeze()

    print("\n2. Loading Dataset & Target WCT...")
    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    librispeech = LibriSpeech(root=dataset_path, ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    
    tgt_spk = "121"
    tgt_paths = df_all[df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")]['path'].tolist()[:20]
    
    X_tgt_list, tgt_wavs = [], []
    for p in tqdm(tgt_paths, desc="Target Data"):
        x, w = extract_features(p, wavlm_large)
        X_tgt_list.append(x)
        tgt_wavs.append(w)
        
    X_tgt_pool = np.concatenate(X_tgt_list, axis=0)
    
    N_COMPONENTS = 128
    mu_tgt, W_tgt, C_tgt = get_wct_matrices(X_tgt_pool, k=N_COMPONENTS)
    
    # We pre-whiten the Target pool for ICP alignment
    H_tgt_pool = np.matmul(X_tgt_pool - mu_tgt, W_tgt)
    
    tgt_embs = [get_embedding(w.numpy()) for w in tgt_wavs[:10]]
    tgt_profile = torch.stack(tgt_embs).mean(dim=0)
    tgt_profile = torch.nn.functional.normalize(tgt_profile, dim=0)

    print(f"\n3. Running ICP-WCT (Iterative Closest Point Rotation on Spheres)...")
    df_src = df_all[~df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")].sample(n=25, random_state=42)
    src_paths = df_src['path'].tolist()
    
    cers = []
    sims = []
    
    ICP_ITERATIONS = 3
    
    for p in tqdm(src_paths, desc="Evaluating ICP-WCT"):
        X_src, w_src = extract_features(p, wavlm_large)
        ref_text = transcribe(w_src)
        
        k_src = min(N_COMPONENTS, X_src.shape[0] - 1)
        mu_src, W_src, _ = get_wct_matrices(X_src, k=k_src)
        
        # 1. Whiten Source
        H_src = np.matmul(X_src - mu_src, W_src)
        
        # 2. Iterative Closest Point (ICP) to find optimal Orthogonal Rotation R
        R = np.eye(H_src.shape[1])
        for _ in range(ICP_ITERATIONS):
            H_src_rot = np.matmul(H_src, R)
            
            # Find nearest neighbors in Whitened Target Pool
            dists = fast_cosine_dist(H_src_rot, H_tgt_pool)
            best_idx = np.argmin(dists, axis=1)
            H_tgt_matched = H_tgt_pool[best_idx]
            
            # Compute Cross-Covariance Matrix and Orthogonal Procrustes
            S = np.matmul(H_src.T, H_tgt_matched)
            U, _, Vt = np.linalg.svd(S)
            R = np.matmul(U, Vt) # Optimal rotation
            
        H_src_final = np.matmul(H_src, R)
        
        # 3. Color with Target
        X_colored = np.matmul(H_src_final, C_tgt) + mu_tgt
        
        wav_conv = vocode(X_colored, hifigan, device)
        conv_text = transcribe(wav_conv.squeeze())
        cer = jiwer.cer(ref_text, conv_text)
        cers.append(cer)
        
        conv_emb = get_embedding(wav_conv.squeeze())
        conv_emb = torch.nn.functional.normalize(conv_emb, dim=0)
        sim = torch.dot(tgt_profile, conv_emb).item()
        sims.append(sim)
            
    avg_cer = np.mean(cers) * 100
    avg_sim = np.mean(sims)
    
    # EER
    genuine_sims = []
    for w in tgt_wavs[10:20]:
        emb = get_embedding(w.numpy())
        emb = torch.nn.functional.normalize(emb, dim=0)
        genuine_sims.append(torch.dot(tgt_profile, emb).item())
        
    y_true = [1] * len(genuine_sims) + [0] * len(sims)
    y_score = genuine_sims + sims
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    print(f"\n================ ICP-WCT RESULTS (Dim={N_COMPONENTS}, Iter={ICP_ITERATIONS}) ================")
    print(f" CER (%):             {avg_cer:.2f}%")
    print(f" Similarity (Cosine): {avg_sim:.4f}")
    print(f" EER (%):             {eer*100:.2f}% (Target=50%)")
    print("==================================================================================")
    
if __name__ == "__main__":
    main()
