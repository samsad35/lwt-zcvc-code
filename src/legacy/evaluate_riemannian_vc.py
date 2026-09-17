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

def matrix_log(C, reg=1e-5):
    evals, evecs = np.linalg.eigh(C)
    evals = np.maximum(evals, reg)
    return evecs @ np.diag(np.log(evals)) @ evecs.T

def matrix_exp(L):
    evals, evecs = np.linalg.eigh(L)
    return evecs @ np.diag(np.exp(evals)) @ evecs.T

def get_coral_wct_matrices(X_s, C_tgt_mean, k=128, coral_reg=0.1):
    mu_s = np.mean(X_s, axis=0)
    C_s = np.cov(X_s - mu_s, rowvar=False)
    
    # EVD for Source
    evals_s, evecs_s = np.linalg.eigh(C_s)
    k_s = min(k, len(evals_s))
    idx_s = np.argsort(evals_s)[::-1][:k_s]
    evals_s, evecs_s = evals_s[idx_s], evecs_s[:, idx_s]
    
    # CORAL Regularization on Source
    evals_s = evals_s + coral_reg
    
    # EVD for Target
    evals_t, evecs_t = np.linalg.eigh(C_tgt_mean)
    k_t = min(k, len(evals_t))
    idx_t = np.argsort(evals_t)[::-1][:k_t]
    evals_t, evecs_t = evals_t[idx_t], evecs_t[:, idx_t]
    
    # CORAL Regularization on Target
    evals_t = evals_t + coral_reg
    
    # Construct W and C matrices
    W = evecs_s @ np.diag(1.0 / np.sqrt(evals_s)) @ evecs_s.T
    C = evecs_t @ np.diag(np.sqrt(evals_t)) @ evecs_t.T
    return mu_s, W, C

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

    print("\n2. Loading Dataset & Computing Log-Euclidean Fréchet Mean...")
    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    librispeech = LibriSpeech(root=dataset_path, ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    
    tgt_spk = "121"
    tgt_paths = df_all[df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")]['path'].tolist()[:20]
    
    X_tgt_list, tgt_wavs = [], []
    for p in tqdm(tgt_paths, desc="Extracting Target"):
        x, w = extract_features(p, wavlm_large)
        X_tgt_list.append(x)
        tgt_wavs.append(w)
        
    # Compute Fréchet Mean of 1024D Covariances
    mu_tgt_global = np.concatenate(X_tgt_list, axis=0).mean(axis=0)
    
    log_covs = []
    for x in X_tgt_list:
        C_i = np.cov(x, rowvar=False)
        log_covs.append(matrix_log(C_i, reg=1e-2)) # Strong enough regularization for SPD
        
    L_mean = np.mean(log_covs, axis=0)
    C_tgt_frechet = matrix_exp(L_mean)
    
    tgt_embs = [get_embedding(w.numpy()) for w in tgt_wavs[:10]]
    tgt_profile = torch.stack(tgt_embs).mean(dim=0)
    tgt_profile = torch.nn.functional.normalize(tgt_profile, dim=0)

    N_COMPONENTS = 128
    CORAL_LAMBDA = 0.5
    print(f"\n3. Running Riemannian CORAL-WCT (Dim={N_COMPONENTS}, lambda={CORAL_LAMBDA})...")
    df_src = df_all[~df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")].sample(n=25, random_state=42)
    src_paths = df_src['path'].tolist()
    
    cers = []
    sims = []
    
    for p in tqdm(src_paths, desc="Evaluating CORAL-WCT"):
        X_src, w_src = extract_features(p, wavlm_large)
        ref_text = transcribe(w_src)
        
        # Calculate CORAL WCT Matrices
        mu_s, W_coral, C_coral = get_coral_wct_matrices(X_src, C_tgt_frechet, k=N_COMPONENTS, coral_reg=CORAL_LAMBDA)
        
        # Convert
        X_whitened = np.matmul(X_src - mu_s, W_coral)
        X_colored = np.matmul(X_whitened, C_coral) + mu_tgt_global
        
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
    
    print(f"\n================ RIEMANNIAN CORAL-WCT RESULTS ================")
    print(f" CER (%):             {avg_cer:.2f}%")
    print(f" Similarity (Cosine): {avg_sim:.4f}")
    print(f" EER (%):             {eer*100:.2f}% (Target=50%)")
    print("==============================================================")
    
if __name__ == "__main__":
    main()
