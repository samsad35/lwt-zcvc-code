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

logging.getLogger("speechbrain").setLevel(logging.ERROR)
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.inference.speaker import EncoderClassifier

sys.path.append(str(Path(__file__).parent.parent))
from une import LibriSpeech

device = "cpu"

def extract_features(wav, wavlm_large):
    wav = wav.to(device)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    with torch.no_grad():
        features_large, _ = wavlm_large.extract_features(wav, output_layer=6)
        x = features_large.squeeze(0).cpu().numpy()
    return x

def load_audio(wav_path):
    wav, sr = torchaudio.load(str(wav_path))
    if sr != 16000:
        wav = F.resample(wav, sr, 16000)
    return wav.squeeze()

def vocode(features, hifigan, device):
    with torch.inference_mode():
        feats_tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        return hifigan(feats_tensor).squeeze(0).cpu()

def get_wct_matrices(X, k=1024):
    mu = np.mean(X, axis=0)
    X_c = X - mu
    cov = np.cov(X_c, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    k = min(k, len(eigenvalues))
    eigenvalues = eigenvalues[:k]
    eigenvectors = eigenvectors[:, :k]
    
    eps = 1e-5
    eigenvalues = np.maximum(eigenvalues, eps)
    
    D_inv_half = np.diag(1.0 / np.sqrt(eigenvalues))
    D_half = np.diag(np.sqrt(eigenvalues))
    
    W = eigenvectors @ D_inv_half @ eigenvectors.T
    C = eigenvectors @ D_half @ eigenvectors.T
    return mu, W, C

def augment_and_extract(wav, wavlm_large):
    # Base
    feats = [extract_features(wav, wavlm_large)]
    
    # Perturbations by resampling (simulating speed/pitch change)
    rates = [14000, 15000, 17000, 18000]
    for r in rates:
        wav_aug = F.resample(wav, 16000, r)
        feats.append(extract_features(wav_aug, wavlm_large))
        
    # Also add some noise augmentation
    noise1 = wav + torch.randn_like(wav) * 0.005
    noise2 = wav + torch.randn_like(wav) * 0.01
    feats.append(extract_features(noise1, wavlm_large))
    feats.append(extract_features(noise2, wavlm_large))
        
    return np.concatenate(feats, axis=0)

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

    print("\n2. Loading Dataset & Computing Target WCT Matrices...")
    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    librispeech = LibriSpeech(root=dataset_path, ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    
    tgt_spk = "121"
    tgt_paths = df_all[df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")]['path'].tolist()[:20]
    
    X_tgt_list, tgt_wavs = [], []
    for p in tqdm(tgt_paths, desc="Target Data"):
        w = load_audio(p)
        x = extract_features(w, wavlm_large)
        X_tgt_list.append(x)
        tgt_wavs.append(w)
        
    X_tgt_pool = np.concatenate(X_tgt_list, axis=0)
    
    N_COMPONENTS = 1024
    mu_tgt, _, C_tgt = get_wct_matrices(X_tgt_pool, k=N_COMPONENTS)
    
    tgt_embs = [get_embedding(w.numpy()) for w in tgt_wavs[:10]]
    tgt_profile = torch.stack(tgt_embs).mean(dim=0)
    tgt_profile = torch.nn.functional.normalize(tgt_profile, dim=0)

    print("\n3. Running Augmented WCT-VC on single source utterances...")
    df_src = df_all[~df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")].sample(n=25, random_state=42)
    src_paths = df_src['path'].tolist()
    
    cers = []
    sims = []
    
    for p in tqdm(src_paths, desc="Evaluating Augmented WCT-VC"):
        w_src = load_audio(p)
        ref_text = transcribe(w_src)
        
        # Original features for conversion
        X_src = extract_features(w_src, wavlm_large)
        
        # Augmented features to build the 1024D covariance matrix!
        X_src_aug = augment_and_extract(w_src, wavlm_large)
        
        k_src = min(N_COMPONENTS, X_src_aug.shape[0] - 1)
        mu_src, W_src, _ = get_wct_matrices(X_src_aug, k=k_src)
        
        # Whiten Original Source
        X_whitened = np.matmul(X_src - mu_src, W_src)
        
        # Color with Target
        X_colored = np.matmul(X_whitened, C_tgt) + mu_tgt
        
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
    
    print(f"\n================ AUGMENTED WCT-VC RESULTS (Dim={N_COMPONENTS}) ================")
    print(f" CER (%):             {avg_cer:.2f}%")
    print(f" Similarity (Cosine): {avg_sim:.4f}")
    print("=========================================================================")
    
if __name__ == "__main__":
    main()
