import torch
import torch.nn as nn
import torch.optim as optim
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
from une import LibriSpeech, WavLM

device = "cpu"

class TargetDecoderMLP(nn.Module):
    def __init__(self, input_dim=128, output_dim=1024):
        super().__init__()
        # A small non-linear network to learn complex vocal tract transformations
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

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
    return h_x[:min_T], h_y[:min_T], wav.squeeze()

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
    Wy = torch.linalg.solve(R_y, Vh.T)
    return Y_mean, Wy.cpu().numpy()

def get_universal_content(Y_features, Y_mean, Wy_sub):
    Y_c = Y_features - Y_mean
    return np.matmul(Y_c, Wy_sub)

def vocode(features, hifigan, device):
    with torch.inference_mode():
        feats_tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        return hifigan(feats_tensor).squeeze(0).cpu()

def main():
    warnings.filterwarnings("ignore")
    print("1. Loading Models (WavLM, HiFi-GAN, ASR, Speaker Verification)...")
    
    wavlm_large = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device).eval()
    wavlm_asr = WavLM(model_name="patrickvonplaten/wavlm-libri-clean-100h-base-plus", device=device)
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
            return speaker_model.encode_batch(wav.squeeze().unsqueeze(0)).squeeze()

    print("\n2. Loading Dataset & Computing CCA Space...")
    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    librispeech = LibriSpeech(root=dataset_path, ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    
    df_cca = df_all.sample(n=250, random_state=42)
    X_list, Y_list = [], []
    for p in tqdm(df_cca['path'], desc="CCA Data"):
        x, y, _ = extract_features(p, wavlm_large, wavlm_asr)
        X_list.append(x)
        Y_list.append(y)
    Y_mean, Wy_full = learn_cca_full(X_list, Y_list, device)
    
    tgt_spk = "121"
    # Target Data (same 25 files to be fair)
    tgt_paths = df_all[df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")]['path'].tolist()[:25]
    
    X_tgt_list, Y_tgt_list, tgt_wavs = [], [], []
    for p in tgt_paths:
        x, y, w = extract_features(p, wavlm_large, wavlm_asr)
        X_tgt_list.append(x)
        Y_tgt_list.append(y)
        tgt_wavs.append(w)
        
    X_tgt_all = np.concatenate(X_tgt_list, axis=0)
    Y_tgt_all = np.concatenate(Y_tgt_list, axis=0)
    
    tgt_embs = [get_embedding(w) for w in tgt_wavs[:10]]
    tgt_profile = torch.stack(tgt_embs).mean(dim=0)
    tgt_profile = torch.nn.functional.normalize(tgt_profile, dim=0)

    # 25 Source utterances
    df_src = df_all[~df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")].sample(n=25, random_state=42)
    src_paths = df_src['path'].tolist()
    print(f"   Collected {len(src_paths)} source utterances.")
    
    source_data = []
    for p in src_paths:
        x, y, w = extract_features(p, wavlm_large, wavlm_asr)
        ref_text = transcribe(w)
        source_data.append({"x": x, "y": y, "wav": w, "text": ref_text})
        
    print("\n3. Learning NON-LINEAR Decoder (MLP) for Target...")
    # Fix dimension to the "Sweet Spot" 128
    dim = 128
    Wy_sub = Wy_full[:, :dim]
    
    Z_c_tgt = get_universal_content(Y_tgt_all, Y_mean, Wy_sub)
    
    Z_tensor = torch.tensor(Z_c_tgt, dtype=torch.float32, device=device)
    X_tensor = torch.tensor(X_tgt_all, dtype=torch.float32, device=device)
    
    mlp = TargetDecoderMLP(input_dim=128, output_dim=1024).to(device)
    optimizer = optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    mlp.train()
    epochs = 1500
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = mlp(Z_tensor)
        loss = criterion(preds, X_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 300 == 0:
            print(f"   [Epoch {epoch+1}/{epochs}] Loss: {loss.item():.4f}")
    mlp.eval()
    
    print("\n4. Running Conversion Pipeline...")
    cers = []
    sims = []
    
    for src in tqdm(source_data, desc="Evaluating CCA-MLP"):
        Z_c_src = get_universal_content(src["y"], Y_mean, Wy_sub)
        Z_src_tensor = torch.tensor(Z_c_src, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            X_converted = mlp(Z_src_tensor).cpu().numpy()
            
        wav_conv = vocode(X_converted, hifigan, device)
        
        conv_text = transcribe(wav_conv)
        cer = jiwer.cer(src["text"], conv_text)
        cers.append(cer)
        
        conv_emb = get_embedding(wav_conv)
        conv_emb = torch.nn.functional.normalize(conv_emb, dim=0)
        sim = torch.dot(tgt_profile, conv_emb).item()
        sims.append(sim)
            
    avg_cer = np.mean(cers) * 100
    avg_sim = np.mean(sims)
    
    print("\n================ CCA-MLP RESULTS (128D) ================")
    print(f" CER (%):             {avg_cer:.2f}%")
    print(f" Similarity (Cosine): {avg_sim:.4f}")
    print("==========================================================")
    
if __name__ == "__main__":
    main()
