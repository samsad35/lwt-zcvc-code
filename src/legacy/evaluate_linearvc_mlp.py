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
from sklearn.neighbors import NearestNeighbors

logging.getLogger("speechbrain").setLevel(logging.ERROR)
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from speechbrain.inference.speaker import EncoderClassifier

sys.path.append(str(Path(__file__).parent.parent))
from une import LibriSpeech

device = "cpu"

class UtteranceMLP(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

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
            return speaker_model.encode_batch(wav.squeeze().unsqueeze(0)).squeeze()

    print("\n2. Loading Dataset & Preparing Target kNN Pool...")
    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    librispeech = LibriSpeech(root=dataset_path, ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    
    tgt_spk = "121"
    tgt_paths = df_all[df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")]['path'].tolist()[:20]
    
    X_tgt_list, tgt_wavs = [], []
    for p in tgt_paths:
        x, w = extract_features(p, wavlm_large)
        X_tgt_list.append(x)
        tgt_wavs.append(w)
        
    X_tgt_pool = np.concatenate(X_tgt_list, axis=0)
    print(f"   Target Pool Shape: {X_tgt_pool.shape}")
    
    tgt_embs = [get_embedding(w) for w in tgt_wavs[:10]]
    tgt_profile = torch.stack(tgt_embs).mean(dim=0)
    tgt_profile = torch.nn.functional.normalize(tgt_profile, dim=0)

    df_src = df_all[~df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")].sample(n=25, random_state=42)
    src_paths = df_src['path'].tolist()
    
    source_data = []
    for p in src_paths:
        x, w = extract_features(p, wavlm_large)
        ref_text = transcribe(w)
        source_data.append({"x": x, "wav": w, "text": ref_text})
        
    print("\n3. Running Neural-LinearVC Pipeline (kNN + MLP)...")
    cers = []
    sims = []
    
    # Pre-compute kNN
    knn = NearestNeighbors(n_neighbors=1, metric="cosine", n_jobs=-1)
    knn.fit(X_tgt_pool)
    
    for src in tqdm(source_data, desc="Evaluating Neural-LinearVC"):
        X_src = src["x"]
        
        # 1. Hard kNN Alignment (comme dans LinearVC)
        distances, indices = knn.kneighbors(X_src)
        X_tgt_matched = X_tgt_pool[indices.squeeze()]
        
        # 2. Train Utterance-Specific MLP (remplace la matrice linéaire OLS)
        mlp = UtteranceMLP().to(device)
        optimizer = optim.Adam(mlp.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        X_tensor = torch.tensor(X_src, dtype=torch.float32, device=device)
        Y_tensor = torch.tensor(X_tgt_matched, dtype=torch.float32, device=device)
        
        mlp.train()
        for epoch in range(250):  # On overfit volontairement la phrase
            optimizer.zero_grad()
            preds = mlp(X_tensor)
            loss = criterion(preds, Y_tensor)
            loss.backward()
            optimizer.step()
        mlp.eval()
        
        # 3. Convertir
        with torch.no_grad():
            X_converted = mlp(X_tensor).cpu().numpy()
        
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
    
    print("\n================ NEURAL-LINEARVC RESULTS ================")
    print(f" CER (%):             {avg_cer:.2f}%")
    print(f" Similarity (Cosine): {avg_sim:.4f}")
    print("=========================================================")
    
if __name__ == "__main__":
    main()
