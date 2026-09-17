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
from sklearn.neighbors import NearestNeighbors

logging.getLogger("speechbrain").setLevel(logging.ERROR)
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.inference.speaker import EncoderClassifier

sys.path.append(str(Path(__file__).parent.parent))
from une import LibriSpeech

device = "cpu"

def extract_features(wav_path, wavlm_large, asr_model, processor):
    wav, sr = torchaudio.load(str(wav_path))
    wav = wav.to(device)
    if sr != 16000:
        wav = F.resample(wav, sr, 16000)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    with torch.no_grad():
        features_large, _ = wavlm_large.extract_features(wav, output_layer=6)
        x = features_large.squeeze(0).cpu().numpy()
        
        inputs = processor(wav.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt").to(device)
        y = asr_model(inputs.input_values).logits.squeeze(0).cpu().numpy()
        
    min_T = min(x.shape[0], y.shape[0])
    return x[:min_T], y[:min_T], wav.squeeze()

def learn_cca(X_list, Y_list, device):
    X_frames = np.concatenate(X_list, axis=0)
    Y_frames = np.concatenate(Y_list, axis=0)
    X_mean = np.mean(X_frames, axis=0)
    Y_mean = np.mean(Y_frames, axis=0)
    Xt = torch.tensor(X_frames - X_mean, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y_frames - Y_mean, dtype=torch.float32, device=device)
    Q_x, R_x = torch.linalg.qr(Xt)
    Q_y, R_y = torch.linalg.qr(Yt)
    C = torch.matmul(Q_x.T, Q_y)
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)
    Wx = torch.linalg.solve(R_x, U)
    return X_mean, Wx.cpu().numpy()
    return X_mean, Wx.cpu().numpy()

def project_to_cca(X, X_mean, Wx):
    return np.matmul(X - X_mean, Wx)

def vocode(features, hifigan, device):
    with torch.inference_mode():
        feats_tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        return hifigan(feats_tensor).squeeze(0).cpu()

def main():
    warnings.filterwarnings("ignore")
    print("1. Loading Models (WavLM-Large, PPGs, HiFi-GAN, ECAPA)...")
    
    wavlm_large = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device).eval()
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, prematched=True, device=device)
    hifigan.eval()
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()
    speaker_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device}, savedir="/tmp/speechbrain")

    def transcribe(wav):
        if torch.is_tensor(wav):
            wav = wav.cpu().numpy()
        inputs = processor(wav, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = asr_model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(predicted_ids)[0]
    
    def get_embedding(wav):
        with torch.no_grad():
            wav_t = torch.tensor(wav).float().unsqueeze(0)
            return speaker_model.encode_batch(wav_t).squeeze()

    print("\n2. Learning CCA Projection Matrix (WavLM -> PPGs)...")
    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    librispeech = LibriSpeech(root=dataset_path, ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    
    df_cca = df_all.sample(n=250, random_state=42)
    X_cca_list, Y_cca_list = [], []
    for p in tqdm(df_cca['path'], desc="CCA Data"):
        x, y, _ = extract_features(p, wavlm_large, asr_model, processor)
        X_cca_list.append(x)
        Y_cca_list.append(y)
        
    X_mean, Wx = learn_cca(X_cca_list, Y_cca_list, device)
    print(f"   CCA Projection Matrix Wx Shape: {Wx.shape}")

    print("\n3. Building Target Pool in CCA Space...")
    tgt_spk = "121"
    tgt_paths = df_all[df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")]['path'].tolist()[:20]
    
    X_tgt_list, tgt_wavs = [], []
    for p in tqdm(tgt_paths, desc="Target Data"):
        x, _, w = extract_features(p, wavlm_large, asr_model, processor)
        X_tgt_list.append(x)
        tgt_wavs.append(w)
        
    X_tgt_pool = np.concatenate(X_tgt_list, axis=0)
    C_tgt_pool = project_to_cca(X_tgt_pool, X_mean, Wx) # (N_frames, 32)
    
    tgt_embs = [get_embedding(w) for w in tgt_wavs[:10]]
    tgt_profile = torch.stack(tgt_embs).mean(dim=0)
    tgt_profile = torch.nn.functional.normalize(tgt_profile, dim=0)
    
    # Initialize kNN purely on the CCA Space
    knn = NearestNeighbors(n_neighbors=1, metric="cosine", n_jobs=-1)
    knn.fit(C_tgt_pool)

    print("\n4. Running CCA-Guided LinearVC Pipeline...")
    df_src = df_all[~df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")].sample(n=25, random_state=42)
    src_paths = df_src['path'].tolist()
    
    cers = []
    sims = []
    
    for p in tqdm(src_paths, desc="Evaluating CCA-Guided VC"):
        X_src, _, w_src = extract_features(p, wavlm_large, asr_model, processor)
        ref_text = transcribe(w_src)
        
        # A. Project Source to CCA Space
        C_src = project_to_cca(X_src, X_mean, Wx)
        
        # B. Find nearest neighbors in CCA Space! (Pure Content Matching)
        _, indices = knn.kneighbors(C_src)
        
        # C. Retrieve the original 1024D WavLM Target frames
        X_tgt_matched = X_tgt_pool[indices.squeeze()]
        
        # D. Learn Regression Matrix W (1024D -> 1024D)
        W, _, _, _ = np.linalg.lstsq(X_src, X_tgt_matched, rcond=None)
        
        # E. Convert
        X_conv = np.matmul(X_src, W)
        
        wav_conv = vocode(X_conv, hifigan, device)
        
        conv_text = transcribe(wav_conv.squeeze().numpy())
        cer = jiwer.cer(ref_text, conv_text)
        cers.append(cer)
        
        conv_emb = get_embedding(wav_conv.squeeze().numpy())
        conv_emb = torch.nn.functional.normalize(conv_emb, dim=0)
        sim = torch.dot(tgt_profile, conv_emb).item()
        sims.append(sim)
            
    avg_cer = np.mean(cers) * 100
    avg_sim = np.mean(sims)
    
    print("\n================ CCA-GUIDED KNN RESULTS ================")
    print(f" CER (%):             {avg_cer:.2f}%")
    print(f" Similarity (Cosine): {avg_sim:.4f}")
    print("========================================================")
    
if __name__ == "__main__":
    main()
