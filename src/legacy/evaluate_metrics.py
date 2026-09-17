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

# Set logging level to ERROR to hide speechbrain warnings
logging.getLogger("speechbrain").setLevel(logging.ERROR)

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.inference.speaker import EncoderClassifier

sys.path.append(str(Path(__file__).parent.parent))
from une import LibriSpeech, WavLM

device = "cpu"

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
    
    # 1.1 SSL Models
    wavlm_large = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device).eval()
    wavlm_asr = WavLM(model_name="patrickvonplaten/wavlm-libri-clean-100h-base-plus", device=device)
    hifigan, _ = torch.hub.load("bshall/knn-vc", "hifigan_wavlm", trust_repo=True, prematched=True, device=device)
    hifigan.eval()
    
    # 1.2 Evaluation Models
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
            # SpeechBrain expects [batch, time], so we must ensure wav is 1D before unsqueeze
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
    
    print("\n3. Setting up Experiment (Target: 121, Sources: 10 random males)...")
    tgt_spk = "121" # Female
    tgt_paths = df_all[df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")]['path'].tolist()[:10]
    
    X_tgt_list, Y_tgt_list, tgt_wavs = [], [], []
    for p in tgt_paths:
        x, y, w = extract_features(p, wavlm_large, wavlm_asr)
        X_tgt_list.append(x)
        Y_tgt_list.append(y)
        tgt_wavs.append(w)
        
    X_tgt_all = np.concatenate(X_tgt_list, axis=0)
    Y_tgt_all = np.concatenate(Y_tgt_list, axis=0)
    
    # Compute Target Speaker Profile (Average Embedding)
    tgt_embs = [get_embedding(w) for w in tgt_wavs]
    tgt_profile = torch.stack(tgt_embs).mean(dim=0)
    tgt_profile = torch.nn.functional.normalize(tgt_profile, dim=0)

    # 25 Random Source utterances (excluding target) for robust statistical evaluation
    df_src = df_all[~df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")].sample(n=25, random_state=42)
    src_paths = df_src['path'].tolist()
    
    print(f"   Collected {len(src_paths)} source utterances.")
    
    source_data = []
    for p in src_paths:
        x, y, w = extract_features(p, wavlm_large, wavlm_asr)
        ref_text = transcribe(w) # Use original audio transcript as Ground Truth for CER
        source_data.append({"x": x, "y": y, "wav": w, "text": ref_text})
        
    dims_to_test = [16, 32, 64, 128, 256, 512, 768]
    results = []
    
    print("\n4. Running Metrics Pipeline...")
    for dim in dims_to_test:
        print(f"   -> Testing Dimension = {dim}")
        Wy_sub = Wy_full[:, :dim]
        
        # Learn Target Matrix
        Z_c_tgt = get_universal_content(Y_tgt_all, Y_mean, Wy_sub)
        Z_c_tgt_bias = np.hstack([Z_c_tgt, np.ones((Z_c_tgt.shape[0], 1))])
        W_tgt, _, _, _ = np.linalg.lstsq(Z_c_tgt_bias, X_tgt_all, rcond=None)
        
        cers = []
        sims = []
        for src in source_data:
            Z_c_src = get_universal_content(src["y"], Y_mean, Wy_sub)
            Z_c_src_bias = np.hstack([Z_c_src, np.ones((Z_c_src.shape[0], 1))])
            
            X_converted = np.matmul(Z_c_src_bias, W_tgt)
            wav_conv = vocode(X_converted, hifigan, device)
            
            # Metric 1: Intelligibility (CER)
            conv_text = transcribe(wav_conv)
            cer = jiwer.cer(src["text"], conv_text)
            cers.append(cer)
            
            # Metric 2: Speaker Similarity (Cosine)
            conv_emb = get_embedding(wav_conv)
            conv_emb = torch.nn.functional.normalize(conv_emb, dim=0)
            sim = torch.dot(tgt_profile, conv_emb).item()
            sims.append(sim)
            
        avg_cer = np.mean(cers) * 100
        avg_sim = np.mean(sims)
        print(f"      CER: {avg_cer:.2f}% | Similarity: {avg_sim:.4f}")
        results.append((dim, avg_cer, avg_sim))
        
    print("\n================ FINAL RESULTS ================")
    print(f"{'Dimension':<12} | {'CER (%)':<10} | {'Similarity (Cosine)':<20}")
    print("-" * 48)
    for res in results:
        print(f"{res[0]:<12} | {res[1]:<10.2f} | {res[2]:<20.4f}")
    print("===============================================")
    
if __name__ == "__main__":
    main()
