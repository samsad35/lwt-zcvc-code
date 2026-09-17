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
import librosa
from sklearn.neighbors import NearestNeighbors

logging.getLogger("speechbrain").setLevel(logging.ERROR)
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.inference.speaker import EncoderClassifier

sys.path.append(str(Path(__file__).parent.parent))
from une import LibriSpeech

device = "cpu"

def extract_features_and_pitch(wav_path, wavlm_large):
    wav, sr = torchaudio.load(str(wav_path))
    if sr != 16000:
        wav = F.resample(wav, sr, 16000)
    wav_np = wav.squeeze().numpy()
    
    # Extract Pitch using Librosa PYIN (hop_length=320 for 50Hz, matching WavLM)
    f0, voiced_flag, voiced_probs = librosa.pyin(wav_np, fmin=50, fmax=500, sr=16000, frame_length=1024, hop_length=320)
    f0 = np.nan_to_num(f0, nan=0.0) # Replace NaNs with 0 (unvoiced)
    
    # Extract WavLM Features
    wav_t = wav.to(device)
    if wav_t.dim() == 1:
        wav_t = wav_t.unsqueeze(0)
    with torch.no_grad():
        features_large, _ = wavlm_large.extract_features(wav_t, output_layer=6)
        x = features_large.squeeze(0).cpu().numpy()
        
    # Align lengths (sometimes PYIN is 1 frame off due to padding)
    min_len = min(len(f0), x.shape[0])
    return x[:min_len], f0[:min_len], wav_np

def vocode(features, hifigan, device):
    with torch.inference_mode():
        feats_tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        return hifigan(feats_tensor).squeeze(0).cpu()

def compute_pitch_stats(f0_list):
    all_f0 = np.concatenate(f0_list)
    voiced_f0 = all_f0[all_f0 > 0]
    log_f0 = np.log(voiced_f0 + 1e-8)
    return np.mean(log_f0), np.std(log_f0)

def shift_pitch(f0, mu_src, std_src, mu_tgt, std_tgt):
    shifted_f0 = np.zeros_like(f0)
    voiced = f0 > 0
    if np.any(voiced):
        log_f0 = np.log(f0[voiced])
        log_f0_shifted = (log_f0 - mu_src) / std_src * std_tgt + mu_tgt
        shifted_f0[voiced] = np.exp(log_f0_shifted)
    return shifted_f0

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
        inputs = processor(wav, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = asr_model(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(predicted_ids)[0]
    
    def get_embedding(wav):
        with torch.no_grad():
            wav_t = torch.tensor(wav).float().unsqueeze(0)
            return speaker_model.encode_batch(wav_t).squeeze()

    print("\n2. Extracting Target Pool & Pitch (using librosa.pyin)...")
    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    librispeech = LibriSpeech(root=dataset_path, ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    
    tgt_spk = "121"
    tgt_paths = df_all[df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")]['path'].tolist()[:20]
    
    X_tgt_list, F0_tgt_list, tgt_wavs = [], [], []
    for p in tqdm(tgt_paths, desc="Target Data"):
        x, f0, w = extract_features_and_pitch(p, wavlm_large)
        X_tgt_list.append(x)
        F0_tgt_list.append(f0)
        tgt_wavs.append(w)
        
    X_tgt_pool = np.concatenate(X_tgt_list, axis=0)
    F0_tgt_pool = np.concatenate(F0_tgt_list, axis=0)
    
    mu_tgt, std_tgt = compute_pitch_stats(F0_tgt_list)
    
    tgt_embs = [get_embedding(w) for w in tgt_wavs[:10]]
    tgt_profile = torch.stack(tgt_embs).mean(dim=0)
    tgt_profile = torch.nn.functional.normalize(tgt_profile, dim=0)

    print("\n3. Running Orthogonal Pitch-Content-Speaker VC...")
    df_src = df_all[~df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")].sample(n=25, random_state=42)
    src_paths = df_src['path'].tolist()
    
    src_f0_all = []
    source_data = []
    for p in src_paths:
        x, f0, w = extract_features_and_pitch(p, wavlm_large)
        ref_text = transcribe(w)
        src_f0_all.append(f0)
        source_data.append({"x": x, "f0": f0, "wav": w, "text": ref_text})
        
    mu_src, std_src = compute_pitch_stats(src_f0_all)

    knn = NearestNeighbors(n_neighbors=1, metric="cosine", n_jobs=-1)
    knn.fit(X_tgt_pool)
    
    cers = []
    sims = []
    RANK_r = 32
    
    out_dir = Path("samples_orthogonal_vc")
    out_dir.mkdir(exist_ok=True)
    
    for idx, src in enumerate(tqdm(source_data, desc="Evaluating Orthogonal-VC")):
        X_1 = src["x"]
        P_1 = src["f0"].reshape(-1, 1)
        
        # kNN Alignment
        _, indices = knn.kneighbors(X_1)
        indices = indices.squeeze()
        X_2 = X_tgt_pool[indices]
        P_2 = F0_tgt_pool[indices].reshape(-1, 1)
        
        # Step A: Pitch Orthogonalization
        W_p1, _, _, _ = np.linalg.lstsq(P_1, X_1, rcond=None)
        W_p2, _, _, _ = np.linalg.lstsq(P_2, X_2, rcond=None)
        
        X_1_perp = X_1 - np.matmul(P_1, W_p1)
        X_2_perp = X_2 - np.matmul(P_2, W_p2)
        
        # Step B: Content / Speaker SVD Factorization
        X_block = np.concatenate([X_1_perp, X_2_perp], axis=1) # (N, 2048)
        U, S_vals, Vh = np.linalg.svd(X_block, full_matrices=False)
        
        # Truncate to rank r
        U_r = U[:, :RANK_r]
        S_r = np.diag(S_vals[:RANK_r])
        Vh_r = Vh[:RANK_r, :]
        
        C = np.matmul(U_r, S_r) # Content
        S_speaker = Vh_r        # Speakers [S1, S2]
        S_2 = S_speaker[:, 1024:] # Target speaker transformation (r x 1024)
        
        # Step C: Pitch Shifting
        P_shifted = shift_pitch(src["f0"], mu_src, std_src, mu_tgt, std_tgt).reshape(-1, 1)
        
        # Step D: Conversion (C * S2 + Pitch)
        X_conv = np.matmul(C, S_2) + np.matmul(P_shifted, W_p2)
        
        wav_conv = vocode(X_conv, hifigan, device)
        
        if idx < 5:
            torchaudio.save(out_dir / f"ortho_vc_sample_{idx}.wav", wav_conv.view(1, -1), 16000)
        
        conv_text = transcribe(wav_conv.squeeze().numpy())
        cer = jiwer.cer(src["text"], conv_text)
        cers.append(cer)
        
        conv_emb = get_embedding(wav_conv.squeeze().numpy())
        conv_emb = torch.nn.functional.normalize(conv_emb, dim=0)
        sim = torch.dot(tgt_profile, conv_emb).item()
        sims.append(sim)
            
    avg_cer = np.mean(cers) * 100
    avg_sim = np.mean(sims)
    
    print(f"\n================ ORTHOGONAL FACTORIZATION RESULTS (Rank={RANK_r}) ================")
    print(f" CER (%):             {avg_cer:.2f}%")
    print(f" Similarity (Cosine): {avg_sim:.4f}")
    print("==================================================================================")
    
if __name__ == "__main__":
    main()
