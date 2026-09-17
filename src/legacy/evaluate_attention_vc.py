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
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

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

    print("\n2. Loading Dataset & Preparing Target Dictionary...")
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
    
    tgt_embs = [get_embedding(w.numpy()) for w in tgt_wavs[:10]]
    tgt_profile = torch.stack(tgt_embs).mean(dim=0)
    tgt_profile = torch.nn.functional.normalize(tgt_profile, dim=0)

    print("\n3. Running ATTENTION-VC (Nadaraya-Watson Kernel Regression)...")
    df_src = df_all[~df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")].sample(n=25, random_state=42)
    src_paths = df_src['path'].tolist()
    
    cers = []
    sims = []
    
    TEMPERATURE = 0.05  # Controls the blur (0 = kNN, inf = Global Average)
    
    for p in tqdm(src_paths, desc="Evaluating Attention-VC"):
        X_src, w_src = extract_features(p, wavlm_large)
        ref_text = transcribe(w_src)
        
        # --- ATTENTION MECHANISM (Nadaraya-Watson) ---
        # 1. Compute Cosine Similarity matrix (T, N_tgt)
        sim_matrix = cosine_similarity(X_src, X_tgt_pool)
        
        # 2. Apply Softmax with Temperature along the target axis
        # We use similarity directly because higher similarity -> higher weight
        W_attn = softmax(sim_matrix / TEMPERATURE, axis=1) # (T, N_tgt)
        
        # 3. Compute Soft-Aligned Features
        X_aligned = np.matmul(W_attn, X_tgt_pool)
        
        # Convert directly (Zero-Shot, no linear regression W needed since it's already soft)
        wav_conv = vocode(X_aligned, hifigan, device)
        
        conv_text = transcribe(wav_conv.squeeze().numpy())
        cer = jiwer.cer(ref_text, conv_text)
        cers.append(cer)
        
        conv_emb = get_embedding(wav_conv.squeeze().numpy())
        conv_emb = torch.nn.functional.normalize(conv_emb, dim=0)
        sim = torch.dot(tgt_profile, conv_emb).item()
        sims.append(sim)
            
    avg_cer = np.mean(cers) * 100
    avg_sim = np.mean(sims)
    
    print(f"\n================ ATTENTION-VC RESULTS (Tau={TEMPERATURE}) ================")
    print(f" CER (%):             {avg_cer:.2f}%")
    print(f" Similarity (Cosine): {avg_sim:.4f}")
    print("=========================================================================")
    
if __name__ == "__main__":
    main()
