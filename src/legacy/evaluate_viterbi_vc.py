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

def viterbi_alignment(X_src, X_tgt, jump_penalty=0.1, stay_penalty=0.05):
    """
    X_src: (T, D)
    X_tgt: (N, D)
    """
    T = X_src.shape[0]
    N = X_tgt.shape[0]
    
    # Compute Cosine Distance Matrix (T, N)
    # Cosine Similarity is in [-1, 1]. Distance = 1 - Sim in [0, 2]
    sim_matrix = cosine_similarity(X_src, X_tgt)
    emission_cost = 1.0 - sim_matrix
    
    V = np.zeros((T, N))
    ptr = np.zeros((T, N), dtype=int)
    
    # Initialization
    V[0, :] = emission_cost[0, :]
    
    for t in range(1, T):
        # The minimum cost of jumping from anywhere in the previous step
        min_prev = np.min(V[t-1, :])
        jump_cost = min_prev + jump_penalty
        best_jump_idx = np.argmin(V[t-1, :])
        
        for j in range(N):
            # Option 1: Jump from anywhere
            cost_jump = jump_cost
            idx_jump = best_jump_idx
            
            # Option 2: Advance naturally (k -> k+1)
            if j > 0:
                cost_adv = V[t-1, j-1] # No transition penalty for natural advance
                idx_adv = j - 1
            else:
                cost_adv = np.inf
                idx_adv = -1
                
            # Option 3: Stay (k -> k)
            cost_stay = V[t-1, j] + stay_penalty
            idx_stay = j
            
            # Find minimum of the three options
            costs = [cost_jump, cost_adv, cost_stay]
            indices = [idx_jump, idx_adv, idx_stay]
            
            best_choice = np.argmin(costs)
            V[t, j] = emission_cost[t, j] + costs[best_choice]
            ptr[t, j] = indices[best_choice]
            
    # Backtrack
    aligned_indices = np.zeros(T, dtype=int)
    aligned_indices[-1] = np.argmin(V[-1, :])
    for t in range(T-2, -1, -1):
        aligned_indices[t] = ptr[t+1, aligned_indices[t+1]]
        
    return aligned_indices

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

    print("\n3. Running VITERBI-LinearVC Pipeline...")
    df_src = df_all[~df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")].sample(n=25, random_state=42)
    src_paths = df_src['path'].tolist()
    
    cers = []
    sims = []
    
    for p in tqdm(src_paths, desc="Evaluating Viterbi-VC"):
        X_src, w_src = extract_features(p, wavlm_large)
        ref_text = transcribe(w_src)
        
        # 1. Viterbi Alignment (Replaces kNN!)
        indices = viterbi_alignment(X_src, X_tgt_pool, jump_penalty=0.15, stay_penalty=0.05)
        
        X_tgt_matched = X_tgt_pool[indices]
        
        # 2. Linear Regression (W) as usual
        W, _, _, _ = np.linalg.lstsq(X_src, X_tgt_matched, rcond=None)
        
        # 3. Convert
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
    
    print("\n================ VITERBI DECODING RESULTS ================")
    print(f" CER (%):             {avg_cer:.2f}%")
    print(f" Similarity (Cosine): {avg_sim:.4f}")
    print("==========================================================")
    
if __name__ == "__main__":
    main()
