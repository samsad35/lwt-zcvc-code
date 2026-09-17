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
    
    eps = 1e-4
    eigenvalues = np.maximum(eigenvalues, eps)
    
    D_inv_half = np.diag(1.0 / np.sqrt(eigenvalues))
    D_half = np.diag(np.sqrt(eigenvalues))
    
    W = eigenvectors @ D_inv_half @ eigenvectors.T
    C = eigenvectors @ D_half @ eigenvectors.T
    return mu, W, C

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

    print("\n2. Loading Dataset & Computing Global WCT Matrices...")
    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    librispeech = LibriSpeech(root=dataset_path, ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    df_all['spk_id'] = df_all['path'].apply(lambda x: Path(x).parts[-3])
    unique_all_spks = df_all['spk_id'].unique()
    
    tgt_spk = "121"
    src_spk = unique_all_spks[1] if unique_all_spks[0] == "121" else unique_all_spks[0]
    
    # ---------------- TARGET POOL ----------------
    tgt_paths = df_all[df_all['spk_id'] == tgt_spk]['path'].tolist()[:30]
    X_tgt_list, tgt_wavs = [], []
    for p in tqdm(tgt_paths, desc="Target Data"):
        x, w = extract_features(p, wavlm_large)
        X_tgt_list.append(x)
        tgt_wavs.append(w)
    X_tgt_pool = np.concatenate(X_tgt_list, axis=0)
    
    tgt_embs = [get_embedding(w.numpy()) for w in tgt_wavs[:15]]
    tgt_profile = torch.stack(tgt_embs).mean(dim=0)
    tgt_profile = torch.nn.functional.normalize(tgt_profile, dim=0)

    # ---------------- SOURCE POOL ----------------
    src_paths = df_all[df_all['spk_id'] == src_spk]['path'].tolist()
    # Use first 20 for global source pool
    X_src_list = []
    for p in tqdm(src_paths[:20], desc="Source Global Pool"):
        x, _ = extract_features(p, wavlm_large)
        X_src_list.append(x)
    X_src_pool = np.concatenate(X_src_list, axis=0)
    
    N_COMPONENTS = 1024
    print(f"\n   Computing Global 1024D Matrices for Target and Source...")
    mu_tgt, _, C_tgt = get_wct_matrices(X_tgt_pool, k=N_COMPONENTS)
    mu_src, W_src, _ = get_wct_matrices(X_src_pool, k=N_COMPONENTS)

    print("\n3. Evaluating Global WCT-VC on Source test set...")
    # Evaluate on sentences from the same source speaker
    test_paths = src_paths[10:35]
    
    cers = []
    sims = []
    
    for p in tqdm(test_paths, desc="Evaluating Global WCT-VC"):
        X_src_utt, w_src = extract_features(p, wavlm_large)
        ref_text = transcribe(w_src)
        
        # 1. Whiten using GLOBAL Source Matrix
        X_whitened = np.matmul(X_src_utt - mu_src, W_src)
        
        # 2. Color with GLOBAL Target Matrix
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
    
    print(f"\n================ GLOBAL WCT-VC RESULTS (Dim={N_COMPONENTS}) ================")
    print(f" CER (%):             {avg_cer:.2f}%")
    print(f" Similarity (Cosine): {avg_sim:.4f}")
    print("=========================================================")
    
if __name__ == "__main__":
    main()
