import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torchaudio.functional as F
import torch.nn.functional as F_nn
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
        h_x = features_large.squeeze(0).cpu().numpy()
    return h_x, wav.squeeze()

def fast_cosine_dist(source_feats, target_feats):
    source_feats = F_nn.normalize(source_feats, dim=1)
    target_feats = F_nn.normalize(target_feats, dim=1)
    dist = 1 - torch.matmul(source_feats, target_feats.T)
    return dist

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

    print("\n2. Loading Dataset...")
    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    librispeech = LibriSpeech(root=dataset_path, ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    
    tgt_spk = "121"
    
    df_src = df_all[~df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")].sample(n=15, random_state=42)
    src_paths = df_src['path'].tolist()
    
    print("   Pre-extracting Source Features...")
    source_data = []
    for p in tqdm(src_paths, desc="Sources"):
        x, w = extract_features(p, wavlm_large)
        ref_text = transcribe(w)
        source_data.append({"x": x, "wav": w, "text": ref_text})
        
    TGT_SIZES = [5, 10, 20, 30]
    
    print("\n3. Running LinearVC Ablation Study...")
    
    results = []
    
    for T_size in TGT_SIZES:
        print(f"\n================ TARGET SIZE: {T_size} ==================")
        tgt_paths = df_all[df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")]['path'].tolist()[:T_size]
        X_tgt_list, tgt_wavs = [], []
        for p in tgt_paths:
            x, w = extract_features(p, wavlm_large)
            X_tgt_list.append(x)
            tgt_wavs.append(w)
            
        X_tgt_pool = np.concatenate(X_tgt_list, axis=0)
        X_tgt_pool_tensor = torch.tensor(X_tgt_pool, dtype=torch.float32, device=device)
        
        tgt_embs = [get_embedding(w) for w in tgt_wavs[:min(10, T_size)]]
        tgt_profile = torch.stack(tgt_embs).mean(dim=0)
        tgt_profile = torch.nn.functional.normalize(tgt_profile, dim=0)
        
        cers = []
        sims = []
        
        for src in source_data:
            X_src = src["x"]
            X_src_tensor = torch.tensor(X_src, dtype=torch.float32, device=device)
            dists = fast_cosine_dist(X_src_tensor, X_tgt_pool_tensor)
            best_indices = dists.argmin(dim=1).cpu().numpy()
            X_tgt_matched = X_tgt_pool[best_indices]
            
            X_src_bias = np.hstack([X_src, np.ones((X_src.shape[0], 1))])
            W_tgt, _, _, _ = np.linalg.lstsq(X_src_bias, X_tgt_matched, rcond=None)
            
            X_converted = np.matmul(X_src_bias, W_tgt)
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
        
        print(f"     [T={T_size}] CER: {avg_cer:.2f}% | Sim: {avg_sim:.4f}")
        results.append({"T_size": T_size, "CER": avg_cer, "Sim": avg_sim})
        
    print("\n\n================ ABLATION SUMMARY ================")
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()
