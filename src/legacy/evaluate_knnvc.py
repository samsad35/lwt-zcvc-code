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
import time

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

    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    librispeech = LibriSpeech(root=dataset_path, ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    
    tgt_spk = "121"
    
    df_src = df_all[~df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")].sample(n=15, random_state=42)
    src_paths = df_src['path'].tolist()
    
    source_data = []
    for p in src_paths:
        x, w = extract_features(p, wavlm_large)
        ref_text = transcribe(w)
        source_data.append({"x": x, "wav": w, "text": ref_text})
        
    T_size = 20
    tgt_paths = df_all[df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")]['path'].tolist()[:T_size]
    X_tgt_list, tgt_wavs = [], []
    for p in tgt_paths:
        x, w = extract_features(p, wavlm_large)
        X_tgt_list.append(x)
        tgt_wavs.append(w)
        
    X_tgt_pool = np.concatenate(X_tgt_list, axis=0)
    X_tgt_pool_tensor = torch.tensor(X_tgt_pool, dtype=torch.float32, device=device)
    
    tgt_embs = [get_embedding(w) for w in tgt_wavs[:10]]
    tgt_profile = torch.stack(tgt_embs).mean(dim=0)
    tgt_profile = torch.nn.functional.normalize(tgt_profile, dim=0)
    
    cers = []
    sims = []
    inf_times = []
    
    k = 4 # Default for kNN-VC
    for src in source_data:
        X_src = src["x"]
        
        start_t = time.perf_counter()
        X_src_tensor = torch.tensor(X_src, dtype=torch.float32, device=device)
        dists = fast_cosine_dist(X_src_tensor, X_tgt_pool_tensor)
        
        # kNN-VC logic: average of top-k closest frames
        best_indices = dists.topk(k, dim=1, largest=False).indices.cpu().numpy()
        
        # Gather and average
        X_tgt_matched = np.zeros_like(X_src)
        for i in range(X_src.shape[0]):
            X_tgt_matched[i] = np.mean(X_tgt_pool[best_indices[i]], axis=0)
        
        end_t = time.perf_counter()
        inf_times.append((end_t - start_t) / (X_src.shape[0] / 50.0)) # RTF
            
        wav_conv = vocode(X_tgt_matched, hifigan, device)
        
        conv_text = transcribe(wav_conv)
        cer = jiwer.cer(src["text"], conv_text)
        cers.append(cer)
        
        conv_emb = get_embedding(wav_conv)
        conv_emb = torch.nn.functional.normalize(conv_emb, dim=0)
        sim = torch.dot(tgt_profile, conv_emb).item()
        sims.append(sim)
        
    avg_cer = np.mean(cers) * 100
    avg_sim = np.mean(sims)
    avg_rtf = np.mean(inf_times)
    
    print(f"\n================ kNN-VC (T={T_size}, k={k}) ================")
    print(f"CER: {avg_cer:.2f}% | Sim: {avg_sim:.4f} | RTF: {avg_rtf:.4f}")

if __name__ == "__main__":
    main()
