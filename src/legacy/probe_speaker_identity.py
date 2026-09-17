import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torchaudio.functional as F
import sys
import warnings

from transformers import Wav2Vec2Model, HubertModel, Wav2Vec2ForCTC, Wav2Vec2Processor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.append(str(Path(__file__).parent.parent))
from une import LibriSpeech, WavLM

device = "cpu"

def extract_features(wav_path, wavlm_large, wavlm_asr, hubert, asr_model, processor):
    wav, sr = torchaudio.load(str(wav_path))
    wav = wav.to(device)
    if sr != 16000:
        wav = F.resample(wav, sr, 16000)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    
    with torch.no_grad():
        features_large, _ = wavlm_large.extract_features(wav, output_layer=6)
        x = features_large.squeeze(0).cpu().numpy()
        
        y1 = wavlm_asr(str(wav_path))[12].squeeze(0).cpu().numpy()
        
        hubert_out = hubert(wav, output_hidden_states=True)
        y2 = hubert_out.hidden_states[9].squeeze(0).cpu().numpy()
        
        inputs = processor(wav.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt").to(device)
        y3 = asr_model(inputs.input_values).logits.squeeze(0).cpu().numpy()
        
    min_T = min(x.shape[0], y1.shape[0], y2.shape[0], y3.shape[0])
    return x[:min_T], y1[:min_T], y2[:min_T], y3[:min_T]

def learn_cca(X_list, Y_list):
    X_frames = np.concatenate(X_list, axis=0)
    Y_frames = np.concatenate(Y_list, axis=0)
    
    X_mean = np.mean(X_frames, axis=0)
    Y_mean = np.mean(Y_frames, axis=0)
    
    Xt = torch.tensor(X_frames - X_mean, dtype=torch.float32)
    Yt = torch.tensor(Y_frames - Y_mean, dtype=torch.float32)
    
    Q_x, R_x = torch.linalg.qr(Xt)
    Q_y, R_y = torch.linalg.qr(Yt)
    
    C = torch.matmul(Q_x.T, Q_y)
    U, S, Vh = torch.linalg.svd(C)
    Wy = torch.linalg.solve(R_y, Vh.T).numpy()
    
    return Y_mean, Wy

def main():
    warnings.filterwarnings("ignore")
    print("1. Loading Models...")
    
    wavlm_large = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device).eval()
    wavlm_asr = WavLM(model_name="patrickvonplaten/wavlm-libri-clean-100h-base-plus", device=device)
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()
    
    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    librispeech = LibriSpeech(root=dataset_path, ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    
    print("\n2. Computing CCA Spaces (50 files)...")
    df_cca = df_all.sample(n=50, random_state=42)
    X_list, Y1_list, Y2_list, Y3_list = [], [], [], []
    for p in tqdm(df_cca['path'], desc="CCA Data"):
        x, y1, y2, y3 = extract_features(p, wavlm_large, wavlm_asr, hubert, asr_model, processor)
        X_list.append(x)
        Y1_list.append(y1)
        Y2_list.append(y2)
        Y3_list.append(y3)
        
    Y1_mean, Wy1 = learn_cca(X_list, Y1_list) # WavLM-ASR
    Y2_mean, Wy2 = learn_cca(X_list, Y2_list) # HuBERT
    Y3_mean, Wy3 = learn_cca(X_list, Y3_list) # PPGs
    
    print("\n3. Preparing Probing Dataset (10 Speakers, 5 files each)...")
    speakers = list(df_all['path'].apply(lambda x: Path(x).parts[-3]).unique())[:10]
    
    Z1_frames, Z2_frames, Z3_frames = [], [], []
    speaker_labels = []
    content_labels = []
    dim = 128
    
    for spk_id, spk in enumerate(tqdm(speakers, desc="Extracting Data")):
        paths = df_all[df_all['path'].astype(str).str.contains(f"/{spk}/")]['path'].tolist()[:5]
        for p in paths:
            x, y1, y2, y3 = extract_features(p, wavlm_large, wavlm_asr, hubert, asr_model, processor)
            
            z1 = np.matmul(y1 - Y1_mean, Wy1[:, :dim])
            z2 = np.matmul(y2 - Y2_mean, Wy2[:, :dim])
            z3 = np.matmul(y3 - Y3_mean, Wy3)
            
            Z1_frames.append(z1)
            Z2_frames.append(z2)
            Z3_frames.append(z3)
            
            speaker_labels.extend([spk_id] * z1.shape[0])
            # Use Wav2Vec2 logits argmax as phonetic ground truth
            content_labels.extend(np.argmax(y3, axis=1).tolist())
            
    Z1_X = np.concatenate(Z1_frames, axis=0)
    Z2_X = np.concatenate(Z2_frames, axis=0)
    Z3_X = np.concatenate(Z3_frames, axis=0)
    Y_spk = np.array(speaker_labels)
    Y_cnt = np.array(content_labels)
    
    print(f"\nTotal Frames to probe: {Y_spk.shape[0]}")
    
    print("\n4. Probing Speaker Identity (Lower accuracy = Better disentanglement)...")
    X_train, X_test, y_train, y_test = train_test_split(Z1_X, Y_spk, test_size=0.2, random_state=42)
    acc1_spk = accuracy_score(y_test, LogisticRegression(max_iter=100, n_jobs=-1).fit(X_train, y_train).predict(X_test))
    
    X_train2, X_test2, _, _ = train_test_split(Z2_X, Y_spk, test_size=0.2, random_state=42)
    acc2_spk = accuracy_score(y_test, LogisticRegression(max_iter=100, n_jobs=-1).fit(X_train2, y_train).predict(X_test2))
    
    X_train3, X_test3, _, _ = train_test_split(Z3_X, Y_spk, test_size=0.2, random_state=42)
    acc3_spk = accuracy_score(y_test, LogisticRegression(max_iter=100, n_jobs=-1).fit(X_train3, y_train).predict(X_test3))
    
    print("\n5. Probing Phonetic Content (Higher accuracy = Better content preservation)...")
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(Z1_X, Y_cnt, test_size=0.2, random_state=42)
    acc1_cnt = accuracy_score(y_test_c, LogisticRegression(max_iter=100, n_jobs=-1).fit(X_train_c, y_train_c).predict(X_test_c))
    
    X_train2_c, X_test2_c, _, _ = train_test_split(Z2_X, Y_cnt, test_size=0.2, random_state=42)
    acc2_cnt = accuracy_score(y_test_c, LogisticRegression(max_iter=100, n_jobs=-1).fit(X_train2_c, y_train_c).predict(X_test2_c))
    
    X_train3_c, X_test3_c, _, _ = train_test_split(Z3_X, Y_cnt, test_size=0.2, random_state=42)
    acc3_cnt = accuracy_score(y_test_c, LogisticRegression(max_iter=100, n_jobs=-1).fit(X_train3_c, y_train_c).predict(X_test3_c))
    
    print("\n================ FINAL PROBING RESULTS ================")
    print(f"{'Espace CCA':<20} | {'Speaker Leakage (↓)':<20} | {'Content Accuracy (↑)':<20}")
    print("-" * 65)
    print(f"{'WavLM-ASR (128D)':<20} | {acc1_spk*100:>17.2f}% | {acc1_cnt*100:>17.2f}%")
    print(f"{'HuBERT-L9 (128D)':<20} | {acc2_spk*100:>17.2f}% | {acc2_cnt*100:>17.2f}%")
    print(f"{'PPGs ASR (~32D)':<20} | {acc3_spk*100:>17.2f}% | {acc3_cnt*100:>17.2f}%")
    print("=========================================================")
    
if __name__ == "__main__":
    main()
