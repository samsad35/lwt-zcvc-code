import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
import sys

# Ajouter src/ au PYTHONPATH pour pouvoir importer 'une'
sys.path.append(str(Path(__file__).parent.parent))

from une import LibriSpeech
from une import WavLM, HuBERTEncoder, WhisperEncoderWrapper

def get_speaker_id(path_str):
    return Path(path_str).parts[-3]

def extract_features_3way(wavlm, hubert, whisper, audio_paths, l_wavlm, l_hubert, l_whisper, desc_msg):
    utt_wavlm = []
    utt_hubert = []
    utt_whisper = []
    speaker_ids = []
    
    with torch.no_grad():
        for path in tqdm(audio_paths, desc=desc_msg):
            hw = wavlm(str(path))[l_wavlm].squeeze(0).cpu().numpy()
            hh = hubert(str(path))[l_hubert].squeeze(0).cpu().numpy()
            hsp = whisper(str(path))[l_whisper].squeeze(0).cpu().numpy()
            
            min_T = min(hw.shape[0], hh.shape[0], hsp.shape[0])
            hw = hw[:min_T]
            hh = hh[:min_T]
            hsp = hsp[:min_T]
            
            utt_wavlm.append(np.mean(hw, axis=0))
            utt_hubert.append(np.mean(hh, axis=0))
            utt_whisper.append(np.mean(hsp, axis=0))
            speaker_ids.append(get_speaker_id(path))
            
    return np.array(utt_wavlm), np.array(utt_hubert), np.array(utt_whisper), np.array(speaker_ids)

def extract_cca_frames_3way(wavlm, hubert, whisper, audio_paths, l_wavlm, l_hubert, l_whisper, desc_msg):
    latents_wavlm = []
    latents_hubert = []
    latents_whisper = []
    with torch.no_grad():
        for path in tqdm(audio_paths, desc=desc_msg):
            hw = wavlm(str(path))[l_wavlm].squeeze(0).cpu().numpy()
            hh = hubert(str(path))[l_hubert].squeeze(0).cpu().numpy()
            hsp = whisper(str(path))[l_whisper].squeeze(0).cpu().numpy()
            
            min_T = min(hw.shape[0], hh.shape[0], hsp.shape[0])
            latents_wavlm.append(hw[:min_T])
            latents_hubert.append(hh[:min_T])
            latents_whisper.append(hsp[:min_T])
            
    return np.concatenate(latents_wavlm, axis=0), np.concatenate(latents_hubert, axis=0), np.concatenate(latents_whisper, axis=0)

def learn_cca_mapping(X_frames, Y_frames, device, dim=64):
    X_mean = np.mean(X_frames, axis=0)
    Y_mean = np.mean(Y_frames, axis=0)
    
    Xt = torch.tensor(X_frames - X_mean, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y_frames - Y_mean, dtype=torch.float32, device=device)
    
    Q_x, R_x = torch.linalg.qr(Xt)
    Q_y, R_y = torch.linalg.qr(Yt)
    
    C = torch.matmul(Q_x.T, Q_y)
    U, S, Vh = torch.linalg.svd(C)
    
    Wx = torch.linalg.solve(R_x, U)
    Wy = torch.linalg.solve(R_y, Vh.T)
    
    Wy_sub = Wy[:, :dim]
    
    C_y_train = torch.matmul(Yt, Wy_sub)
    M = torch.linalg.lstsq(C_y_train, Xt).solution
    
    return X_mean, Y_mean, Wy_sub, M

def project(Y_utt, X_mean, Y_mean, Wy_sub, M, device):
    Y_c = torch.tensor(Y_utt - Y_mean, dtype=torch.float32, device=device)
    C_y = torch.matmul(Y_c, Wy_sub)
    X_pred_c = torch.matmul(C_y, M)
    return X_pred_c.cpu().numpy() + X_mean

def main():
    warnings.filterwarnings("ignore")
    dataset_path = "/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean"
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("1. Préparation des données...")
    librispeech = LibriSpeech(root=Path(dataset_path), ext="flac")
    librispeech.generate_table()
    
    df_all = librispeech.table
    df_all['speaker_id'] = df_all['path'].apply(get_speaker_id)
    
    df_cca = df_all.sample(n=100, random_state=42)
    df_remaining = df_all.drop(df_cca.index)
    
    df_probe_train = df_remaining.sample(n=400, random_state=42)
    df_probe_test = df_remaining.drop(df_probe_train.index).sample(n=150, random_state=42)
    
    cca_paths = df_cca['path'].tolist()
    probe_train_paths = df_probe_train['path'].tolist()
    probe_test_paths = df_probe_test['path'].tolist()
    
    print("\n2. Chargement des modèles...")
    wavlm = WavLM(model_name="microsoft/wavlm-base", device=device)
    hubert = HuBERTEncoder(model_name="facebook/hubert-base-ls960", device=device)
    whisper = WhisperEncoderWrapper(model_name="openai/whisper-base", device=device)
    
    L_W = 12 # Target WavLM
    L_H = 8  # Source HuBERT
    L_S = 6  # Source Whisper
    
    print("\n3. Extraction des features et Alignement Temporel...")
    X_cca, Y_cca_hubert, Y_cca_whisper = extract_cca_frames_3way(wavlm, hubert, whisper, cca_paths, L_W, L_H, L_S, "CCA Extraction")
    
    X_train, Y_train_hubert, Y_train_whisper, y_train = extract_features_3way(wavlm, hubert, whisper, probe_train_paths, L_W, L_H, L_S, "Probe Train Extraction")
    X_test, Y_test_hubert, Y_test_whisper, y_test = extract_features_3way(wavlm, hubert, whisper, probe_test_paths, L_W, L_H, L_S, "Probe Test Extraction")
    
    del wavlm, hubert, whisper
    torch.cuda.empty_cache()
    
    print("\n4. Apprentissage des espaces communs (CCA)...")
    DIM = 64
    print("   -> HuBERT vers WavLM...")
    xM_h, yM_h, Wy_h, M_h = learn_cca_mapping(X_cca, Y_cca_hubert, device, dim=DIM)
    
    print("   -> Whisper vers WavLM...")
    xM_w, yM_w, Wy_w, M_w = learn_cca_mapping(X_cca, Y_cca_whisper, device, dim=DIM)
    
    print("\n5. Projections via CCA...")
    X_train_recon_hubert = project(Y_train_hubert, xM_h, yM_h, Wy_h, M_h, device)
    X_test_recon_hubert = project(Y_test_hubert, xM_h, yM_h, Wy_h, M_h, device)
    
    X_train_recon_whisper = project(Y_train_whisper, xM_w, yM_w, Wy_w, M_w, device)
    X_test_recon_whisper = project(Y_test_whisper, xM_w, yM_w, Wy_w, M_w, device)
    
    print("\n6. Entraînement des Sondes (Probing Utterance-level)...")
    # WavLM Brut
    scaler_orig = StandardScaler().fit(X_train)
    clf_orig = LogisticRegression(max_iter=1000, random_state=42).fit(scaler_orig.transform(X_train), y_train)
    acc_orig = accuracy_score(y_test, clf_orig.predict(scaler_orig.transform(X_test)))
    
    # Recon HuBERT
    scaler_h = StandardScaler().fit(X_train_recon_hubert)
    clf_h = LogisticRegression(max_iter=1000, random_state=42).fit(scaler_h.transform(X_train_recon_hubert), y_train)
    acc_h = accuracy_score(y_test, clf_h.predict(scaler_h.transform(X_test_recon_hubert)))
    
    # Recon Whisper
    scaler_w = StandardScaler().fit(X_train_recon_whisper)
    clf_w = LogisticRegression(max_iter=1000, random_state=42).fit(scaler_w.transform(X_train_recon_whisper), y_train)
    acc_w = accuracy_score(y_test, clf_w.predict(scaler_w.transform(X_test_recon_whisper)))
    
    print("\n7. Génération du graphique de comparaison...")
    labels = ['WavLM Brut L12', f'Reconstruit via\nHuBERT L8 (goulot {DIM}d)', f'Reconstruit via\nWhisper L6 (goulot {DIM}d)']
    accuracies = [acc_orig * 100, acc_h * 100, acc_w * 100]
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(x=labels, y=accuracies, palette=["#2ecc71", "#3498db", "#9b59b6"], ax=ax)
    
    ax.set_ylabel("Speaker Recognition Accuracy (%)", fontweight='bold')
    ax.set_ylim(0, 105)
    
    for i, bar in enumerate(bars.patches):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f"{accuracies[i]:.1f}%",
                ha='center', va='bottom', fontweight='bold', fontsize=12)
                
    plt.title("Probing Identité Locuteur : Comparaison des Goulots CCA\nHuBERT vs Whisper", 
              fontsize=14, fontweight='bold', pad=15)
    
    out_file = output_dir / "cca_speaker_compare_hubert_whisper.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"🎉 Graphique sauvegardé dans : {out_file}")

if __name__ == "__main__":
    main()
