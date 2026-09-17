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
import ppgs

# Ajouter src/ au PYTHONPATH pour pouvoir importer 'une'
sys.path.append(str(Path(__file__).parent.parent))

from une import LibriSpeech
from une import WavLM

def extract_cca_frames(wavlm_target, wavlm_source, audio_paths, l_target, l_source, desc_msg):
    latents_target = []
    latents_source = []
    with torch.no_grad():
        for path in tqdm(audio_paths, desc=desc_msg):
            h_t = wavlm_target(str(path))[l_target].squeeze(0).cpu().numpy()
            h_s = wavlm_source(str(path))[l_source].squeeze(0).cpu().numpy()
            latents_target.append(h_t)
            latents_source.append(h_s)
            
    return np.concatenate(latents_target, axis=0), np.concatenate(latents_source, axis=0)

def extract_features_ppg(wavlm_target, wavlm_source, audio_paths, l_target, l_source, desc_msg, gpu=0):
    frames_target = []
    frames_source = []
    labels_ppg = []
    
    with torch.no_grad():
        for path in tqdm(audio_paths, desc=desc_msg):
            h_t = wavlm_target(str(path))[l_target].squeeze(0).cpu().numpy()
            h_s = wavlm_source(str(path))[l_source].squeeze(0).cpu().numpy()
            
            # Extract PPGs
            audio = ppgs.load.audio(str(path))
            p = ppgs.from_audio(audio, ppgs.SAMPLE_RATE, gpu=gpu).squeeze(0) # Shape: (40, T_ppg)
            
            # Downsample PPGs (100 Hz -> 50 Hz)
            p = p[:, ::2]
            # Get argmax (Phoneme class 0-39)
            p_class = torch.argmax(p, dim=0).cpu().numpy()
            
            # Align everything
            min_T = min(h_t.shape[0], h_s.shape[0], p_class.shape[0])
            h_t = h_t[:min_T]
            h_s = h_s[:min_T]
            p_class = p_class[:min_T]
            
            # Randomly sample 100 frames per utterance
            np.random.seed(42)
            if min_T > 100:
                idx = np.random.choice(min_T, size=100, replace=False)
            else:
                idx = np.arange(min_T)
                
            frames_target.append(h_t[idx])
            frames_source.append(h_s[idx])
            labels_ppg.append(p_class[idx])
            
    return np.concatenate(frames_target, axis=0), np.concatenate(frames_source, axis=0), np.concatenate(labels_ppg, axis=0)

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

def project(Y_frames, X_mean, Y_mean, Wy_sub, M, device):
    Y_c = torch.tensor(Y_frames - Y_mean, dtype=torch.float32, device=device)
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
    
    df_cca = df_all.sample(n=100, random_state=42)
    df_remaining = df_all.drop(df_cca.index)
    
    df_probe_train = df_remaining.sample(n=200, random_state=42)
    df_probe_test = df_remaining.drop(df_probe_train.index).sample(n=100, random_state=42)
    
    cca_paths = df_cca['path'].tolist()
    probe_train_paths = df_probe_train['path'].tolist()
    probe_test_paths = df_probe_test['path'].tolist()
    
    print("\n2. Chargement des modèles...")
    wavlm_base = WavLM(model_name="microsoft/wavlm-base", device=device)
    wavlm_asr = WavLM(model_name="patrickvonplaten/wavlm-libri-clean-100h-base-plus", device=device)
    
    L_TARGET = 8   # WavLM Base L8
    L_SOURCE = 12  # WavLM ASR L12
    gpu_id = 0 if device == "cuda" else None
    
    print("\n3. Extraction des features et Alignement Temporel...")
    X_cca, Y_cca = extract_cca_frames(wavlm_base, wavlm_asr, cca_paths, L_TARGET, L_SOURCE, "CCA Extraction")
    
    X_train, Y_train, y_train = extract_features_ppg(wavlm_base, wavlm_asr, probe_train_paths, L_TARGET, L_SOURCE, "Probe Train Extraction", gpu=gpu_id)
    X_test, Y_test, y_test = extract_features_ppg(wavlm_base, wavlm_asr, probe_test_paths, L_TARGET, L_SOURCE, "Probe Test Extraction", gpu=gpu_id)
    
    del wavlm_base, wavlm_asr
    torch.cuda.empty_cache()
    
    print("\n4. Apprentissage des espaces communs (CCA)...")
    DIM = 64
    print(f"   -> WavLM ASR vers WavLM Base (goulot {DIM}d)...")
    xM, yM, Wy, M = learn_cca_mapping(X_cca, Y_cca, device, dim=DIM)
    
    print("\n5. Projections via CCA...")
    X_train_recon = project(Y_train, xM, yM, Wy, M, device)
    X_test_recon = project(Y_test, xM, yM, Wy, M, device)
    
    print("\n6. Entraînement des Sondes (Probing Phonétique FRAME-LEVEL)...")
    # WavLM Brut L8
    scaler_orig = StandardScaler().fit(X_train)
    clf_orig = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1).fit(scaler_orig.transform(X_train), y_train)
    acc_orig = accuracy_score(y_test, clf_orig.predict(scaler_orig.transform(X_test)))
    
    # Recon WavLM ASR
    scaler_recon = StandardScaler().fit(X_train_recon)
    clf_recon = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1).fit(scaler_recon.transform(X_train_recon), y_train)
    acc_recon = accuracy_score(y_test, clf_recon.predict(scaler_recon.transform(X_test_recon)))
    
    print(f"\n---> Résultat Probing PPG (WavLM Base L8) : {acc_orig*100:.2f}%")
    print(f"---> Résultat Probing PPG (Reconstruit via WavLM ASR L12) : {acc_recon*100:.2f}%")
    
    print("\n7. Génération du graphique de comparaison...")
    labels = ['WavLM Base (Couche 8)', f'Reconstruit via WavLM ASR\n(Couche 12, goulot {DIM}d)']
    accuracies = [acc_orig * 100, acc_recon * 100]
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = sns.barplot(x=labels, y=accuracies, palette=["#2ecc71", "#e74c3c"], ax=ax)
    
    ax.set_ylabel("Phoneme Classification Accuracy (%)", fontweight='bold')
    ax.set_ylim(0, 105)
    
    for i, bar in enumerate(bars.patches):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f"{accuracies[i]:.1f}%",
                ha='center', va='bottom', fontweight='bold', fontsize=12)
                
    plt.title("Probing Phonétique (PPG) : Conservation par Finetuning ASR", 
              fontsize=14, fontweight='bold', pad=15)
    
    out_file = output_dir / "cca_ppg_wavlm_asr.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"🎉 Graphique sauvegardé dans : {out_file}")

if __name__ == "__main__":
    main()
