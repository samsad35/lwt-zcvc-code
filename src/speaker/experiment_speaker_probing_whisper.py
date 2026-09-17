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

from une import LibriSpeech
from une import WavLM, WhisperEncoderWrapper

def get_speaker_id(path_str):
    return Path(path_str).parts[-3]

def extract_features_aligned(model_x, model_y, audio_paths, layer_x, layer_y, desc_msg, apply_in=False):
    utterance_latents_x = []
    utterance_latents_y = []
    speaker_ids = []
    
    with torch.no_grad():
        for path in tqdm(audio_paths, desc=desc_msg):
            hx = model_x(str(path))[layer_x].squeeze(0).cpu().numpy()
            hy = model_y(str(path))[layer_y].squeeze(0).cpu().numpy()
            
            min_T = min(hx.shape[0], hy.shape[0])
            hx = hx[:min_T]
            hy = hy[:min_T]
            
            if apply_in:
                mean_x = np.mean(hx, axis=0, keepdims=True)
                std_x = np.std(hx, axis=0, keepdims=True) + 1e-8
                hx = (hx - mean_x) / std_x
                
            utterance_latents_x.append(np.mean(hx, axis=0))
            utterance_latents_y.append(np.mean(hy, axis=0))
            speaker_ids.append(get_speaker_id(path))
            
    return np.array(utterance_latents_x), np.array(utterance_latents_y), np.array(speaker_ids)

def extract_cca_frames_aligned(model_x, model_y, audio_paths, layer_x, layer_y, desc_msg):
    latents_x = []
    latents_y = []
    with torch.no_grad():
        for path in tqdm(audio_paths, desc=desc_msg):
            hx = model_x(str(path))[layer_x].squeeze(0).cpu().numpy()
            hy = model_y(str(path))[layer_y].squeeze(0).cpu().numpy()
            
            min_T = min(hx.shape[0], hy.shape[0])
            latents_x.append(hx[:min_T])
            latents_y.append(hy[:min_T])
            
    return np.concatenate(latents_x, axis=0), np.concatenate(latents_y, axis=0)

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
    whisper = WhisperEncoderWrapper(model_name="openai/whisper-base", device=device)
    
    LAYER_X = 12
    LAYER_Y = 6
    
    print("\n3. Extraction des features et Alignement Temporel...")
    # CCA Set (Frames)
    X_cca_frames, Y_cca_frames = extract_cca_frames_aligned(wavlm, whisper, cca_paths, LAYER_X, LAYER_Y, "CCA Extraction")
    
    # Probe Train/Test Set (Mean Pooled)
    X_ptrain, Y_ptrain, y_ptrain = extract_features_aligned(wavlm, whisper, probe_train_paths, LAYER_X, LAYER_Y, "Probe Train Extraction", apply_in=False)
    X_ptest, Y_ptest, y_ptest = extract_features_aligned(wavlm, whisper, probe_test_paths, LAYER_X, LAYER_Y, "Probe Test Extraction", apply_in=False)
    
    X_ptrain_in, _, _ = extract_features_aligned(wavlm, whisper, probe_train_paths, LAYER_X, LAYER_Y, "Probe Train (IN) Extraction", apply_in=True)
    X_ptest_in, _, _ = extract_features_aligned(wavlm, whisper, probe_test_paths, LAYER_X, LAYER_Y, "Probe Test (IN) Extraction", apply_in=True)
    
    del wavlm, whisper
    torch.cuda.empty_cache()
    
    print("\n4. Apprentissage de l'espace commun CCA (Whisper -> WavLM)...")
    X_mean = np.mean(X_cca_frames, axis=0)
    Y_mean = np.mean(Y_cca_frames, axis=0)
    
    X_c = X_cca_frames - X_mean
    Y_c = Y_cca_frames - Y_mean
    
    Xt = torch.tensor(X_c, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y_c, dtype=torch.float32, device=device)
    
    Q_x, R_x = torch.linalg.qr(Xt)
    Q_y, R_y = torch.linalg.qr(Yt)
    
    C = torch.matmul(Q_x.T, Q_y)
    U, S, Vh = torch.linalg.svd(C)
    
    Wx = torch.linalg.solve(R_x, U)
    Wy = torch.linalg.solve(R_y, Vh.T)
    
    DIM = 64
    Wy_sub = Wy[:, :DIM]
    
    C_y_train = torch.matmul(Yt, Wy_sub)
    M = torch.linalg.lstsq(C_y_train, Xt).solution
    
    print("\n5. Projection via CCA...")
    def project_y_to_x(Y_utt):
        Y_c = torch.tensor(Y_utt - Y_mean, dtype=torch.float32, device=device)
        C_y = torch.matmul(Y_c, Wy_sub)
        X_pred_c = torch.matmul(C_y, M)
        return X_pred_c.cpu().numpy() + X_mean

    X_ptrain_recon = project_y_to_x(Y_ptrain)
    X_ptest_recon = project_y_to_x(Y_ptest)
    
    print("\n6. Entraînement des Sondes (Probing Utterance-level)...")
    scaler_orig = StandardScaler().fit(X_ptrain)
    clf_orig = LogisticRegression(max_iter=1000, random_state=42)
    clf_orig.fit(scaler_orig.transform(X_ptrain), y_ptrain)
    acc_orig = accuracy_score(y_ptest, clf_orig.predict(scaler_orig.transform(X_ptest)))
    
    scaler_in = StandardScaler().fit(X_ptrain_in)
    clf_in = LogisticRegression(max_iter=1000, random_state=42)
    clf_in.fit(scaler_in.transform(X_ptrain_in), y_ptrain)
    acc_in = accuracy_score(y_ptest, clf_in.predict(scaler_in.transform(X_ptest_in)))
    
    scaler_recon = StandardScaler().fit(X_ptrain_recon)
    clf_recon = LogisticRegression(max_iter=1000, random_state=42)
    clf_recon.fit(scaler_recon.transform(X_ptrain_recon), y_ptrain)
    acc_recon = accuracy_score(y_ptest, clf_recon.predict(scaler_recon.transform(X_ptest_recon)))
    
    print("\n7. Génération du graphique de comparaison...")
    labels = ['WavLM Brut L12\n(Mean Pool)', 'WavLM + IN\n(Mean Pool)', f'WavLM Reconstruit\nvia Whisper L6 CCA']
    accuracies = [acc_orig * 100, acc_in * 100, acc_recon * 100]
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(x=labels, y=accuracies, palette=["#2ecc71", "#f39c12", "#9b59b6"], ax=ax)
    
    ax.set_ylabel("Speaker Recognition Accuracy (%)", fontweight='bold')
    ax.set_ylim(0, 105)
    
    for i, bar in enumerate(bars.patches):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f"{accuracies[i]:.1f}%",
                ha='center', va='bottom', fontweight='bold', fontsize=12)
                
    plt.title("Probing Identité Locuteur (Utterance-level)\nEffet Normalisation vs Goulot CCA (Whisper)", 
              fontsize=14, fontweight='bold', pad=15)
    
    out_file = output_dir / "cca_speaker_probing_whisper_mean.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"🎉 Graphique sauvegardé dans : {out_file}")

if __name__ == "__main__":
    main()
