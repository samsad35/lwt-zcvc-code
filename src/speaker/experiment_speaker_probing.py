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
from une import WavLM, HuBERTEncoder

def get_speaker_id(path_str):
    return Path(path_str).parts[-3]

def extract_features(model, audio_paths, layer_idx, desc_msg, apply_in=False):
    utterance_latents = []
    speaker_ids = []
    
    with torch.no_grad():
        for path in tqdm(audio_paths, desc=desc_msg):
            h = model(str(path))[layer_idx].squeeze(0).cpu().numpy()
            
            if apply_in:
                mean = np.mean(h, axis=0, keepdims=True)
                std = np.std(h, axis=0, keepdims=True) + 1e-8
                h = (h - mean) / std
                
            # Mean Pooling over time -> 1 vector per audio
            utterance_latents.append(np.mean(h, axis=0))
            speaker_ids.append(get_speaker_id(path))
            
    return np.array(utterance_latents), np.array(speaker_ids)

def extract_cca_frames(model, audio_paths, layer_idx, desc_msg):
    latents = []
    with torch.no_grad():
        for path in tqdm(audio_paths, desc=desc_msg):
            h = model(str(path))[layer_idx].squeeze(0).cpu().numpy()
            latents.append(h)
    return np.concatenate(latents, axis=0)

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
    
    TARGET_LAYER = 8
    
    print("\n3. Extraction des features (Mean Pooling)...")
    # CCA Set (Frames)
    X_cca_frames = extract_cca_frames(wavlm, cca_paths, TARGET_LAYER, "CCA WavLM")
    Y_cca_frames = extract_cca_frames(hubert, cca_paths, TARGET_LAYER, "CCA HuBERT")
    
    # Probe Train/Test Set (Mean Pooled)
    X_ptrain, y_ptrain = extract_features(wavlm, probe_train_paths, TARGET_LAYER, "Probe Train WavLM Raw", apply_in=False)
    X_ptest, y_ptest = extract_features(wavlm, probe_test_paths, TARGET_LAYER, "Probe Test WavLM Raw", apply_in=False)
    
    X_ptrain_in, _ = extract_features(wavlm, probe_train_paths, TARGET_LAYER, "Probe Train WavLM IN", apply_in=True)
    X_ptest_in, _ = extract_features(wavlm, probe_test_paths, TARGET_LAYER, "Probe Test WavLM IN", apply_in=True)
    
    Y_ptrain, _ = extract_features(hubert, probe_train_paths, TARGET_LAYER, "Probe Train HuBERT", apply_in=False)
    Y_ptest, _ = extract_features(hubert, probe_test_paths, TARGET_LAYER, "Probe Test HuBERT", apply_in=False)
    
    del wavlm, hubert
    torch.cuda.empty_cache()
    
    print("\n4. Apprentissage de l'espace commun CCA...")
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
    labels = ['WavLM Brut\n(Mean Pool)', 'WavLM + IN\n(Mean Pool)', f'WavLM Reconstruit\nvia CCA HuBERT']
    accuracies = [acc_orig * 100, acc_in * 100, acc_recon * 100]
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = sns.barplot(x=labels, y=accuracies, palette=["#2ecc71", "#f39c12", "#e74c3c"], ax=ax)
    
    ax.set_ylabel("Speaker Recognition Accuracy (%)", fontweight='bold')
    ax.set_ylim(0, 105)
    
    for i, bar in enumerate(bars.patches):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f"{accuracies[i]:.1f}%",
                ha='center', va='bottom', fontweight='bold', fontsize=12)
                
    plt.title("Probing Identité Locuteur (Utterance-level)\nEffet Normalisation (IN) vs Goulot CCA", 
              fontsize=14, fontweight='bold', pad=15)
    
    out_file = output_dir / "cca_speaker_probing_in_mean.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"🎉 Graphique sauvegardé dans : {out_file}")

if __name__ == "__main__":
    main()
