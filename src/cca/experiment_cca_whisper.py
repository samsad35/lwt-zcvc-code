import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from une import LibriSpeech
from une import WavLM, WhisperEncoderWrapper

def extract_aligned_layer_latents(model_x, model_y, audio_paths, layer_x, layer_y, desc_msg):
    latents_x = []
    latents_y = []
    with torch.no_grad():
        for path in tqdm(audio_paths, desc=desc_msg):
            hx = model_x(str(path))
            hy = model_y(str(path))
            
            # Alignement temporel : on gère le padding de 30s de Whisper
            Tx = hx[layer_x].shape[1]
            Ty = hy[layer_y].shape[1]
            min_T = min(Tx, Ty)
            
            latents_x.append(hx[layer_x].squeeze(0)[:min_T].cpu().numpy())
            latents_y.append(hy[layer_y].squeeze(0)[:min_T].cpu().numpy())
            
    return np.concatenate(latents_x, axis=0), np.concatenate(latents_y, axis=0)

def main():
    dataset_path = "/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean"
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("1. Préparation des données (60 Train / 20 Test)...")
    librispeech = LibriSpeech(root=Path(dataset_path), ext="flac")
    librispeech.generate_table()
    
    df_train = librispeech.table.sample(n=60, random_state=42)
    df_test = librispeech.table.drop(df_train.index).sample(n=20, random_state=42)
    
    train_paths = df_train['path'].tolist()
    test_paths = df_test['path'].tolist()
    
    print("\n2. Chargement des modèles...")
    wavlm = WavLM(model_name="microsoft/wavlm-base", device=device)
    whisper = WhisperEncoderWrapper(model_name="openai/whisper-base", device=device)
    
    LAYER_X = 12 # Dernière couche WavLM
    LAYER_Y = 6  # Dernière couche Whisper
    
    print(f"\n3. Extraction et Alignement temporel (Train)...")
    X_train, Y_train = extract_aligned_layer_latents(wavlm, whisper, train_paths, LAYER_X, LAYER_Y, "Train WavLM & Whisper")
    
    print(f"\n4. Extraction et Alignement temporel (Test)...")
    X_test, Y_test = extract_aligned_layer_latents(wavlm, whisper, test_paths, LAYER_X, LAYER_Y, "Test WavLM & Whisper")
    
    del wavlm, whisper
    torch.cuda.empty_cache()
    
    print("\n5. Apprentissage de l'espace commun (CCA complète)...")
    X_mean = np.mean(X_train, axis=0)
    Y_mean = np.mean(Y_train, axis=0)
    
    X_c = X_train - X_mean
    Y_c = Y_train - Y_mean
    
    Xt = torch.tensor(X_c, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y_c, dtype=torch.float32, device=device)
    
    Q_x, R_x = torch.linalg.qr(Xt)
    Q_y, R_y = torch.linalg.qr(Yt)
    
    C = torch.matmul(Q_x.T, Q_y)
    U, S, Vh = torch.linalg.svd(C)
    
    Wx = torch.linalg.solve(R_x, U)
    Wy = torch.linalg.solve(R_y, Vh.T)
    
    # 512 est le maximum possible puisque Whisper n'a que 512 dimensions !
    dims_to_test = [16, 32, 64, 128, 256, 512]
    results = []
    
    print("\n6. Évaluation pour différentes tailles d'espace commun...")
    Y_test_c = torch.tensor(Y_test - Y_mean, dtype=torch.float32, device=device)
    
    for dim in dims_to_test:
        print(f"-> Évaluation pour dim = {dim}...")
        Wy_sub = Wy[:, :dim]
        
        # Mapping Least Squares Train
        C_y_train = torch.matmul(Yt, Wy_sub)
        M = torch.linalg.lstsq(C_y_train, Xt).solution
        
        # Projection Test
        C_y_test = torch.matmul(Y_test_c, Wy_sub)
        X_pred_c = torch.matmul(C_y_test, M)
        X_pred = X_pred_c.cpu().numpy() + X_mean
        
        # Metrics
        mse = np.mean((X_test - X_pred)**2)
        
        num = np.sum(X_test * X_pred, axis=1)
        den = np.linalg.norm(X_test, axis=1) * np.linalg.norm(X_pred, axis=1)
        cos_sim = np.mean(num / (den + 1e-8))
        
        results.append({
            "n_components": dim,
            "MSE": mse,
            "Cosine Similarity": cos_sim
        })
        print(f"   MSE: {mse:.4f} | Cosine: {cos_sim:.4f}")

    print("\n7. Génération des graphiques...")
    df_res = pd.DataFrame(results)
    
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "#333"})
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = '#1f77b4'
    ax1.set_xlabel("Dimension de l'Espace Commun (n_components)", fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel("Cosine Similarity", color=color, fontsize=12, fontweight='bold', labelpad=10)
    sns.lineplot(data=df_res, x="n_components", y="Cosine Similarity", marker="o", markersize=10, linewidth=3, color=color, ax=ax1)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.0)
    
    ax2 = ax1.twinx()
    color = '#d62728'
    ax2.set_ylabel("MSE (Mean Squared Error)", color=color, fontsize=12, fontweight='bold', labelpad=10)
    sns.lineplot(data=df_res, x="n_components", y="MSE", marker="s", markersize=10, linewidth=3, color=color, ax=ax2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    ax1.set_xticks(dims_to_test)
    ax1.set_xticklabels([str(d) for d in dims_to_test])
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax2.grid(False)
    
    plt.title("Reconstruction Whisper (Couche 6) -> WavLM (Couche 12) via CCA", fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    
    out_file = output_dir / "cca_reconstruction_whisper.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"🎉 Courbes sauvegardées dans : {out_file}")

if __name__ == "__main__":
    main()
