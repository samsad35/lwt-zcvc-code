import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from une import LibriSpeech
from une import WavLM, HuBERTEncoder

def extract_layer_latents(model, audio_paths, layer_idx, desc_msg):
    latents = []
    with torch.no_grad():
        for path in tqdm(audio_paths, desc=desc_msg):
            h = model(str(path))
            latents.append(h[layer_idx].squeeze(0).cpu().numpy())
    return np.concatenate(latents, axis=0)

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
    hubert = HuBERTEncoder(model_name="facebook/hubert-base-ls960", device=device)
    
    TARGET_LAYER = 8
    
    print(f"\n3. Extraction de la couche {TARGET_LAYER}...")
    X_train = extract_layer_latents(wavlm, train_paths, TARGET_LAYER, "Train WavLM")
    Y_train = extract_layer_latents(hubert, train_paths, TARGET_LAYER, "Train HuBERT")
    
    X_test = extract_layer_latents(wavlm, test_paths, TARGET_LAYER, "Test WavLM")
    Y_test = extract_layer_latents(hubert, test_paths, TARGET_LAYER, "Test HuBERT")
    
    del wavlm, hubert
    torch.cuda.empty_cache()
    
    print("\n4. Apprentissage de l'espace commun (CCA complète)...")
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
    
    dims_to_test = [16, 32, 64, 128, 256, 512]
    results = []
    
    print("\n5. Évaluation pour différentes tailles d'espace commun (n_components)...")
    Y_test_c = torch.tensor(Y_test - Y_mean, dtype=torch.float32, device=device)
    
    for dim in dims_to_test:
        print(f"-> Évaluation pour dim = {dim}...")
        Wy_sub = Wy[:, :dim]
        
        # Train linear mapping from Common Space to WavLM
        C_y_train = torch.matmul(Yt, Wy_sub)
        M = torch.linalg.lstsq(C_y_train, Xt).solution
        
        # Test projection
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

    print("\n6. Génération des graphiques...")
    df_res = pd.DataFrame(results)
    
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "#333"})
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = '#1f77b4' # blue
    ax1.set_xlabel("Dimension de l'Espace Commun (n_components)", fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel("Cosine Similarity", color=color, fontsize=12, fontweight='bold', labelpad=10)
    sns.lineplot(data=df_res, x="n_components", y="Cosine Similarity", marker="o", markersize=10, linewidth=3, color=color, ax=ax1)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.0)
    
    ax2 = ax1.twinx()
    color = '#d62728' # red
    ax2.set_ylabel("MSE (Mean Squared Error)", color=color, fontsize=12, fontweight='bold', labelpad=10)
    sns.lineplot(data=df_res, x="n_components", y="MSE", marker="s", markersize=10, linewidth=3, color=color, ax=ax2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    ax1.set_xticks(dims_to_test)
    ax1.set_xticklabels([str(d) for d in dims_to_test])
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax2.grid(False) # avoid double grid
    
    plt.title("Reconstruction HuBERT -> WavLM en fonction du goulot dimensionnel (CCA)", fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    
    out_file = output_dir / "cca_reconstruction_dims.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"🎉 Courbes sauvegardées dans : {out_file}")

if __name__ == "__main__":
    main()
