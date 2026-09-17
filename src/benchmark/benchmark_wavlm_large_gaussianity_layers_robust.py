import os, sys, time, torch, torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import anderson
import matplotlib.pyplot as plt
import seaborn as sns
import torchaudio.functional as F

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("================================================================================")
print("=== BENCHMARK ROBUSTE DE GAUSSIANITÉ SUR WAVLM-LARGE (L0 À L24)              ===")
print("================================================================================")

# Fonction de test de gaussianité par projections normalisées avec répétitions (robustesse)
def test_gaussianity_projections(matrix, num_projections=500, sample_size=250, num_draws=10):
    if len(matrix) == 0:
        return {"ad_pct": 0.0, "ad_pct_std": 0.0}
        
    # Standardize matrix
    matrix = (matrix - np.mean(matrix, axis=0)) / (np.std(matrix, axis=0) + 1e-8)
    hidden_size = matrix.shape[1]
    
    # Exécution sur plusieurs tirages bootstrap pour garantir une robustesse maximale
    draw_scores = []
    np.random.seed(42)
    
    for _ in range(num_draws):
        projections = np.random.randn(hidden_size, num_projections)
        projections = projections / np.linalg.norm(projections, axis=0, keepdims=True)
        
        if sample_size and len(matrix) > sample_size:
            indices = np.random.choice(len(matrix), sample_size, replace=False)
            sub_matrix = matrix[indices]
        else:
            sub_matrix = matrix
            
        projected = sub_matrix @ projections
        
        gaussian_count = 0
        for i in range(num_projections):
            proj_data = projected[:, i]
            result = anderson(proj_data, dist='norm')
            # index 2 corresponds to 5% significance level for normal dist in scipy
            if result.statistic < result.critical_values[2]:
                gaussian_count += 1
                
        draw_scores.append(gaussian_count / num_projections)
        
    return {
        "ad_pct": float(np.mean(draw_scores)),
        "ad_pct_std": float(np.std(draw_scores))
    }

# 1. Chargement de WavLM-Large
print("Chargement du modèle WavLM-Large...")
wavlm = torch.hub.load('bshall/knn-vc', 'wavlm_large', trust_repo=True, device=device).eval()

# 2. Échantillonnage étendu : 200 audios de LibriSpeech test-clean
dataset_root = Path('/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean')
all_audio_files = sorted(list(dataset_root.rglob('*.flac')))

np.random.seed(42)
num_audios_to_test = min(200, len(all_audio_files))
sampled_indices = np.random.choice(len(all_audio_files), num_audios_to_test, replace=False)
audio_paths = [all_audio_files[i] for i in sorted(sampled_indices)]
print(f"Chargement de {len(audio_paths)} fichiers audio (échantillonnage large) depuis {dataset_root}...")

# 3. Extraction de toutes les couches (0 à 24)
num_layers = 25
layers_to_test = list(range(num_layers))
latents_per_layer = {layer: [] for layer in layers_to_test}

print("Extraction des représentations latentes (Frame-Level) pour chaque couche...")
t0 = time.time()
with torch.no_grad():
    for idx, path in enumerate(audio_paths):
        w, sr = torchaudio.load(str(path))
        if sr != 16000:
            w = F.resample(w, sr, 16000)
        w = w.to(device)
        if w.dim() == 1:
            w = w.unsqueeze(0)
            
        for layer in layers_to_test:
            feat, _ = wavlm.extract_features(w, output_layer=layer)
            latents_per_layer[layer].append(feat.squeeze(0).cpu().numpy())
            
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(audio_paths)} audios extraits ({time.time()-t0:.1f}s)...")

flat_latents = {}
for layer in layers_to_test:
    flat_latents[layer] = np.concatenate(latents_per_layer[layer], axis=0)

total_frames = len(flat_latents[0])
hidden_dim = flat_latents[0].shape[1]
print(f"Extraction terminée : {total_frames} trames de dimension {hidden_dim} par couche en {time.time()-t0:.1f}s.")

# 4. Test d'Anderson-Darling robuste (500 projections x 10 tirages)
print("\n--- Évaluation robuste de Gaussianité (500 projections x 10 tirages bootstrap) ---")
results = []
for layer in layers_to_test:
    matrix = flat_latents[layer]
    stats = test_gaussianity_projections(matrix, num_projections=500, sample_size=250, num_draws=10)
    ad_pct = stats["ad_pct"] * 100
    ad_std = stats["ad_pct_std"] * 100
    results.append({
        "Layer_Idx": layer,
        "Couche": f"L{layer}",
        "Gaussianité (AD %)": ad_pct,
        "Gaussianité_std": ad_std
    })
    print(f"👉 Couche {layer:2d} (L{layer:2d}) : Gaussianité (AD %) = {ad_pct:5.2f}% ± {ad_std:4.2f}%")

df_results = pd.DataFrame(results)

# Sauvegarde
out_tables = Path('/local_scratch/ssadok/un_projet_audio/output/tables')
out_tables.mkdir(parents=True, exist_ok=True)
csv_path = out_tables / 'wavlm_large_gaussianity_layers_robust.csv'
df_results.to_csv(csv_path, index=False)
print(f"\nRésultats sauvegardés dans : {csv_path}")

# 5. Tracé de la figure sans texte, avec juste le trait vertical à la couche 6
out_figures = Path('/local_scratch/ssadok/un_projet_audio/output/figures')
out_figures.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="ticks", rc={
    "font.family": "sans-serif",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333"
})

fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=300)

col_wavlm = '#2a9d8f'  # Persian Green / Teal
col_line = '#e63946'    # Crimson Red pour le trait vertical

# Courbe avec intervalle d'incertitude
y_mean = df_results["Gaussianité (AD %)"]
y_std = df_results["Gaussianité_std"]

ax.plot(
    df_results["Layer_Idx"], 
    y_mean, 
    marker='o', 
    markersize=7.5, 
    linewidth=2.4, 
    color=col_wavlm, 
    label='WavLM-Large'
)
ax.fill_between(
    df_results["Layer_Idx"],
    y_mean - y_std,
    y_mean + y_std,
    color=col_wavlm,
    alpha=0.20
)

# Trait vertical unique à la couche 6 (aucun texte sur la figure)
ax.axvline(x=6, color=col_line, linestyle='--', linewidth=1.8, alpha=0.9, label='Layer 6 (LWT feature space)')

# Point marqué sur la couche 6
l6_val = df_results.loc[df_results['Layer_Idx'] == 6, 'Gaussianité (AD %)'].values[0]
ax.plot([6], [l6_val], marker='o', markersize=10, color=col_line, zorder=5)

ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['top'].set_color('#333333')
ax.spines['right'].set_color('#333333')

ax.set_title("Evolution of Latent Space Gaussianity across Layers (WavLM-Large)", fontsize=13, fontweight='bold', pad=16)
ax.set_xlabel("Transformer Layer Index", fontsize=11, fontweight='bold', labelpad=10)
ax.set_ylabel("Gaussian Fraction (AD Test %)", fontsize=11, fontweight='bold', labelpad=10)

all_layer_indices = sorted(df_results["Layer_Idx"].unique())
ax.set_xticks(all_layer_indices)
ax.set_xticklabels([f"L{l}" for l in all_layer_indices], fontsize=9)

ax.set_ylim(0, 105)
ax.grid(True, linestyle="--", alpha=0.45, which="major", axis="y")
ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10.5)

plt.tight_layout()

out_png = out_figures / "figure_wavlm_large_gaussianity_layers.png"
out_pdf = out_figures / "figure_wavlm_large_gaussianity_layers.pdf"
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()

print(f"🎉 Graphes générés avec succès :")
print(f"   PNG : {out_png}")
print(f"   PDF : {out_pdf}")
print("\n[FIN] Benchmark robuste terminé avec succès !")
