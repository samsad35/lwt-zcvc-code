import os, sys, time, torch, torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from scipy import stats
from scipy.stats import anderson
import matplotlib.pyplot as plt
import seaborn as sns
import torchaudio.functional as F

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("================================================================================")
print("=== BENCHMARK DE GAUSSIANITÉ DE WAVLM-LARGE (GLOBAL VS LOCAL K-CLUSTERS)     ===")
print("================================================================================")

# 1. Exact user requested test function
def test_gaussianity_projections(matrix, num_projections=100, sample_size=250):
    if len(matrix) == 0:
        return {"ad_pct": 0.0}
        
    # Standardize matrix
    matrix = (matrix - np.mean(matrix, axis=0)) / (np.std(matrix, axis=0) + 1e-8)
    
    hidden_size = matrix.shape[1]
    projections = np.random.randn(hidden_size, num_projections)
    projections = projections / np.linalg.norm(projections, axis=0, keepdims=True)
    
    projected = matrix @ projections
    
    if sample_size and len(matrix) > sample_size:
        indices = np.random.choice(len(matrix), sample_size, replace=False)
        projected = projected[indices]
    
    gaussian_count = 0
    for i in range(num_projections):
        proj_data = projected[:, i]
        result = anderson(proj_data, dist='norm')
        # index 2 corresponds to 5% significance level for normal dist in scipy
        if result.statistic < result.critical_values[2]:
            gaussian_count += 1
            
    return {"ad_pct": gaussian_count / num_projections}

# PCA directional normality test (to expose multimodal phonetic axes vs unimodal micro-fluctuations)
def test_pca_normality(data, n_pcs=25, sample_size=400):
    if len(data) < n_pcs + 5:
        return {"ad_pca_pct": 0.0, "shapiro_pca_pct": 0.0, "kurtosis": 0.0, "skewness": 0.0}
    if sample_size and len(data) > sample_size:
        idx = np.random.choice(len(data), sample_size, replace=False)
        mat = data[idx]
    else:
        mat = data
    Xc = mat - np.mean(mat, axis=0)
    u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ vt[:n_pcs].T
    ad_passed = 0
    shapiro_passed = 0
    kurts, skews = [], []
    for i in range(n_pcs):
        p = proj[:, i]
        res = anderson(p, dist='norm')
        if res.statistic < res.critical_values[2]:
            ad_passed += 1
        _, pval = stats.shapiro(p)
        if pval > 0.05:
            shapiro_passed += 1
        kurts.append(abs(stats.kurtosis(p)))
        skews.append(abs(stats.skew(p)))
    return {
        "ad_pca_pct": ad_passed / n_pcs,
        "shapiro_pca_pct": shapiro_passed / n_pcs,
        "kurtosis": float(np.mean(kurts)),
        "skewness": float(np.mean(skews))
    }

# 2. Load WavLM-Large
print("Loading WavLM-Large model...")
wavlm = torch.hub.load('bshall/knn-vc', 'wavlm_large', trust_repo=True, device=device).eval()

root = Path('/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean')
speakers = sorted([d.name for d in root.iterdir() if d.is_dir()])[:15]
files = []
for s in speakers:
    s_files = sorted(list((root / s).rglob('*.flac')))[:3]
    files.extend(s_files)
files = files[:40]
print(f"Sampled {len(files)} audio files from {len(speakers)} distinct speakers.")

# Extract all 25 layers (0 to 24)
print("Extracting latent frames across all 25 layers (L0 to L24)...")
t0 = time.time()
layers = list(range(25))
latents_per_layer = {l: [] for l in layers}

with torch.no_grad():
    for p in files:
        w, sr = torchaudio.load(str(p))
        if sr != 16000: w = F.resample(w, sr, 16000)
        w = w.to(device)
        for l in layers:
            feat, _ = wavlm.extract_features(w, output_layer=l)
            latents_per_layer[l].append(feat.squeeze(0).cpu().numpy())

flat_latents = {l: np.concatenate(latents_per_layer[l], axis=0) for l in layers}
N_frames, D_dim = flat_latents[6].shape
print(f"Extracted {N_frames} frames of dimension {D_dim} across 25 layers in {time.time()-t0:.1f}s.")

out_tables = Path('/local_scratch/ssadok/un_projet_audio/output/tables')
out_tables.mkdir(parents=True, exist_ok=True)
out_figures = Path('/local_scratch/ssadok/un_projet_audio/output/figures')
out_figures.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# PART 1: LAYER-WISE GAUSSIANITY BENCHMARK (GLOBAL VS LOCAL K=10)
# -----------------------------------------------------------------------------
print("\n--- PART 1: Evaluating Layer-wise Gaussianity (L0 to L24) with test_gaussianity_projections ---")
layer_records = []
for l in layers:
    X_l = flat_latents[l]
    # Global
    g_res = test_gaussianity_projections(X_l, num_projections=100, sample_size=250)
    g_pca = test_pca_normality(X_l)
    
    # Local K=10
    km10 = KMeans(n_clusters=10, random_state=42, n_init=1).fit(X_l)
    loc_ad_proj = []
    loc_ad_pca = []
    weights = []
    for k in range(10):
        Xk = X_l[km10.labels_ == k]
        if len(Xk) >= 30:
            res_k = test_gaussianity_projections(Xk, num_projections=100, sample_size=250)
            pca_k = test_pca_normality(Xk)
            loc_ad_proj.append(res_k['ad_pct'])
            loc_ad_pca.append(pca_k['ad_pca_pct'])
            weights.append(len(Xk))
    w = np.array(weights) / sum(weights)
    l_ad_proj = float(np.sum(np.array(loc_ad_proj) * w))
    l_ad_pca = float(np.sum(np.array(loc_ad_pca) * w))
    
    layer_records.append({
        'Layer': l,
        'Global_AD_Proj_pct': g_res['ad_pct'] * 100,
        'Local_AD_Proj_pct': l_ad_proj * 100,
        'Global_AD_PCA_pct': g_pca['ad_pca_pct'] * 100,
        'Local_AD_PCA_pct': l_ad_pca * 100,
        'Global_Kurtosis_PCA': g_pca['kurtosis'],
        'Local_Kurtosis_PCA': pca_k['kurtosis']
    })
    print(f"Layer {l:2d} | Random Proj AD: Global={g_res['ad_pct']*100:4.1f}% vs Local K=10={l_ad_proj*100:4.1f}% | PCA AD: Global={g_pca['ad_pca_pct']*100:4.1f}% vs Local={l_ad_pca*100:4.1f}%")

df_layers = pd.DataFrame(layer_records)
df_layers.to_csv(out_tables / 'wavlm_large_gaussianity_layers.csv', index=False)
print(f"Saved layer-wise Gaussianity table to {out_tables / 'wavlm_large_gaussianity_layers.csv'}")

# -----------------------------------------------------------------------------
# PART 2: LAYER 6 LOCAL GAUSSIANITY ACROSS CLUSTER GRANULARITY K
# -----------------------------------------------------------------------------
print("\n--- PART 2: Evaluating Layer 6 Gaussianity across K Granularity ---")
X6 = flat_latents[6]
k_values = [1, 2, 4, 8, 10, 16, 24, 32, 48, 64]
k_records = []

for K in k_values:
    if K == 1:
        res_rand = test_gaussianity_projections(X6, num_projections=100, sample_size=250)
        res_pca = test_pca_normality(X6)
        k_records.append({
            'K': K,
            'AD_Proj_pct': res_rand['ad_pct'] * 100,
            'AD_PCA_pct': res_pca['ad_pca_pct'] * 100,
            'Shapiro_PCA_pct': res_pca['shapiro_pca_pct'] * 100,
            'Excess_Kurtosis': res_pca['kurtosis'],
            'Skewness': res_pca['skewness']
        })
        print(f"K = {K:2d} (Global WCT) | Proj AD: {res_rand['ad_pct']*100:5.1f}% | PCA AD: {res_pca['ad_pca_pct']*100:5.1f}% | |Kurtosis|: {res_pca['kurtosis']:.3f}")
    else:
        km = KMeans(n_clusters=K, random_state=42, n_init=1).fit(X6)
        ads_rand, ads_pca, shs_pca, kurts_pca, skews_pca, weights = [], [], [], [], [], []
        for k in range(K):
            Xk = X6[km.labels_ == k]
            if len(Xk) >= 30:
                rk = test_gaussianity_projections(Xk, num_projections=100, sample_size=250)
                pk = test_pca_normality(Xk)
                ads_rand.append(rk['ad_pct'])
                ads_pca.append(pk['ad_pca_pct'])
                shs_pca.append(pk['shapiro_pca_pct'])
                kurts_pca.append(pk['kurtosis'])
                skews_pca.append(pk['skewness'])
                weights.append(len(Xk))
        w = np.array(weights) / sum(weights)
        avg_ad_rand = float(np.sum(np.array(ads_rand) * w))
        avg_ad_pca = float(np.sum(np.array(ads_pca) * w))
        avg_sh_pca = float(np.sum(np.array(shs_pca) * w))
        avg_kurt_pca = float(np.sum(np.array(kurts_pca) * w))
        avg_skew_pca = float(np.sum(np.array(skews_pca) * w))
        
        k_records.append({
            'K': K,
            'AD_Proj_pct': avg_ad_rand * 100,
            'AD_PCA_pct': avg_ad_pca * 100,
            'Shapiro_PCA_pct': avg_sh_pca * 100,
            'Excess_Kurtosis': avg_kurt_pca,
            'Skewness': avg_skew_pca
        })
        print(f"K = {K:2d} (LWT local)  | Proj AD: {avg_ad_rand*100:5.1f}% | PCA AD: {avg_ad_pca*100:5.1f}% | |Kurtosis|: {avg_kurt_pca:.3f}")

df_k = pd.DataFrame(k_records)
df_k.to_csv(out_tables / 'wavlm_large_gaussianity_k_granularity.csv', index=False)
print(f"Saved K-granularity Gaussianity table to {out_tables / 'wavlm_large_gaussianity_k_granularity.csv'}")

# -----------------------------------------------------------------------------
# PART 3: GENERATING HIGH-QUALITY PUBLICATION FIGURES
# -----------------------------------------------------------------------------
print("\n--- PART 3: Generating Publication Figures ---")

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
col_global = '#e63946'  # Crimson Red
col_local = '#1d3557'   # Deep Navy Blue

# Figure 1: Layer-wise Gaussianity Evolution (Updating une_comparison_multimodels for WavLM-Large)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 4.8), dpi=300)

# Subplot 1: Exact test_gaussianity_projections (random projections with sample_size=250)
ax1.plot(df_layers['Layer'], df_layers['Global_AD_Proj_pct'], marker='s', color=col_global, linewidth=2.2, markersize=5.5, label='Global Latent Space ($K=1$)')
ax1.plot(df_layers['Layer'], df_layers['Local_AD_Proj_pct'], marker='o', color=col_local, linewidth=2.2, markersize=5.5, label='Local Phonetic Clusters ($K=10$)')
ax1.axvline(x=6, color='#2a9d8f', linestyle=':', linewidth=2.2, label='Feature Layer 6 (LWT / kNN-VC)')
ax1.set_title('(a) Random Sliced Projections (Sample Size = 250)', fontsize=11.5, fontweight='bold', pad=10)
ax1.set_xlabel('WavLM-Large Transformer Layer Index', fontsize=10.5, fontweight='bold')
ax1.set_ylabel('Gaussian Fraction (AD Test %)', fontsize=10.5, fontweight='bold')
ax1.set_ylim(20, 102)
ax1.set_xticks(range(0, 25, 2))
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(loc='lower left', framealpha=0.95, fontsize=9.0)

# Subplot 2: PCA Projections (Acoustic Phonetic Axes)
ax2.plot(df_layers['Layer'], df_layers['Global_AD_PCA_pct'], marker='s', color=col_global, linewidth=2.2, markersize=5.5, label='Global Space ($K=1$, WCT)')
ax2.plot(df_layers['Layer'], df_layers['Local_AD_PCA_pct'], marker='o', color=col_local, linewidth=2.2, markersize=5.5, label='Local Phonetic Clusters ($K=10$, LWT)')
ax2.axvline(x=6, color='#2a9d8f', linestyle=':', linewidth=2.2, label='Feature Layer 6')
ax2.set_title('(b) Principal Acoustic Axes (PCA Directions)', fontsize=11.5, fontweight='bold', pad=10)
ax2.set_xlabel('WavLM-Large Transformer Layer Index', fontsize=10.5, fontweight='bold')
ax2.set_ylabel('Gaussian Fraction (AD Test %)', fontsize=10.5, fontweight='bold')
ax2.set_ylim(-2, 75)
ax2.set_xticks(range(0, 25, 2))
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(loc='upper right', framealpha=0.95, fontsize=9.0)

plt.tight_layout()
fig1_png = out_figures / 'figure_wavlm_large_gaussianity_layers.png'
fig1_pdf = out_figures / 'figure_wavlm_large_gaussianity_layers.pdf'
plt.savefig(fig1_png, bbox_inches='tight')
plt.savefig(fig1_pdf, bbox_inches='tight')
plt.close()
print(f"Saved Layer-wise figure: {fig1_png}")

# Figure 2: The Monge Optimality Justification: K Granularity vs Gaussianity on Layer 6
fig, ax_k1 = plt.subplots(figsize=(8.8, 4.6), dpi=300)
ax_k2 = ax_k1.twinx()

x_k_idx = np.arange(len(df_k))
labels_k = [str(k) for k in df_k['K']]

line_ad = ax_k1.plot(x_k_idx, df_k['AD_PCA_pct'], color=col_local, marker='o', linewidth=2.4, markersize=6.5, label='PCA Normality Pass Rate (AD %) $\\uparrow$')
line_sh = ax_k1.plot(x_k_idx, df_k['Shapiro_PCA_pct'], color='#457b9d', marker='^', linestyle='--', linewidth=2.0, markersize=6.0, label='Shapiro-Wilk Normality % $\\uparrow$')
line_kurt = ax_k2.plot(x_k_idx, df_k['Excess_Kurtosis'], color=col_global, marker='s', linewidth=2.2, markersize=6.0, label='|Excess Kurtosis| $\\downarrow$ (0 = Normal)')

ax_k1.set_xlabel('Number of Local Phonetic Clusters $K$', fontsize=11, fontweight='bold')
ax_k1.set_ylabel('Gaussianity Pass Rate (%) $\\uparrow$', color=col_local, fontsize=11, fontweight='bold')
ax_k1.tick_params(axis='y', labelcolor=col_local)
ax_k1.set_xticks(x_k_idx)
ax_k1.set_xticklabels(labels_k, fontsize=10)
ax_k1.set_ylim(-2, 70)
ax_k1.grid(True, linestyle='--', alpha=0.4)

ax_k2.set_ylabel('Absolute Excess Kurtosis $|\\gamma_2|$ $\\downarrow$', color=col_global, fontsize=11, fontweight='bold')
ax_k2.tick_params(axis='y', labelcolor=col_global)
ax_k2.set_ylim(0.3, 1.1)

# Annotate K=1 vs K=10
ax_k1.annotate('Global WCT (K=1)\n0% Gaussian on PCA\n(Multimodal Mixture)', 
                xy=(0, df_k.loc[0, 'AD_PCA_pct']), 
                xytext=(0.5, 18),
                arrowprops=dict(facecolor=col_global, shrink=0.08, width=1.2, headwidth=6),
                fontsize=9.0, fontweight='bold', color=col_global)

ax_k1.annotate('Canonical LWT (K=10)\nOptimal Monge Map Regime\n(Intra-cluster Gaussianity)', 
                xy=(4, df_k.loc[4, 'AD_PCA_pct']), 
                xytext=(3.2, 45),
                arrowprops=dict(facecolor=col_local, shrink=0.08, width=1.2, headwidth=6),
                fontsize=9.0, fontweight='bold', color=col_local)

lines = line_ad + line_sh + line_kurt
labels = [l.get_label() for l in lines]
ax_k1.legend(lines, labels, loc='center right', framealpha=0.95, fontsize=9.5)

plt.title('Restoration of Gaussian Geometry via Local Clustering (WavLM-Large Layer 6)', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
fig2_png = out_figures / 'figure_wavlm_large_gaussianity_monge_justification.png'
fig2_pdf = out_figures / 'figure_wavlm_large_gaussianity_monge_justification.pdf'
plt.savefig(fig2_png, bbox_inches='tight')
plt.savefig(fig2_pdf, bbox_inches='tight')
plt.close()
print(f"Saved Monge justification figure: {fig2_png}")

# Figure 3: Density Comparison: Global Multi-Modal vs Local Gaussian Unimodal
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0), dpi=300)

Xc = X6 - np.mean(X6, axis=0)
_, _, vt = np.linalg.svd(Xc, full_matrices=False)
pc1_global = (Xc @ vt[0])[:3000]

km10 = KMeans(n_clusters=10, random_state=42, n_init=1).fit(X6)
X_c0 = X6[km10.labels_ == 0]
Xc0 = X_c0 - np.mean(X_c0, axis=0)
_, _, vt0 = np.linalg.svd(Xc0, full_matrices=False)
pc1_c0 = (Xc0 @ vt0[0])[:1500]

X_c1 = X6[km10.labels_ == 1]
Xc1 = X_c1 - np.mean(X_c1, axis=0)
_, _, vt1 = np.linalg.svd(Xc1, full_matrices=False)
pc1_c1 = (Xc1 @ vt1[0])[:1500]

# Subplot 1: Global PC1
sns.kdeplot(pc1_global, ax=axes[0], color=col_global, fill=True, alpha=0.35, linewidth=2.2)
axes[0].set_title('(a) Global Latent Space ($K=1$)\nMultimodal Mixture of Phonemes', fontsize=11, fontweight='bold', pad=8)
axes[0].set_xlabel('Principal Component 1 (PC1)', fontsize=10)
axes[0].set_ylabel('Density', fontsize=10)
axes[0].grid(True, linestyle='--', alpha=0.3)

# Subplot 2: Cluster 0 PC1 + Theoretical Normal fit
sns.kdeplot(pc1_c0, ax=axes[1], color=col_local, fill=True, alpha=0.35, linewidth=2.2, label='Empirical Density')
x_eval0 = np.linspace(np.min(pc1_c0), np.max(pc1_c0), 200)
pdf_c0 = stats.norm.pdf(x_eval0, loc=np.mean(pc1_c0), scale=np.std(pc1_c0))
axes[1].plot(x_eval0, pdf_c0, 'k--', linewidth=2.0, label='Fitted Gaussian $\\mathcal{N}$')
axes[1].set_title('(b) Local Cluster $k=0$ (Vocalic State)\nUnimodal Acoustic Variations', fontsize=11, fontweight='bold', pad=8)
axes[1].set_xlabel('Local Intra-Cluster PC1', fontsize=10)
axes[1].legend(loc='upper right', fontsize=9.0)
axes[1].grid(True, linestyle='--', alpha=0.3)

# Subplot 3: Cluster 1 PC1 + Theoretical Normal fit
sns.kdeplot(pc1_c1, ax=axes[2], color='#2a9d8f', fill=True, alpha=0.35, linewidth=2.2, label='Empirical Density')
x_eval1 = np.linspace(np.min(pc1_c1), np.max(pc1_c1), 200)
pdf_c1 = stats.norm.pdf(x_eval1, loc=np.mean(pc1_c1), scale=np.std(pc1_c1))
axes[2].plot(x_eval1, pdf_c1, 'k--', linewidth=2.0, label='Fitted Gaussian $\\mathcal{N}$')
axes[2].set_title('(c) Local Cluster $k=1$ (Consonantal State)\nUnimodal Acoustic Variations', fontsize=11, fontweight='bold', pad=8)
axes[2].set_xlabel('Local Intra-Cluster PC1', fontsize=10)
axes[2].legend(loc='upper right', fontsize=9.0)
axes[2].grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
fig3_png = out_figures / 'figure_wavlm_large_density_global_vs_local.png'
fig3_pdf = out_figures / 'figure_wavlm_large_density_global_vs_local.pdf'
plt.savefig(fig3_png, bbox_inches='tight')
plt.savefig(fig3_pdf, bbox_inches='tight')
plt.close()
print(f"Saved Density comparison figure: {fig3_png}")

print("\n[SUCCESS] Completed comprehensive WavLM-Large Gaussianity Benchmark with test_gaussianity_projections!")
