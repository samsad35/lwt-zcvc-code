import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from pathlib import Path
import warnings
import sys
import torchaudio.functional as F

warnings.filterwarnings("ignore")

device = "cpu"

sys.path.append(str(Path(__file__).parent.parent))
from une import LibriSpeech

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
    return h_x

def plot_ellipse(ax, mean, cov, color, label=None):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:,order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    # 2 standard deviations
    width, height = 2 * 2 * np.sqrt(vals)
    ellip = Ellipse(xy=mean, width=width, height=height, angle=theta, 
                    edgecolor=color, fc='None', lw=2, label=label, zorder=10)
    ax.add_patch(ellip)

def main():
    print("Loading models and dataset...")
    wavlm_large = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=device).eval()
    
    dataset_path = Path("/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean")
    librispeech = LibriSpeech(root=dataset_path, ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    
    tgt_spk = "121"
    tgt_paths = df_all[df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")]['path'].tolist()[:10]
    
    print("Extracting features...")
    X_list = [extract_features(p, wavlm_large) for p in tgt_paths]
    X = np.concatenate(X_list, axis=0) # shape (N, 1024)
    
    # ---------------------------------------------------------
    # FIGURE 1: Geometric Intuition (Global vs GMM)
    # ---------------------------------------------------------
    print("Generating Figure 1 (Covariances)...")
    import seaborn as sns
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "0.15", "axes.linewidth": 1.25})
    
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10), dpi=300)
    
    # Panel A: Global WCT
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], ax=ax1, s=15, alpha=0.5, color='#95a5a6', edgecolor='none')
    global_mean = np.mean(X_2d, axis=0)
    global_cov = np.cov(X_2d, rowvar=False)
    plot_ellipse(ax1, global_mean, global_cov, '#e74c3c')
    ax1.set_title("Panel A: Classic WCT\n(Single Global Covariance)", fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel("Principal Component 1", fontsize=12, fontweight='semibold')
    ax1.set_ylabel("Principal Component 2", fontsize=12, fontweight='semibold')
    
    # Panel B: GMM-WCT (K=5)
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(X_2d)
    
    palette = sns.color_palette("husl", 5)
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=labels, palette=palette, ax=ax2, s=15, alpha=0.6, edgecolor='none', legend=False)
    
    for k in range(5):
        X_k = X_2d[labels == k]
        mean_k = np.mean(X_k, axis=0)
        cov_k = np.cov(X_k, rowvar=False)
        plot_ellipse(ax2, mean_k, cov_k, '#2c3e50')
        
    ax2.set_title("Panel B: Soft GMM-WCT\n(Piecewise Phonetic Covariances, K=5)", fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel("Principal Component 1", fontsize=12, fontweight='semibold')
    ax2.set_ylabel("Principal Component 2", fontsize=12, fontweight='semibold')
    
    plt.tight_layout()
    fig1_path = str(Path(__file__).parent.parent / "figure1_covariances.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ---------------------------------------------------------
    # FIGURE 2: Topological Purity (t-SNE)
    # ---------------------------------------------------------
    print("Generating Figure 2 (t-SNE Purity)...")
    src_paths = df_all[~df_all['path'].astype(str).str.contains(f"/{tgt_spk}/")]['path'].tolist()[:3]
    X_src = np.concatenate([extract_features(p, wavlm_large) for p in src_paths], axis=0)
    
    # Pseudo-phonetic labels using KMeans (7 phonetic classes)
    phonetic_labels = KMeans(n_clusters=7, random_state=42).fit_predict(X_src)
    
    # Simulate WCT conversion on the source (Global WCT for simplicity, or we can just show the transformed space)
    mu_s = np.mean(X_src, axis=0)
    cov_s = np.cov(X_src, rowvar=False) + np.eye(X_src.shape[1])*1e-5
    inv_s = np.linalg.inv(np.linalg.cholesky(cov_s))
    
    mu_t = np.mean(X, axis=0)
    cov_t = np.cov(X, rowvar=False) + np.eye(X.shape[1])*1e-5
    chol_t = np.linalg.cholesky(cov_t)
    
    X_w = np.matmul(X_src - mu_s, inv_s.T)
    X_conv = np.matmul(X_w, chol_t.T) + mu_t
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_conv)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=phonetic_labels, cmap='tab10', s=10, alpha=0.8)
    plt.title("t-SNE of Converted Latent Space (Colored by Phonetic Class)")
    plt.colorbar(scatter, label='Phonetic Cluster ID')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    fig2_path = str(Path(__file__).parent.parent / "figure2_tsne.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Done! Figures saved.")

if __name__ == "__main__":
    main()
