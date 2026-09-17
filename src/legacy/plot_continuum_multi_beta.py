import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def plot_multi_beta_continuum():
    csv_path = Path("/local_scratch/ssadok/un_projet_audio/continuum_2d_grid_6spk.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return
        
    df = pd.read_csv(csv_path)
    
    # K order along the continuum
    K_order = [1, 3, 8, 25, 80, 250, 9999]
    K_display_labels = [
        "Classic WCT\n(K = 1)",
        "Soft WCT\n(K = 3)",
        "Soft WCT\n(K = 8)",
        "Soft WCT*\n(K = 25)",
        "Soft WCT\n(K = 80)",
        "Soft WCT\n(K = 250)",
        "1-NN (kNN-VC)\n(K = Nt, β→∞)"
    ]
    
    # 3 Trajectories of Beta: 20, 100, 1000
    beta_list = [20.0, 100.0, 1000.0]
    beta_styles = {
        20.0: {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o', 'label': 'Trajectory β = 20* (Sweet Spot)'},
        100.0: {'color': '#ff7f0e', 'linestyle': '-.', 'marker': '^', 'label': 'Trajectory β = 100 (Sharp)'},
        1000.0: {'color': '#8c564b', 'linestyle': ':', 'marker': 's', 'label': 'Trajectory β = 1000 (Hard limit)'}
    }
    
    # Compute mean across 6 speakers for each (Beta, K)
    mean_df = df.groupby(['Beta', 'K'], sort=False).mean(numeric_only=True)
    std_df = df.groupby(['Beta', 'K'], sort=False).std(numeric_only=True)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'axes.edgecolor': '#2b2b2b',
        'axes.linewidth': 1.6,
        'grid.color': '#e0e0e0',
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
        'xtick.major.size': 7,
        'ytick.major.size': 7,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'font.size': 13
    })

    # ULTRA-WIDE HORIZONTAL LAYOUT (22.5 x 5.8 inches)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22.5, 5.8), dpi=300)
    x_indices = np.arange(len(K_order))

    # =========================================================================
    # Panel (a): Optimal Trajectory (Beta = 20) with Speaker Variance (±1σ)
    # =========================================================================
    sims_b20 = [mean_df.loc[(20.0, k), 'Sim'] for k in K_order]
    cers_b20 = [mean_df.loc[(20.0, k), 'CER'] for k in K_order]
    sim_stds_b20 = [std_df.loc[(20.0, k), 'Sim'] for k in K_order]
    cer_stds_b20 = [std_df.loc[(20.0, k), 'CER'] for k in K_order]

    color_sim = '#1f77b4'
    color_cer = '#d62728'

    l1 = ax1.plot(x_indices, sims_b20, color=color_sim, marker='o', markersize=10.5, 
                  linewidth=3.4, label='Speaker Cosine Sim (↑)', zorder=4)
    ax1.fill_between(x_indices, np.array(sims_b20) - np.array(sim_stds_b20), 
                     np.array(sims_b20) + np.array(sim_stds_b20), color=color_sim, alpha=0.12, zorder=3)
    ax1.set_ylabel('Speaker Cosine Similarity (↑)', color=color_sim, fontsize=15, fontweight='bold', labelpad=12)
    ax1.tick_params(axis='y', labelcolor=color_sim, labelsize=13)
    ax1.set_ylim(0.58, 0.82)

    ax1_twin = ax1.twinx()
    l2 = ax1_twin.plot(x_indices, cers_b20, color=color_cer, marker='s', markersize=9.5, 
                       linewidth=3.2, linestyle='--', label='CER % (↓)', zorder=4)
    ax1_twin.fill_between(x_indices, np.array(cers_b20) - np.array(cer_stds_b20), 
                          np.array(cers_b20) + np.array(cer_stds_b20), color=color_cer, alpha=0.12, zorder=3)
    ax1_twin.set_ylabel('Character Error Rate (CER %) (↓)', color=color_cer, fontsize=15, fontweight='bold', labelpad=14)
    ax1_twin.tick_params(axis='y', labelcolor=color_cer, labelsize=13)
    ax1_twin.set_ylim(0.0, 2.4)

    # Sweet Spot at K=25
    best_k_idx = 3
    ax1.axvspan(best_k_idx - 0.35, best_k_idx + 0.35, color='#2ca02c', alpha=0.16, zorder=1)
    ax1.text(best_k_idx, 0.795, f'★ Optimal Sweet Spot\n(Lowest CER: {cers_b20[best_k_idx]:.2f}%)', 
             ha='center', va='center', fontsize=12, fontweight='bold', color='#155724',
             bbox=dict(boxstyle='round,pad=0.45', facecolor='#d4edda', edgecolor='#28a745', lw=1.3))

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(K_display_labels, fontsize=12, fontweight='medium')
    ax1.grid(True, axis='both')
    ax1.set_title('(a) Primary Alignment Continuum Trajectory (β = 20, 6 Speakers Average ± 1σ)', 
                  fontsize=15.5, fontweight='bold', pad=15, loc='left')

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left', bbox_to_anchor=(0.02, 0.05), 
               fontsize=12.5, framealpha=0.95, edgecolor='#c0c0c0')

    # =========================================================================
    # Panel (b): Multi-Beta Continuum Trajectories (Intelligibility vs. Similarity)
    # =========================================================================
    all_sims = []
    all_cers = []

    # Plot each beta trajectory
    for b in beta_list:
        cfg = beta_styles[b]
        b_sims = [mean_df.loc[(b, k), 'Sim'] for k in K_order]
        b_cers = [mean_df.loc[(b, k), 'CER'] for k in K_order]
        
        lw = 3.4 if b == 20.0 else 2.2
        ms = 9 if b == 20.0 else 7
        alpha = 1.0 if b == 20.0 else 0.85
        zorder = 5 if b == 20.0 else 3
        
        ax2.plot(b_sims, b_cers, color=cfg['color'], linestyle=cfg['linestyle'], 
                 marker=cfg['marker'], markersize=ms, linewidth=lw, alpha=alpha,
                 label=cfg['label'], zorder=zorder)
        
        # Annotate points on beta=20 trajectory
        if b == 20.0:
            for i, k in enumerate(K_order):
                if k in [3, 8, 80, 250]:
                    ax2.annotate(f'K={k}', (b_sims[i], b_cers[i]), textcoords="offset points", 
                                 xytext=(10, 8) if i%2==1 else (-15, -16),
                                 fontsize=11, color='#1f77b4', fontweight='bold')

    # Highlight common origin: Classic WCT (K=1)
    sim_k1 = mean_df.loc[(20.0, 1), 'Sim']
    cer_k1 = mean_df.loc[(20.0, 1), 'CER']
    ax2.scatter(sim_k1, cer_k1, color='#333333', s=200, marker='o', edgecolors='white', lw=2.0, 
                zorder=7, label='Classic WCT (K=1, Common Origin)')
    ax2.annotate('Classic WCT (K=1)\n(Common Origin)', (sim_k1, cer_k1), textcoords="offset points", 
                 xytext=(-15, 14), ha='center', fontsize=12, color='#222222', fontweight='bold')

    # Highlight common limit: 1-NN (K=Nt)
    sim_1nn = mean_df.loc[(20.0, 9999), 'Sim']
    cer_1nn = mean_df.loc[(20.0, 9999), 'CER']
    ax2.scatter(sim_1nn, cer_1nn, color='#d62728', s=220, marker='s', edgecolors='black', lw=2.0, 
                zorder=7, label='1-NN (K=Nt, Asymptotic Limit)')
    ax2.annotate('1-NN (kNN-VC)\n(Asymptotic Limit)', (sim_1nn, cer_1nn), textcoords="offset points", 
                 xytext=(-40, 14), fontsize=12, color='#d62728', fontweight='bold')

    # Highlight Optimal Sweet Spot (K=25, beta=20)
    sim_opt = mean_df.loc[(20.0, 25), 'Sim']
    cer_opt = mean_df.loc[(20.0, 25), 'CER']
    ax2.scatter(sim_opt, cer_opt, color='#28a745', s=450, marker='*', edgecolors='black', lw=2.2, 
                zorder=8, label='Optimal Sweet Spot (K=25, β=20)*')
    ax2.annotate(f'★ Proposed Sweet Spot\n(K=25, β=20 | CER: {cer_opt:.2f}%)', (sim_opt, cer_opt), 
                 textcoords="offset points", xytext=(0, -44), ha='center', fontsize=12, color='#155724', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#28a745', lw=1.8))

    ax2.set_xlabel('Speaker Cosine Similarity (↑)', fontsize=15, fontweight='bold', labelpad=12)
    ax2.set_ylabel('Character Error Rate (CER %) (↓)', fontsize=15, fontweight='bold', labelpad=12)
    ax2.tick_params(axis='both', labelsize=13)
    ax2.set_xlim(0.615, 0.745)
    ax2.set_ylim(0.40, 1.80)
    ax2.grid(True)
    ax2.set_title('(b) 2D Alignment Continuum: Trajectories for β ∈ {20, 100, 1000}', 
                  fontsize=15.5, fontweight='bold', pad=15, loc='left')
    ax2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.96), fontsize=11, framealpha=0.95, edgecolor='#c0c0c0')

    plt.tight_layout()

    out_dir = Path("/local_scratch/ssadok/un_projet_audio/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "figure_continuum_multi_beta_6spk.png"
    pdf_path = out_dir / "figure_continuum_multi_beta_6spk.pdf"
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved multi-beta continuum figure to:\n  {png_path}\n  {pdf_path}")
    plt.close(fig)

if __name__ == '__main__':
    plot_multi_beta_continuum()
