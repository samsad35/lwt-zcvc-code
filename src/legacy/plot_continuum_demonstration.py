import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def plot_elegant_continuum():
    csv_path = Path("/local_scratch/ssadok/un_projet_audio/continuum_multires_4spk.csv")
    df = pd.read_csv(csv_path)
    
    methods_order = [
        "Classic WCT (K=1)",
        "Soft Local WCT (K=5)",
        "Soft Local WCT (K=25)",
        "Soft Local WCT (K=100)",
        "1-NN (K=Nt)",
        "4-NN (kNN-VC)"
    ]
    
    display_names = [
        "Classic WCT\n(K = 1)",
        "Soft WCT\n(K = 5)",
        "Soft WCT*\n(K = 25)",
        "Soft WCT\n(K = 100)",
        "1-NN (kNN-VC)\n(K = Nt, β→∞)",
        "4-NN\n(kNN-VC)"
    ]
    
    agg = df.groupby("Method", sort=False).mean(numeric_only=True).reindex(methods_order)
    std_agg = df.groupby("Method", sort=False).std(numeric_only=True).reindex(methods_order)
    
    sims = agg["Sim"].values
    cers = agg["CER"].values
    sim_stds = std_agg["Sim"].values
    cer_stds = std_agg["CER"].values
    
    # Matplotlib styling for top-tier publication
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

    color_sim = '#1f77b4' # Rich corporate blue
    color_cer = '#d62728' # Deep crimson

    # ULTRA-WIDE HORIZONTAL LAYOUT (21 x 5.8 inches)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 5.8), dpi=300)
    x_indices = np.arange(len(methods_order))

    # =========================================================================
    # Panel (a): Multi-Speaker Alignment Continuum Trajectory (Dual Axis)
    # =========================================================================
    l1 = ax1.plot(x_indices, sims, color=color_sim, marker='o', markersize=11, 
                  linewidth=3.4, label='Speaker Cosine Sim (↑)', zorder=4)
    ax1.fill_between(x_indices, sims - sim_stds, sims + sim_stds, color=color_sim, alpha=0.12, zorder=3)
    ax1.set_ylabel('Speaker Cosine Similarity (↑)', color=color_sim, fontsize=15, fontweight='bold', labelpad=12)
    ax1.tick_params(axis='y', labelcolor=color_sim, labelsize=13)
    ax1.set_ylim(0.55, 0.77)

    ax1_twin = ax1.twinx()
    l2 = ax1_twin.plot(x_indices, cers, color=color_cer, marker='s', markersize=10, 
                       linewidth=3.2, linestyle='--', label='CER % (↓)', zorder=4)
    ax1_twin.fill_between(x_indices, cers - cer_stds, cers + cer_stds, color=color_cer, alpha=0.12, zorder=3)
    ax1_twin.set_ylabel('Character Error Rate (CER %) (↓)', color=color_cer, fontsize=15, fontweight='bold', labelpad=14)
    ax1_twin.tick_params(axis='y', labelcolor=color_cer, labelsize=13)
    ax1_twin.set_ylim(0.0, 2.5)

    # Highlight Sweet Spot at K=25
    ax1.axvspan(1.65, 2.35, color='#2ca02c', alpha=0.16, zorder=1)
    ax1.text(2.0, 0.715, '★ Optimal Sweet Spot\n(Lowest CER: 0.49%)', ha='center', va='center', 
             fontsize=12, fontweight='bold', color='#155724',
             bbox=dict(boxstyle='round,pad=0.45', facecolor='#d4edda', edgecolor='#28a745', lw=1.3))

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(display_names, fontsize=12, fontweight='medium')
    ax1.grid(True, axis='both')
    ax1.set_title('(a) Alignment Continuum Trajectory (4 Balanced Speakers Average ± 1σ)', 
                  fontsize=15.5, fontweight='bold', pad=15, loc='left')

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left', bbox_to_anchor=(0.02, 0.05), 
               fontsize=12.5, framealpha=0.95, edgecolor='#c0c0c0')

    # =========================================================================
    # Panel (b): Pareto Frontier (CER vs Speaker Sim)
    # =========================================================================
    # Trajectory path along continuum
    ax2.plot(sims, cers, color='#888888', linestyle=':', lw=2.4, zorder=2)

    # Classic WCT (K=1)
    ax2.scatter(sims[0], cers[0], color='#6c757d', s=160, marker='o', edgecolors='black', lw=1.6, 
                label='Classic WCT (K=1)', zorder=5)
    ax2.annotate('Classic WCT\n(K=1)', (sims[0], cers[0]), textcoords="offset points", 
                 xytext=(-15, 14), ha='center', fontsize=12, color='#333333', fontweight='semibold')

    # Soft WCT (K=5, 100)
    ax2.scatter([sims[1], sims[3]], [cers[1], cers[3]], color='#1f77b4', s=160, marker='D', 
                edgecolors='black', lw=1.6, zorder=5, label='Soft WCT (K=5, 100)')
    ax2.annotate('K=5', (sims[1], cers[1]), textcoords="offset points", xytext=(-16, -16), fontsize=12, color='#1f77b4', fontweight='bold')
    ax2.annotate('K=100', (sims[3], cers[3]), textcoords="offset points", xytext=(14, 8), fontsize=12, color='#1f77b4', fontweight='bold')

    # Sweet Spot (K=25)
    ax2.scatter(sims[2], cers[2], color='#28a745', s=420, marker='*', edgecolors='black', lw=2.0, 
                label='Proposed Soft WCT (K=25)*', zorder=6)
    ax2.annotate(f'★ Proposed (K=25)\nLowest CER: {cers[2]:.2f}%', (sims[2], cers[2]), textcoords="offset points", 
                 xytext=(0, -44), ha='center', fontsize=12, color='#155724', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#28a745', lw=1.8))

    # Instance-based (1-NN & 4-NN)
    ax2.scatter(sims[4:], cers[4:], color='#dc3545', s=170, marker='s', edgecolors='black', lw=1.6, 
                label='kNN-VC (1-NN, 4-NN)', zorder=5)
    ax2.annotate('1-NN (K=Nt, β→∞)\n(Discontinuous jitter)', (sims[4], cers[4]), textcoords="offset points", 
                 xytext=(-35, 14), fontsize=12, color='#dc3545', fontweight='bold')
    ax2.annotate('4-NN (kNN-VC)', (sims[5], cers[5]), textcoords="offset points", xytext=(14, -6), fontsize=12, color='#dc3545', fontweight='bold')

    ax2.set_xlabel('Speaker Cosine Similarity (↑)', fontsize=15, fontweight='bold', labelpad=12)
    ax2.set_ylabel('Character Error Rate (CER %) (↓)', fontsize=15, fontweight='bold', labelpad=12)
    ax2.tick_params(axis='both', labelsize=13)
    ax2.set_xlim(0.615, 0.725)
    ax2.set_ylim(0.2, 2.0)
    ax2.grid(True)
    ax2.set_title('(b) Intelligibility vs. Similarity Frontier (4 Balanced Speakers Average)', 
                  fontsize=15.5, fontweight='bold', pad=15, loc='left')
    ax2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.95), fontsize=12, framealpha=0.95, edgecolor='#c0c0c0')

    plt.tight_layout()

    out_dir = Path("/local_scratch/ssadok/un_projet_audio/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "figure_continuum_demonstration.png"
    pdf_path = out_dir / "figure_continuum_demonstration.pdf"
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to:\n  {png_path}\n  {pdf_path}")
    plt.close(fig)

if __name__ == '__main__':
    plot_elegant_continuum()
