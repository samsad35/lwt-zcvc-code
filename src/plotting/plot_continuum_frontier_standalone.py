import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def plot_standalone_frontier():
    csv_path = Path("/local_scratch/ssadok/un_projet_audio/output/theorem_continuum_6spk.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Filter out Step 6 (K=Nt, beta=20 which is over-smoothed over 4000 frames)
    df = df[df['Step_order'] != 6].copy()
    
    steps_order = [1, 2, 3, 4, 5, 7, 8, 9]
    step_names = [
        "Classic WCT (K=1)",
        "Soft WCT (K=3)",
        "Soft WCT (K=8)",
        "Soft WCT (K=25)",
        "Soft WCT (K=80)",
        "K=Nt (β=40)",
        "K=Nt (β=80)",
        "1-NN (kNN-VC)"
    ]
    
    agg = df.groupby('Step_order', sort=True).agg({
        'Sim': ['mean', 'std'],
        'CER': ['mean', 'std']
    }).reindex(steps_order)
    
    sims = agg[('Sim', 'mean')].values
    sim_stds = agg[('Sim', 'std')].values
    cers = agg[('CER', 'mean')].values
    cer_stds = agg[('CER', 'std')].values
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'axes.edgecolor': '#2b2b2b',
        'axes.linewidth': 1.6,
        'grid.color': '#e2e2e2',
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
        'xtick.major.size': 7,
        'ytick.major.size': 7,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'font.size': 13
    })

    # GPU inference latencies in milliseconds per utterance
    latencies_ms = np.array([0.13, 0.30, 0.60, 2.06, 5.06, 0.27, 0.27, 0.16])
    # Bubble area scale: base 140 + 70 * latency
    scatter_sizes = 140 + 70 * latencies_ms

    fig, ax = plt.subplots(figsize=(11.5, 7.0), dpi=300)

    # =========================================================================
    # Phase 1: Spatial Localisation (K=1 to K=80 at beta=20)
    # =========================================================================
    p1_sims = sims[:5]
    p1_cers = cers[:5]
    ax.plot(p1_sims, p1_cers, color='#1f77b4', linestyle='-', lw=3.2, zorder=3,
            label=r'Phase 1: Spatial Localisation ($K \to 80$, $\beta=20$)')
    ax.scatter(p1_sims, p1_cers, color='#1f77b4', s=scatter_sizes[:5], zorder=5, edgecolors='black', lw=1.5)

    # =========================================================================
    # Phase 2: Temperature Hardening (Connecting from K=80 to K=Nt as beta -> inf)
    # =========================================================================
    p2_sims = sims[4:]
    p2_cers = cers[4:]
    p2_sizes = scatter_sizes[4:]
    ax.plot(p2_sims, p2_cers, color='#d62728', linestyle='--', lw=3.2, zorder=3,
            label=r'Phase 2: Temperature Hardening ($K=N_t$, $\beta \to \infty$)')
    ax.scatter(p2_sims, p2_cers, color='#d62728', s=p2_sizes, marker='s', zorder=5, edgecolors='black', lw=1.5)

    # Common Origin: Classic WCT (K=1)
    ax.scatter(sims[0], cers[0], color='#333333', s=scatter_sizes[0] + 40, marker='o', edgecolors='white', lw=2.2, 
               zorder=7, label='Classic WCT (K=1, Origin)')
    ax.annotate('Classic WCT (K=1)\n(0.13 ms)', (sims[0], cers[0]), textcoords="offset points", 
                xytext=(15, 12), fontsize=11.5, color='#222222', fontweight='bold')

    # Sweet Spot: Soft WCT (K=25)
    ax.scatter(sims[3], cers[3], color='#28a745', s=scatter_sizes[3] + 280, marker='*', edgecolors='black', lw=2.2, 
               zorder=8, label='Optimal Sweet Spot (K=25)*')
    ax.annotate(f'★ Proposed Sweet Spot\n(K=25, 2.06 ms | CER: {cers[3]:.2f}%)', (sims[3], cers[3]), 
                textcoords="offset points", xytext=(0, -44), ha='center', fontsize=12, color='#155724', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#28a745', lw=1.8))

    # Intermediate Points Annotations
    ax.annotate('K=8 (0.6 ms)', (sims[2], cers[2]), textcoords="offset points", xytext=(-26, -18), fontsize=10.5, color='#1f77b4', fontweight='bold')
    ax.annotate('K=80 (5.1 ms)', (sims[4], cers[4]), textcoords="offset points", xytext=(-16, 18), fontsize=10.5, color='#1f77b4', fontweight='bold')
    ax.annotate(r'$K=N_t$ (β=40, 0.3 ms)', (sims[5], cers[5]), textcoords="offset points", xytext=(0, -22), ha='center', fontsize=10.5, color='#d62728', fontweight='bold')
    ax.annotate('β=80 (0.3 ms)', (sims[6], cers[6]), textcoords="offset points", xytext=(14, -6), fontsize=10.5, color='#d62728', fontweight='bold')

    # Final Endpoint: 1-NN (kNN-VC)
    ax.scatter(sims[-1], cers[-1], color='#dc3545', s=scatter_sizes[-1] + 50, marker='s', edgecolors='black', lw=2.2, 
               zorder=7, label=r'1-NN (kNN-VC Limit, $\beta \to \infty$)')
    ax.annotate(r'1-NN (kNN-VC)' + '\n(0.16 ms)', (sims[-1], cers[-1]), textcoords="offset points", 
                xytext=(-32, 14), fontsize=12, color='#dc3545', fontweight='bold')

    ax.set_xlabel('Speaker Cosine Similarity (↑)', fontsize=15, fontweight='bold', labelpad=12)
    ax.set_ylabel('Character Error Rate (CER %) (↓)', fontsize=15, fontweight='bold', labelpad=12)
    ax.tick_params(axis='both', labelsize=13)
    ax.set_xlim(0.61, 0.745)
    ax.set_ylim(0.35, 1.85)
    ax.grid(True)
    ax.set_title('Intelligibility vs. Similarity Frontier: The Alignment Continuum\n(Scatter Size $\\propto$ GPU Inference Latency in ms)', 
                 fontsize=14.5, fontweight='bold', pad=16, loc='center')
    
    # Method Legend in Upper Left
    leg1 = ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.97), fontsize=11.5, framealpha=0.95, edgecolor='#c0c0c0')
    ax.add_artist(leg1)

    # Secondary Legend for Scatter Bubble Sizes in Lower Left
    sample_times = [0.2, 2.0, 5.0]
    sample_sizes = 140 + 70 * np.array(sample_times)
    legend_handles = [plt.scatter([], [], s=sz, color='#666666', alpha=0.6, edgecolors='black', lw=1.2) for sz in sample_sizes]
    legend_labels = [f'{t} ms' for t in sample_times]
    ax.legend(legend_handles, legend_labels, loc='lower left', bbox_to_anchor=(0.02, 0.05),
              title=r'$\bf{GPU\ Latency\ (ms)}$', title_fontsize=11.5, fontsize=10.5,
              framealpha=0.95, edgecolor='#c0c0c0', labelspacing=1.1, borderpad=0.8)

    plt.tight_layout()

    out_dir = Path("/local_scratch/ssadok/un_projet_audio/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "figure_continuum_frontier_standalone.png"
    pdf_path = out_dir / "figure_continuum_frontier_standalone.pdf"
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved standalone frontier figure to:\n  {png_path}\n  {pdf_path}")
    plt.close(fig)

if __name__ == '__main__':
    plot_standalone_frontier()
