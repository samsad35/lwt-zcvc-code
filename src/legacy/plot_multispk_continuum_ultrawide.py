import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def plot_multispk_continuum():
    csv_path = Path("/local_scratch/ssadok/un_projet_audio/multispk_continuum_table.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist yet.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Method order along the continuum
    methods_order = [
        "Global WCT (K=1)",
        "Soft WCT (K=3)",
        "Soft WCT (K=5)",
        "Soft WCT (K=8)",
        "1-NN (K=Nt)",
        "4-NN (kNN-VC)"
    ]
    
    display_names = [
        "Global WCT\n(K = 1)",
        "Soft WCT\n(K = 3)",
        "Soft WCT*\n(K = 5)",
        "Soft WCT\n(K = 8)",
        "1-NN\n(K = Nt)",
        "4-NN\n(kNN-VC)"
    ]
    
    means = df.groupby("Method").mean(numeric_only=True).reindex(methods_order)
    stds = df.groupby("Method").std(numeric_only=True).reindex(methods_order)
    
    sims = means["Cosine Sim"].values
    cers = means["CER (%)"].values
    sim_stds = stds["Cosine Sim"].values
    cer_stds = stds["CER (%)"].values
    
    print("Multi-speaker aggregated values across 5 target speakers:")
    for m, s, c in zip(methods_order, sims, cers):
        print(f"  {m:20s}: Sim = {s:.4f}, CER = {c:.2f}%")

    out_dir = Path("/local_scratch/ssadok/un_projet_audio/output")
    out_dir.mkdir(exist_ok=True, parents=True)

    # Academic styling
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'axes.edgecolor': '#2b2b2b',
        'axes.linewidth': 1.6,
        'grid.color': '#e2e2e2',
        'grid.linestyle': '--',
        'grid.linewidth': 0.9,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.major.width': 1.4,
        'ytick.major.width': 1.4,
        'font.size': 13
    })

    color_sim = '#1f77b4' # Rich blue
    color_cer = '#d62728' # Deep crimson

    # =========================================================================
    # ULTRA-WIDE HORIZONTAL LAYOUT (20.5 x 5.8 inches)
    # Stretched in width for clarity and wide presentation
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20.5, 5.8), dpi=300)
    x_indices = np.arange(len(methods_order))

    # -------------------------------------------------------------------------
    # Panel (a): Multi-Speaker Continuum Trajectory (Dual Axis)
    # -------------------------------------------------------------------------
    l1 = ax1.plot(x_indices, sims, color=color_sim, marker='o', markersize=11, 
                  linewidth=3.2, label='Speaker Cosine Sim (↑)', zorder=4)
    ax1.fill_between(x_indices, sims - sim_stds, sims + sim_stds, color=color_sim, alpha=0.10, zorder=3)
    ax1.set_ylabel('Speaker Cosine Similarity (↑)', color=color_sim, fontsize=14.5, fontweight='bold', labelpad=12)
    ax1.tick_params(axis='y', labelcolor=color_sim, labelsize=12.5)
    ax1.set_ylim(0.51, 0.77)

    ax1_twin = ax1.twinx()
    l2 = ax1_twin.plot(x_indices, cers, color=color_cer, marker='s', markersize=10, 
                       linewidth=3.0, linestyle='--', label='CER % (↓)', zorder=4)
    ax1_twin.fill_between(x_indices, cers - cer_stds, cers + cer_stds, color=color_cer, alpha=0.10, zorder=3)
    ax1_twin.set_ylabel('Character Error Rate (CER %) (↓)', color=color_cer, fontsize=14.5, fontweight='bold', labelpad=14)
    ax1_twin.tick_params(axis='y', labelcolor=color_cer, labelsize=12.5)
    ax1_twin.set_ylim(0.5, 4.8)

    # Highlight Sweet Spot at K=5
    ax1.axvspan(1.65, 2.35, color='#2ca02c', alpha=0.15, zorder=1)
    ax1.text(2.0, 0.535, '★ Optimal Sweet Spot\n(Lowest CER: 1.37%)', ha='center', va='center', 
             fontsize=12.0, fontweight='bold', color='#155724',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#d4edda', edgecolor='#28a745', lw=1.2))

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(display_names, fontsize=12.5, fontweight='medium')
    ax1.grid(True, axis='both')
    ax1.set_title('(a) Alignment Continuum Trajectory (5 Target Speakers Average ± 1σ)', 
                  fontsize=15, fontweight='bold', pad=14, loc='left')

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.02, 0.95), 
               fontsize=12, framealpha=0.95, edgecolor='#c0c0c0')

    # -------------------------------------------------------------------------
    # Panel (b): Pareto Trade-off Frontier (CER vs Speaker Sim)
    # -------------------------------------------------------------------------
    # Trajectory path along continuum
    ax2.plot(sims, cers, color='#888888', linestyle=':', lw=2.4, zorder=2)

    # Global WCT (K=1)
    ax2.scatter(sims[0], cers[0], color='#6c757d', s=160, marker='o', edgecolors='black', lw=1.6, 
                label='Global WCT (K=1)', zorder=5)
    ax2.annotate('Global WCT\n(K=1)', (sims[0], cers[0]), textcoords="offset points", 
                 xytext=(14, -14), fontsize=12, color='#333333', fontweight='semibold')

    # Soft WCT points
    ax2.scatter(sims[1], cers[1], color='#1f77b4', s=160, marker='D', edgecolors='black', lw=1.6, 
                label='Soft WCT (K=3, 8)', zorder=5)
    ax2.annotate('K=3', (sims[1], cers[1]), textcoords="offset points", xytext=(-16, 12), fontsize=12.5, color='#1f77b4', fontweight='bold')
    
    ax2.scatter(sims[3], cers[3], color='#1f77b4', s=160, marker='D', edgecolors='black', lw=1.6, zorder=5)
    ax2.annotate('K=8', (sims[3], cers[3]), textcoords="offset points", xytext=(12, 10), fontsize=12.5, color='#1f77b4', fontweight='bold')

    # Sweet Spot (K=5)
    ax2.scatter(sims[2], cers[2], color='#28a745', s=380, marker='*', edgecolors='black', lw=2.0, 
                label='Proposed Soft WCT (K=5)*', zorder=6)
    ax2.annotate(f'★ Proposed (K=5)\nLowest CER: {cers[2]:.2f}%', (sims[2], cers[2]), textcoords="offset points", 
                 xytext=(-20, -42), ha='center', fontsize=12, color='#155724', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#28a745', lw=1.8))

    # Instance-based (1-NN & 4-NN)
    ax2.scatter(sims[4:], cers[4:], color='#dc3545', s=170, marker='s', edgecolors='black', lw=1.6, 
                label='kNN-VC (1-NN, 4-NN)', zorder=5)
    ax2.annotate('1-NN (K=Nt)', (sims[4], cers[4]), textcoords="offset points", xytext=(-40, 10), fontsize=12.5, color='#dc3545', fontweight='bold')
    ax2.annotate('4-NN (kNN-VC)', (sims[5], cers[5]), textcoords="offset points", xytext=(14, -6), fontsize=12.5, color='#dc3545', fontweight='bold')

    ax2.set_xlabel('Speaker Cosine Similarity (↑)', fontsize=14.5, fontweight='bold', labelpad=12)
    ax2.set_ylabel('Character Error Rate (CER %) (↓)', fontsize=14.5, fontweight='bold', labelpad=12)
    ax2.tick_params(axis='both', labelsize=12.5)
    ax2.set_xlim(0.585, 0.730)
    ax2.set_ylim(0.7, 4.4)
    ax2.grid(True)
    ax2.set_title('(b) Intelligibility vs. Similarity Frontier (5 Target Speakers Average)', 
                  fontsize=15, fontweight='bold', pad=14, loc='left')
    ax2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.95), fontsize=12, framealpha=0.95, edgecolor='#c0c0c0')

    plt.tight_layout()

    out_png = out_dir / "figure_continuum_ultrawide.png"
    out_pdf = out_dir / "figure_continuum_ultrawide.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved ultra-wide figures to:\n  {out_png}\n  {out_pdf}")

if __name__ == "__main__":
    plot_multispk_continuum()
