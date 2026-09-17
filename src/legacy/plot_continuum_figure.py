import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def plot_alignment_continuum():
    # Set academic font and styling
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'axes.edgecolor': '#2b2b2b',
        'axes.linewidth': 1.0,
        'grid.color': '#e5e5e5',
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'font.size': 9
    })

    # Data from full_continuum_table.csv
    models = [
        "Global\n(K=1)",
        "Soft\n(K=3)",
        "Soft*\n(K=5)",
        "Soft\n(K=8)",
        "1-NN\n(K=Nt)",
        "4-NN\n(kNN-VC)"
    ]
    x_indices = np.arange(len(models))
    sims = [0.5184, 0.5546, 0.5737, 0.5932, 0.6098, 0.6259]
    cers = [1.046,  1.649,  0.972,  1.168,  3.492,  3.305]

    # =========================================================================
    # VERSION 1: Vertical 1-Column Layout (ICASSP Column Width: 3.5 in)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 5.8), dpi=300)

    color_sim = '#1f77b4' # Deep Blue
    color_cer = '#d62728' # Coral / Crimson

    # --- Panel (a) ---
    l1 = ax1.plot(x_indices, sims, color=color_sim, marker='o', markersize=6.0, 
                  linewidth=2.0, label='Cosine Sim (↑)', zorder=4)
    ax1.set_ylabel('Cosine Sim (↑)', color=color_sim, fontsize=8.5, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_sim, labelsize=8)
    ax1.set_ylim(0.48, 0.66)

    ax1_twin = ax1.twinx()
    l2 = ax1_twin.plot(x_indices, cers, color=color_cer, marker='s', markersize=5.5, 
                       linewidth=1.8, linestyle='--', label='CER % (↓)', zorder=4)
    ax1_twin.set_ylabel('CER % (↓)', color=color_cer, fontsize=8.5, fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor=color_cer, labelsize=8)
    ax1_twin.set_ylim(0.4, 4.4)

    # Highlight Sweet Spot at K=5 (badge placed at bottom)
    ax1.axvspan(1.65, 2.35, color='#2ca02c', alpha=0.15, zorder=1)
    ax1.text(2.0, 0.495, '★ Sweet Spot', ha='center', va='bottom', fontsize=7.2, 
             fontweight='bold', color='#1e7e34',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#eafaf1', edgecolor='#2ca02c', lw=0.6))

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(models, fontsize=7.5)
    ax1.grid(True, axis='both')
    ax1.text(0.03, 0.95, '(a) Alignment Continuum Trajectory', transform=ax1.transAxes, 
             fontsize=8.5, fontweight='bold', va='top')

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.03, 0.88), 
               fontsize=7.2, framealpha=0.92, edgecolor='#d0d0d0')

    # --- Panel (b) ---
    arrow_x = [sims[0], sims[1], sims[2], sims[3], sims[4], sims[5]]
    arrow_y = [cers[0], cers[1], cers[2], cers[3], cers[4], cers[5]]
    ax2.plot(arrow_x, arrow_y, color='#a0a0a0', linestyle=':', lw=1.2, zorder=2)

    ax2.scatter(sims[0], cers[0], color='#7f7f7f', s=60, marker='o', edgecolors='black', lw=0.8, 
                label='Global WCT (K=1)', zorder=5)
    ax2.annotate('Global WCT\n(K=1)', (sims[0], cers[0]), textcoords="offset points", 
                 xytext=(8, -6), fontsize=6.8, color='#444444', fontweight='semibold')

    ax2.scatter([sims[1], sims[3]], [cers[1], cers[3]], color='#2b5c8f', s=60, marker='D', 
                edgecolors='black', lw=0.8, zorder=5, label='Soft WCT (K=3, 8)')
    ax2.annotate('K=3', (sims[1], cers[1]), textcoords="offset points", xytext=(-6, 7), fontsize=7.0, color='#2b5c8f', fontweight='bold')
    ax2.annotate('K=8', (sims[3], cers[3]), textcoords="offset points", xytext=(7, 2), fontsize=7.0, color='#2b5c8f', fontweight='bold')

    ax2.scatter(sims[2], cers[2], color='#2ca02c', s=130, marker='*', edgecolors='black', lw=0.9, 
                label='Proposed (K=5)*', zorder=6)
    ax2.annotate('★ Proposed (K=5)\nCER: 0.97%', (sims[2], cers[2]), textcoords="offset points", 
                 xytext=(0, -22), ha='center', fontsize=6.8, color='#1e7e34', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=0.8))

    ax2.scatter(sims[4:], cers[4:], color='#d62728', s=60, marker='s', edgecolors='black', lw=0.8, 
                label='kNN-VC (1-NN, 4-NN)', zorder=5)
    ax2.annotate('1-NN', (sims[4], cers[4]), textcoords="offset points", xytext=(-26, 4), fontsize=7.0, color='#d62728', fontweight='bold')
    ax2.annotate('4-NN (kNN-VC)', (sims[5], cers[5]), textcoords="offset points", xytext=(8, -3), fontsize=7.0, color='#d62728', fontweight='bold')

    ax2.set_xlabel('Speaker Cosine Similarity (↑)', fontsize=8.5, fontweight='bold')
    ax2.set_ylabel('CER % (↓, Lower is Better)', fontsize=8.5, fontweight='bold')
    ax2.set_xlim(0.49, 0.69)
    ax2.set_ylim(0.3, 4.4)
    ax2.grid(True)
    ax2.text(0.03, 0.95, '(b) Intelligibility vs. Similarity Frontier', transform=ax2.transAxes, 
             fontsize=8.5, fontweight='bold', va='top')
    ax2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.88), fontsize=6.5, framealpha=0.92, edgecolor='#d0d0d0')

    plt.tight_layout()

    out_dir = Path("/local_scratch/ssadok/un_projet_audio/output")
    out_dir.mkdir(exist_ok=True, parents=True)

    png_path = out_dir / "figure_continuum.png"
    pdf_path = out_dir / "figure_continuum.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # VERSION 2: Horizontal 2-Column Wide Layout (7.2 x 3.2 in)
    # =========================================================================
    fig_w, (ax1_w, ax2_w) = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=300)

    l1_w = ax1_w.plot(x_indices, sims, color=color_sim, marker='o', markersize=6.0, 
                      linewidth=2.0, label='Cosine Sim (↑)', zorder=4)
    ax1_w.set_ylabel('Cosine Sim (↑)', color=color_sim, fontsize=8.5, fontweight='bold')
    ax1_w.tick_params(axis='y', labelcolor=color_sim, labelsize=8)
    ax1_w.set_ylim(0.48, 0.66)

    ax1_w_twin = ax1_w.twinx()
    l2_w = ax1_w_twin.plot(x_indices, cers, color=color_cer, marker='s', markersize=5.5, 
                           linewidth=1.8, linestyle='--', label='CER % (↓)', zorder=4)
    ax1_w_twin.set_ylabel('CER % (↓)', color=color_cer, fontsize=8.5, fontweight='bold')
    ax1_w_twin.tick_params(axis='y', labelcolor=color_cer, labelsize=8)
    ax1_w_twin.set_ylim(0.4, 4.4)

    ax1_w.axvspan(1.65, 2.35, color='#2ca02c', alpha=0.15, zorder=1)
    ax1_w.text(2.0, 0.495, '★ Sweet Spot', ha='center', va='bottom', fontsize=7.2, 
               fontweight='bold', color='#1e7e34',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='#eafaf1', edgecolor='#2ca02c', lw=0.6))

    ax1_w.set_xticks(x_indices)
    ax1_w.set_xticklabels(models, fontsize=7.5)
    ax1_w.grid(True, axis='both')
    ax1_w.text(0.03, 0.95, '(a) Alignment Continuum Trajectory', transform=ax1_w.transAxes, 
               fontsize=8.5, fontweight='bold', va='top')
    lines_w = l1_w + l2_w
    labels_w = [l.get_label() for l in lines_w]
    ax1_w.legend(lines_w, labels_w, loc='upper left', bbox_to_anchor=(0.03, 0.88), 
                 fontsize=7.2, framealpha=0.92, edgecolor='#d0d0d0')

    # Right: Pareto
    ax2_w.plot(arrow_x, arrow_y, color='#a0a0a0', linestyle=':', lw=1.2, zorder=2)
    ax2_w.scatter(sims[0], cers[0], color='#7f7f7f', s=60, marker='o', edgecolors='black', lw=0.8, 
                  label='Global WCT (K=1)', zorder=5)
    ax2_w.annotate('Global WCT (K=1)', (sims[0], cers[0]), textcoords="offset points", 
                   xytext=(8, -8), fontsize=7.0, color='#444444', fontweight='semibold')

    ax2_w.scatter([sims[1], sims[3]], [cers[1], cers[3]], color='#2b5c8f', s=60, marker='D', 
                  edgecolors='black', lw=0.8, zorder=5, label='Soft WCT (K=3, 8)')
    ax2_w.annotate('K=3', (sims[1], cers[1]), textcoords="offset points", xytext=(-6, 7), fontsize=7.0, color='#2b5c8f', fontweight='bold')
    ax2_w.annotate('K=8', (sims[3], cers[3]), textcoords="offset points", xytext=(7, 2), fontsize=7.0, color='#2b5c8f', fontweight='bold')

    ax2_w.scatter(sims[2], cers[2], color='#2ca02c', s=130, marker='*', edgecolors='black', lw=0.9, 
                  label='Proposed (K=5)*', zorder=6)
    ax2_w.annotate('★ Proposed (K=5)\nCER: 0.97%', (sims[2], cers[2]), textcoords="offset points", 
                   xytext=(0, -22), ha='center', fontsize=7.0, color='#1e7e34', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=0.8))

    ax2_w.scatter(sims[4:], cers[4:], color='#d62728', s=60, marker='s', edgecolors='black', lw=0.8, 
                  label='kNN-VC (1-NN, 4-NN)', zorder=5)
    ax2_w.annotate('1-NN', (sims[4], cers[4]), textcoords="offset points", xytext=(-26, 4), fontsize=7.0, color='#d62728', fontweight='bold')
    ax2_w.annotate('4-NN (kNN-VC)', (sims[5], cers[5]), textcoords="offset points", xytext=(8, -3), fontsize=7.0, color='#d62728', fontweight='bold')

    ax2_w.set_xlabel('Speaker Cosine Similarity (↑)', fontsize=8.5, fontweight='bold')
    ax2_w.set_ylabel('CER % (↓, Lower is Better)', fontsize=8.5, fontweight='bold')
    ax2_w.set_xlim(0.49, 0.69)
    ax2_w.set_ylim(0.3, 4.4)
    ax2_w.grid(True)
    ax2_w.text(0.03, 0.95, '(b) Intelligibility vs. Similarity Frontier', transform=ax2_w.transAxes, 
               fontsize=8.5, fontweight='bold', va='top')
    ax2_w.legend(loc='upper left', bbox_to_anchor=(0.03, 0.88), fontsize=6.8, framealpha=0.92, edgecolor='#d0d0d0')

    plt.tight_layout()

    png_path_wide = out_dir / "figure_continuum_wide.png"
    pdf_path_wide = out_dir / "figure_continuum_wide.pdf"
    plt.savefig(png_path_wide, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path_wide, bbox_inches='tight')
    plt.close()
    
    print(f"Figures successfully generated in {out_dir}:")
    print(f"  - Vertical (1-col): figure_continuum.png, figure_continuum.pdf")
    print(f"  - Wide (2-col): figure_continuum_wide.png, figure_continuum_wide.pdf")

if __name__ == "__main__":
    plot_alignment_continuum()
