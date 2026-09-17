import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_alignment_continuum():
    # Set bold, clear academic styling with large readable fonts
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'axes.edgecolor': '#2b2b2b',
        'axes.linewidth': 1.6,
        'grid.color': '#e0e0e0',
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

    models = [
        "Global WCT\n(K = 1)",
        "Soft WCT\n(K = 3)",
        "Soft WCT*\n(K = 5)",
        "Soft WCT\n(K = 8)",
        "1-NN\n(K = Nt)",
        "4-NN\n(kNN-VC)"
    ]
    x_indices = np.arange(len(models))
    sims = [0.5184, 0.5546, 0.5737, 0.5932, 0.6098, 0.6259]
    cers = [1.046,  1.649,  0.972,  1.168,  3.492,  3.305]

    out_dir = Path("/local_scratch/ssadok/un_projet_audio/output")
    out_dir.mkdir(exist_ok=True, parents=True)

    color_sim = '#1f77b4' # Deep vibrant blue
    color_cer = '#d62728' # Crimson red

    # =========================================================================
    # VERSION 1: Large Wide Side-by-Side Figure (14 x 6.2 inches)
    # =========================================================================
    fig_w, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2), dpi=300)

    # -------------------------------------------------------------------------
    # Panel (a): Trajectory Across Continuum
    # -------------------------------------------------------------------------
    l1 = ax1.plot(x_indices, sims, color=color_sim, marker='o', markersize=10, 
                  linewidth=3.2, label='Cosine Similarity (↑)', zorder=4)
    ax1.set_ylabel('Speaker Cosine Similarity (↑)', color=color_sim, fontsize=14, fontweight='bold', labelpad=10)
    ax1.tick_params(axis='y', labelcolor=color_sim, labelsize=12)
    ax1.set_ylim(0.48, 0.66)

    ax1_twin = ax1.twinx()
    l2 = ax1_twin.plot(x_indices, cers, color=color_cer, marker='s', markersize=9, 
                       linewidth=3.0, linestyle='--', label='CER % (↓)', zorder=4)
    ax1_twin.set_ylabel('Character Error Rate (CER %) (↓)', color=color_cer, fontsize=14, fontweight='bold', labelpad=12)
    ax1_twin.tick_params(axis='y', labelcolor=color_cer, labelsize=12)
    ax1_twin.set_ylim(0.4, 4.4)

    # Highlight Sweet Spot centered in the green band between the two curves
    ax1.axvspan(1.65, 2.35, color='#2ca02c', alpha=0.18, zorder=1)
    ax1.text(2.0, 0.542, '★ Optimal Sweet Spot\n(Continuous Affine)', ha='center', va='center', 
             fontsize=11.0, fontweight='bold', color='#155724',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#d4edda', edgecolor='#28a745', lw=1.2))

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(models, fontsize=12, fontweight='medium')
    ax1.grid(True, axis='both')
    ax1.set_title('(a) The Alignment Continuum Trajectory', fontsize=15, fontweight='bold', pad=14, loc='left')

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.03, 0.95), 
               fontsize=11.5, framealpha=0.95, edgecolor='#c0c0c0')

    # -------------------------------------------------------------------------
    # Panel (b): Pareto Trade-off Frontier
    # -------------------------------------------------------------------------
    arrow_x = [sims[0], sims[1], sims[2], sims[3], sims[4], sims[5]]
    arrow_y = [cers[0], cers[1], cers[2], cers[3], cers[4], cers[5]]
    ax2.plot(arrow_x, arrow_y, color='#888888', linestyle=':', lw=2.2, zorder=2)

    # Global WCT
    ax2.scatter(sims[0], cers[0], color='#6c757d', s=140, marker='o', edgecolors='black', lw=1.5, 
                label='Global WCT (K=1)', zorder=5)
    ax2.annotate('Global WCT\n(K=1)', (sims[0], cers[0]), textcoords="offset points", 
                 xytext=(12, -10), fontsize=11.5, color='#333333', fontweight='semibold')

    # Soft Local WCT points
    ax2.scatter([sims[1], sims[3]], [cers[1], cers[3]], color='#1f77b4', s=140, marker='D', 
                edgecolors='black', lw=1.5, zorder=5, label='Soft WCT (K=3, 8)')
    ax2.annotate('K=3', (sims[1], cers[1]), textcoords="offset points", xytext=(-10, 10), fontsize=12, color='#1f77b4', fontweight='bold')
    ax2.annotate('K=8', (sims[3], cers[3]), textcoords="offset points", xytext=(12, 4), fontsize=12, color='#1f77b4', fontweight='bold')

    # Sweet Spot (K=5)
    ax2.scatter(sims[2], cers[2], color='#28a745', s=320, marker='*', edgecolors='black', lw=1.8, 
                label='Proposed Soft WCT (K=5)*', zorder=6)
    ax2.annotate('★ Proposed (K=5)\nLowest CER: 0.97%', (sims[2], cers[2]), textcoords="offset points", 
                 xytext=(0, -38), ha='center', fontsize=11.5, color='#155724', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#28a745', lw=1.5))

    # Instance-based (1-NN & 4-NN)
    ax2.scatter(sims[4:], cers[4:], color='#dc3545', s=150, marker='s', edgecolors='black', lw=1.5, 
                label='kNN-VC (1-NN, 4-NN)', zorder=5)
    ax2.annotate('1-NN', (sims[4], cers[4]), textcoords="offset points", xytext=(-35, 6), fontsize=12, color='#dc3545', fontweight='bold')
    ax2.annotate('4-NN (kNN-VC)', (sims[5], cers[5]), textcoords="offset points", xytext=(12, -5), fontsize=12, color='#dc3545', fontweight='bold')

    ax2.set_xlabel('Speaker Cosine Similarity (↑)', fontsize=14, fontweight='bold', labelpad=10)
    ax2.set_ylabel('Character Error Rate (CER %) (↓)', fontsize=14, fontweight='bold', labelpad=10)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_xlim(0.495, 0.675)
    ax2.set_ylim(0.3, 4.4)
    ax2.grid(True)
    ax2.set_title('(b) Intelligibility vs. Similarity Frontier', fontsize=15, fontweight='bold', pad=14, loc='left')
    
    # Legend placed in upper left where space is completely empty
    ax2.legend(loc='upper left', bbox_to_anchor=(0.03, 0.95), fontsize=11.5, framealpha=0.95, edgecolor='#c0c0c0')

    plt.tight_layout()

    png_wide = out_dir / "figure_continuum_large.png"
    pdf_wide = out_dir / "figure_continuum_large.pdf"
    plt.savefig(png_wide, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_wide, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # VERSION 2: Large Vertical 2-Panel Figure (8.5 x 11.5 inches)
    # =========================================================================
    fig_v, (ax1_v, ax2_v) = plt.subplots(2, 1, figsize=(8.5, 11.5), dpi=300)

    # Panel A
    l1_v = ax1_v.plot(x_indices, sims, color=color_sim, marker='o', markersize=10, 
                      linewidth=3.2, label='Cosine Similarity (↑)', zorder=4)
    ax1_v.set_ylabel('Speaker Cosine Similarity (↑)', color=color_sim, fontsize=14, fontweight='bold', labelpad=10)
    ax1_v.tick_params(axis='y', labelcolor=color_sim, labelsize=12)
    ax1_v.set_ylim(0.48, 0.66)

    ax1_v_twin = ax1_v.twinx()
    l2_v = ax1_v_twin.plot(x_indices, cers, color=color_cer, marker='s', markersize=9, 
                           linewidth=3.0, linestyle='--', label='CER % (↓)', zorder=4)
    ax1_v_twin.set_ylabel('Character Error Rate (CER %) (↓)', color=color_cer, fontsize=14, fontweight='bold', labelpad=12)
    ax1_v_twin.tick_params(axis='y', labelcolor=color_cer, labelsize=12)
    ax1_v_twin.set_ylim(0.4, 4.4)

    ax1_v.axvspan(1.65, 2.35, color='#2ca02c', alpha=0.18, zorder=1)
    ax1_v.text(2.0, 0.542, '★ Optimal Sweet Spot\n(Continuous Affine)', ha='center', va='center', 
               fontsize=11.0, fontweight='bold', color='#155724',
               bbox=dict(boxstyle='round,pad=0.35', facecolor='#d4edda', edgecolor='#28a745', lw=1.2))

    ax1_v.set_xticks(x_indices)
    ax1_v.set_xticklabels(models, fontsize=12, fontweight='medium')
    ax1_v.grid(True, axis='both')
    ax1_v.set_title('(a) The Alignment Continuum Trajectory', fontsize=15, fontweight='bold', pad=14, loc='left')

    lines_v = l1_v + l2_v
    labels_v = [l.get_label() for l in lines_v]
    ax1_v.legend(lines_v, labels_v, loc='upper left', bbox_to_anchor=(0.03, 0.95), 
                 fontsize=11.5, framealpha=0.95, edgecolor='#c0c0c0')

    # Panel B
    ax2_v.plot(arrow_x, arrow_y, color='#888888', linestyle=':', lw=2.2, zorder=2)
    ax2_v.scatter(sims[0], cers[0], color='#6c757d', s=140, marker='o', edgecolors='black', lw=1.5, 
                  label='Global WCT (K=1)', zorder=5)
    ax2_v.annotate('Global WCT\n(K=1)', (sims[0], cers[0]), textcoords="offset points", 
                   xytext=(12, -10), fontsize=11.5, color='#333333', fontweight='semibold')

    ax2_v.scatter([sims[1], sims[3]], [cers[1], cers[3]], color='#1f77b4', s=140, marker='D', 
                  edgecolors='black', lw=1.5, zorder=5, label='Soft WCT (K=3, 8)')
    ax2_v.annotate('K=3', (sims[1], cers[1]), textcoords="offset points", xytext=(-10, 10), fontsize=12, color='#1f77b4', fontweight='bold')
    ax2_v.annotate('K=8', (sims[3], cers[3]), textcoords="offset points", xytext=(12, 4), fontsize=12, color='#1f77b4', fontweight='bold')

    ax2_v.scatter(sims[2], cers[2], color='#28a745', s=320, marker='*', edgecolors='black', lw=1.8, 
                  label='Proposed Soft WCT (K=5)*', zorder=6)
    ax2_v.annotate('★ Proposed (K=5)\nLowest CER: 0.97%', (sims[2], cers[2]), textcoords="offset points", 
                   xytext=(0, -38), ha='center', fontsize=11.5, color='#155724', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='#28a745', lw=1.5))

    ax2_v.scatter(sims[4:], cers[4:], color='#dc3545', s=150, marker='s', edgecolors='black', lw=1.5, 
                  label='kNN-VC (1-NN, 4-NN)', zorder=5)
    ax2_v.annotate('1-NN', (sims[4], cers[4]), textcoords="offset points", xytext=(-35, 6), fontsize=12, color='#dc3545', fontweight='bold')
    ax2_v.annotate('4-NN (kNN-VC)', (sims[5], cers[5]), textcoords="offset points", xytext=(12, -5), fontsize=12, color='#dc3545', fontweight='bold')

    ax2_v.set_xlabel('Speaker Cosine Similarity (↑)', fontsize=14, fontweight='bold', labelpad=10)
    ax2_v.set_ylabel('Character Error Rate (CER %) (↓)', fontsize=14, fontweight='bold', labelpad=10)
    ax2_v.tick_params(axis='both', labelsize=12)
    ax2_v.set_xlim(0.495, 0.675)
    ax2_v.set_ylim(0.3, 4.4)
    ax2_v.grid(True)
    ax2_v.set_title('(b) Intelligibility vs. Similarity Frontier', fontsize=15, fontweight='bold', pad=14, loc='left')
    ax2_v.legend(loc='upper left', bbox_to_anchor=(0.03, 0.95), fontsize=11.5, framealpha=0.95, edgecolor='#c0c0c0')

    plt.tight_layout()

    png_vert = out_dir / "figure_continuum_vertical_large.png"
    pdf_vert = out_dir / "figure_continuum_vertical_large.pdf"
    plt.savefig(png_vert, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_vert, bbox_inches='tight')
    plt.close()

    print(f"Generated high-resolution large figures in {out_dir}:")
    print(f"  - Wide (14x6.2 in): figure_continuum_large.png (.pdf)")
    print(f"  - Vertical (8.5x11.5 in): figure_continuum_vertical_large.png (.pdf)")

if __name__ == "__main__":
    plot_alignment_continuum()
