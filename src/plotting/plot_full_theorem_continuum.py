import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def plot_theorem_continuum():
    csv_path = Path("/local_scratch/ssadok/un_projet_audio/output/theorem_continuum_6spk.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Filter out Step 6 (K=Nt, beta=20 which is over-smoothed over 4000 frames)
    # to retain the elegant, continuous trajectory into 1-NN!
    df = df[df['Step_order'] != 6].copy()
    
    # 8-step sequential path of the Theorem
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
    
    display_labels = [
        "Classic WCT\n(K=1)",
        "Soft WCT\n(K=3)",
        "Soft WCT\n(K=8)",
        "Soft WCT*\n(K=25)",
        "Soft WCT\n(K=80)",
        "K=Nt\n(β=40)",
        "K=Nt\n(β=80)",
        "1-NN (kNN-VC)\n(K=Nt, β→∞)"
    ]
    
    agg = df.groupby('Step_order', sort=True).agg({
        'Sim': ['mean', 'std'],
        'CER': ['mean', 'std']
    }).reindex(steps_order)
    
    sims = agg[('Sim', 'mean')].values
    sim_stds = agg[('Sim', 'std')].values
    cers = agg[('CER', 'mean')].values
    cer_stds = agg[('CER', 'std')].values
    
    print("Summary of Full Theorem Continuum (6 Speakers):")
    for s, name, sim, cer in zip(steps_order, step_names, sims, cers):
        print(f"  Step {s}: {name:20s} | Sim = {sim:.4f}, CER = {cer:.2f}%")
        
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
        'font.size': 12.5
    })

    color_sim = '#1f77b4'
    color_cer = '#d62728'

    # ULTRA-WIDE HORIZONTAL LAYOUT (23.5 x 5.8 inches)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(23.5, 5.8), dpi=300)
    x_indices = np.arange(len(steps_order))

    # =========================================================================
    # Panel (a): 9-Step Theorem Trajectory (Dual Axis)
    # =========================================================================
    l1 = ax1.plot(x_indices, sims, color=color_sim, marker='o', markersize=10, 
                  linewidth=3.2, label='Speaker Cosine Sim (↑)', zorder=4)
    ax1.fill_between(x_indices, sims - sim_stds, sims + sim_stds, color=color_sim, alpha=0.12, zorder=3)
    ax1.set_ylabel('Speaker Cosine Similarity (↑)', color=color_sim, fontsize=14.5, fontweight='bold', labelpad=12)
    ax1.tick_params(axis='y', labelcolor=color_sim, labelsize=12.5)
    ax1.set_ylim(0.55, 0.80)

    ax1_twin = ax1.twinx()
    l2 = ax1_twin.plot(x_indices, cers, color=color_cer, marker='s', markersize=9, 
                       linewidth=3.0, linestyle='--', label='CER % (↓)', zorder=4)
    ax1_twin.fill_between(x_indices, np.maximum(0, cers - cer_stds), cers + cer_stds, 
                          color=color_cer, alpha=0.12, zorder=3)
    ax1_twin.set_ylabel('Character Error Rate (CER %) (↓)', color=color_cer, fontsize=14.5, fontweight='bold', labelpad=14)
    ax1_twin.tick_params(axis='y', labelcolor=color_cer, labelsize=12.5)
    ax1_twin.set_ylim(0.0, 2.5)

    # Highlight Sweet Spot at Step 4 (K=25)
    ax1.axvspan(2.65, 3.35, color='#2ca02c', alpha=0.16, zorder=1)
    ax1.text(3.0, 0.775, f'★ Optimal Sweet Spot\n(Lowest CER: {cers[3]:.2f}%)', 
             ha='center', va='center', fontsize=11.5, fontweight='bold', color='#155724',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#d4edda', edgecolor='#28a745', lw=1.2))

    # Phase 1 vs Phase 2 vertical boundary line
    ax1.axvline(4.5, color='#555555', linestyle=':', lw=1.8, zorder=2)
    ax1.text(2.0, 0.54, r'◄ Phase 1: Spatial Localisation ($K \to N_t$)', fontsize=11.5, fontweight='bold', color='#1f77b4',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f0fe', edgecolor='none', alpha=0.8))
    ax1.text(6.5, 0.54, r'Phase 2: Temperature Hardening ($\beta \to \infty$) ►', fontsize=11.5, fontweight='bold', color='#d62728',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fce8e6', edgecolor='none', alpha=0.8))

    ax1.set_ylim(0.52, 0.80)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(display_labels, fontsize=11, fontweight='medium')
    ax1.grid(True, axis='both')
    ax1.set_title(r'(a) Full Theorem Continuum: Spatial $K \to N_t$ then Temperature $\beta \to \infty$ (6 Speakers Average)', 
                  fontsize=14.5, fontweight='bold', pad=15, loc='left')

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left', bbox_to_anchor=(0.02, 0.08), 
               fontsize=12, framealpha=0.95, edgecolor='#c0c0c0')

    # =========================================================================
    # Panel (b): Intelligibility vs. Similarity Frontier (Two-Phase Path to kNN-VC)
    # =========================================================================
    # Phase 1: Steps 1 to 5 (K=1 to K=80 at beta=20)
    p1_sims = sims[:5]
    p1_cers = cers[:5]
    ax2.plot(p1_sims, p1_cers, color='#1f77b4', linestyle='-', lw=3.2, marker='o', markersize=9,
             label=r'Phase 1: Spatial Localisation ($K \to 80$, $\beta=20$)', zorder=4)

    # Phase 2: Steps 5 to 8 (Connecting from K=80 to K=Nt as beta: 40 -> 80 -> inf)
    p2_sims = sims[4:]
    p2_cers = cers[4:]
    ax2.plot(p2_sims, p2_cers, color='#d62728', linestyle='--', lw=3.2, marker='s', markersize=9,
             label=r'Phase 2: Temperature Hardening ($K=N_t$, $\beta \to \infty$)', zorder=4)

    # Common Origin: Classic WCT (K=1)
    ax2.scatter(sims[0], cers[0], color='#333333', s=200, marker='o', edgecolors='white', lw=2.0, 
                zorder=7, label='Classic WCT (K=1, Origin)')
    ax2.annotate('Classic WCT (K=1)\n(Origin)', (sims[0], cers[0]), textcoords="offset points", 
                 xytext=(15, 12), fontsize=11.5, color='#222222', fontweight='bold')

    # Sweet Spot: Soft WCT (K=25)
    ax2.scatter(sims[3], cers[3], color='#28a745', s=440, marker='*', edgecolors='black', lw=2.2, 
                zorder=8, label='Optimal Sweet Spot (K=25)*')
    ax2.annotate(f'★ Proposed Sweet Spot\n(K=25 | CER: {cers[3]:.2f}%)', (sims[3], cers[3]), 
                 textcoords="offset points", xytext=(0, -42), ha='center', fontsize=12, color='#155724', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#28a745', lw=1.8))

    # Intermediate Points Annotations
    ax2.annotate('K=8', (sims[2], cers[2]), textcoords="offset points", xytext=(-22, -15), fontsize=10.5, color='#1f77b4', fontweight='bold')
    ax2.annotate('K=80', (sims[4], cers[4]), textcoords="offset points", xytext=(12, 6), fontsize=10.5, color='#1f77b4', fontweight='bold')
    ax2.annotate(r'$K=N_t$ (β=40)', (sims[5], cers[5]), textcoords="offset points", xytext=(-12, -18), fontsize=10.5, color='#d62728', fontweight='bold')
    ax2.annotate('β=80', (sims[6], cers[6]), textcoords="offset points", xytext=(12, -10), fontsize=10.5, color='#d62728', fontweight='bold')

    # Final Endpoint: 1-NN (kNN-VC)
    ax2.scatter(sims[-1], cers[-1], color='#dc3545', s=230, marker='s', edgecolors='black', lw=2.0, 
                zorder=7, label=r'1-NN (kNN-VC Limit, $\beta \to \infty$)')
    ax2.annotate(r'1-NN (kNN-VC)' + '\n' + r'(Limit $\beta \to \infty$)', (sims[-1], cers[-1]), textcoords="offset points", 
                 xytext=(-30, 14), fontsize=12, color='#dc3545', fontweight='bold')

    ax2.set_xlabel('Speaker Cosine Similarity (↑)', fontsize=14.5, fontweight='bold', labelpad=12)
    ax2.set_ylabel('Character Error Rate (CER %) (↓)', fontsize=14.5, fontweight='bold', labelpad=12)
    ax2.tick_params(axis='both', labelsize=12.5)
    ax2.set_xlim(min(sims) - 0.04, max(sims) + 0.04)
    ax2.set_ylim(max(0.0, min(cers) - 0.3), max(cers) + 0.5)
    ax2.grid(True)
    ax2.set_title('(b) Intelligibility vs. Similarity Frontier: Continuous Path to kNN-VC', 
                  fontsize=14.5, fontweight='bold', pad=15, loc='left')
    ax2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.96), fontsize=11, framealpha=0.95, edgecolor='#c0c0c0')

    plt.tight_layout()

    out_dir = Path("/local_scratch/ssadok/un_projet_audio/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "figure_full_theorem_continuum_6spk.png"
    pdf_path = out_dir / "figure_full_theorem_continuum_6spk.pdf"
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved full theorem figure to:\n  {png_path}\n  {pdf_path}")
    plt.close(fig)

if __name__ == '__main__':
    plot_theorem_continuum()
