import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def plot_expanded_continuum():
    csv_path = Path("/local_scratch/ssadok/un_projet_audio/continuum_expanded_6spk.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    methods_order = [
        "Classic WCT (K=1)",
        "Soft Local WCT (K=3)",
        "Soft Local WCT (K=8)",
        "Soft Local WCT (K=25)",
        "Soft Local WCT (K=80)",
        "Soft Local WCT (K=250)",
        "1-NN (K=Nt, β→∞)"
    ]
    
    display_names = [
        "Classic WCT\n(K = 1)",
        "Soft WCT\n(K = 3)",
        "Soft WCT\n(K = 8)",
        "Soft WCT*\n(K = 25)",
        "Soft WCT\n(K = 80)",
        "Soft WCT\n(K = 250)",
        "1-NN (kNN-VC)\n(K = Nt, β→∞)"
    ]
    
    agg = df.groupby("Method", sort=False).mean(numeric_only=True).reindex(methods_order)
    std_agg = df.groupby("Method", sort=False).std(numeric_only=True).reindex(methods_order)
    
    sims = agg["Sim"].values
    cers = agg["CER"].values
    sim_stds = std_agg["Sim"].values
    cer_stds = std_agg["CER"].values
    
    print("Multi-Speaker Summary (6 Speakers):")
    for name, s, c in zip(methods_order, sims, cers):
        print(f"  {name:25s}: Sim = {s:.4f}, CER = {c:.2f}%")
        
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

    color_sim = '#1f77b4' # Deep blue
    color_cer = '#d62728' # Deep crimson

    # ULTRA-WIDE HORIZONTAL LAYOUT (22.5 x 5.8 inches)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22.5, 5.8), dpi=300)
    x_indices = np.arange(len(methods_order))

    # =========================================================================
    # Panel (a): Dense Multi-Speaker Continuum Trajectory (Dual Axis)
    # =========================================================================
    l1 = ax1.plot(x_indices, sims, color=color_sim, marker='o', markersize=10.5, 
                  linewidth=3.4, label='Speaker Cosine Sim (↑)', zorder=4)
    ax1.fill_between(x_indices, sims - sim_stds, sims + sim_stds, color=color_sim, alpha=0.12, zorder=3)
    ax1.set_ylabel('Speaker Cosine Similarity (↑)', color=color_sim, fontsize=15, fontweight='bold', labelpad=12)
    ax1.tick_params(axis='y', labelcolor=color_sim, labelsize=13)
    y1_min = max(0.45, min(sims - sim_stds) - 0.03)
    y1_max = min(0.85, max(sims + sim_stds) + 0.05)
    ax1.set_ylim(y1_min, y1_max)

    ax1_twin = ax1.twinx()
    l2 = ax1_twin.plot(x_indices, cers, color=color_cer, marker='s', markersize=9.5, 
                       linewidth=3.2, linestyle='--', label='CER % (↓)', zorder=4)
    ax1_twin.fill_between(x_indices, cers - cer_stds, cers + cer_stds, color=color_cer, alpha=0.12, zorder=3)
    ax1_twin.set_ylabel('Character Error Rate (CER %) (↓)', color=color_cer, fontsize=15, fontweight='bold', labelpad=14)
    ax1_twin.tick_params(axis='y', labelcolor=color_cer, labelsize=13)
    ax1_twin.set_ylim(0.0, max(cers + cer_stds) + 0.5)

    # Highlight Sweet Spot around K=25
    best_idx = np.argmin(cers)
    ax1.axvspan(best_idx - 0.35, best_idx + 0.35, color='#2ca02c', alpha=0.16, zorder=1)
    badge_y = y1_max - 0.04
    ax1.text(best_idx, badge_y, f'★ Optimal Sweet Spot\n(Lowest CER: {cers[best_idx]:.2f}%)', ha='center', va='center', 
             fontsize=12, fontweight='bold', color='#155724',
             bbox=dict(boxstyle='round,pad=0.45', facecolor='#d4edda', edgecolor='#28a745', lw=1.3))

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(display_names, fontsize=12, fontweight='medium')
    ax1.grid(True, axis='both')
    ax1.set_title('(a) Alignment Continuum Trajectory (6 Balanced Speakers Average ± 1σ)', 
                  fontsize=15.5, fontweight='bold', pad=15, loc='left')

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left', bbox_to_anchor=(0.02, 0.05), 
               fontsize=12.5, framealpha=0.95, edgecolor='#c0c0c0')

    # =========================================================================
    # Panel (b): Pareto Frontier (CER vs Speaker Sim)
    # =========================================================================
    # Continuum trajectory curve
    ax2.plot(sims, cers, color='#888888', linestyle=':', lw=2.4, zorder=2)

    # Classic WCT (K=1)
    ax2.scatter(sims[0], cers[0], color='#6c757d', s=160, marker='o', edgecolors='black', lw=1.6, 
                label='Classic WCT (K=1)', zorder=5)
    ax2.annotate('Classic WCT\n(K=1)', (sims[0], cers[0]), textcoords="offset points", 
                 xytext=(-15, 14), ha='center', fontsize=12, color='#333333', fontweight='semibold')

    # Intermediate Soft WCT points
    for idx in range(1, len(methods_order) - 1):
        k_val = methods_order[idx].split('(')[1].split(')')[0]
        if idx == best_idx:
            continue
        ax2.scatter(sims[idx], cers[idx], color='#1f77b4', s=140, marker='D', edgecolors='black', lw=1.5, zorder=5)
        offset = (10, 10) if idx % 2 == 1 else (-15, -16)
        ax2.annotate(k_val, (sims[idx], cers[idx]), textcoords="offset points", xytext=offset, 
                     fontsize=11.5, color='#1f77b4', fontweight='bold')

    # Sweet Spot
    ax2.scatter(sims[best_idx], cers[best_idx], color='#28a745', s=420, marker='*', edgecolors='black', lw=2.0, 
                label='Proposed Soft WCT (K=25)*', zorder=6)
    ax2.annotate(f'★ Proposed Sweet Spot\nLowest CER: {cers[best_idx]:.2f}%', (sims[best_idx], cers[best_idx]), 
                 textcoords="offset points", xytext=(0, -42), ha='center', fontsize=12, color='#155724', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#28a745', lw=1.8))

    # 1-NN (K=Nt, beta -> inf)
    ax2.scatter(sims[-1], cers[-1], color='#dc3545', s=170, marker='s', edgecolors='black', lw=1.6, 
                label='1-NN (K=Nt, β→∞)', zorder=5)
    ax2.annotate('1-NN (kNN-VC)\n(K=Nt, β→∞)', (sims[-1], cers[-1]), textcoords="offset points", 
                 xytext=(-30, 14), fontsize=12, color='#dc3545', fontweight='bold')

    ax2.set_xlabel('Speaker Cosine Similarity (↑)', fontsize=15, fontweight='bold', labelpad=12)
    ax2.set_ylabel('Character Error Rate (CER %) (↓)', fontsize=15, fontweight='bold', labelpad=12)
    ax2.tick_params(axis='both', labelsize=13)
    ax2.set_xlim(min(sims) - 0.03, max(sims) + 0.04)
    ax2.set_ylim(max(0.0, min(cers) - 0.3), max(cers) + 0.6)
    ax2.grid(True)
    ax2.set_title('(b) Intelligibility vs. Similarity Frontier (6 Balanced Speakers Average)', 
                  fontsize=15.5, fontweight='bold', pad=15, loc='left')
    ax2.legend(loc='upper left', bbox_to_anchor=(0.02, 0.95), fontsize=12, framealpha=0.95, edgecolor='#c0c0c0')

    plt.tight_layout()

    out_dir = Path("/local_scratch/ssadok/un_projet_audio/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "figure_continuum_expanded_6spk.png"
    pdf_path = out_dir / "figure_continuum_expanded_6spk.pdf"
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to:\n  {png_path}\n  {pdf_path}")
    plt.close(fig)

if __name__ == '__main__':
    plot_expanded_continuum()
