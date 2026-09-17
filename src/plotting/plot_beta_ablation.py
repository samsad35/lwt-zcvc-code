import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def plot_beta_experiment():
    csv_path = Path("/local_scratch/ssadok/un_projet_audio/beta_ablation_6spk.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return
        
    df = pd.read_csv(csv_path)
    
    beta_order = ["β=2", "β=5", "β=10", "β=20", "β=40", "β=100", "Hard (β→∞)"]
    display_labels = ["β = 2\n(Near-uniform)", "β = 5", "β = 10", "β = 20*\n(Sweet Spot)", "β = 40", "β = 100", "Hard\n(β → ∞)"]
    
    agg = df.groupby("Beta_label", sort=False).mean(numeric_only=True).reindex(beta_order)
    std_agg = df.groupby("Beta_label", sort=False).std(numeric_only=True).reindex(beta_order)
    
    entropies = agg["Entropy"].values
    entropy_stds = std_agg["Entropy"].values
    sims = agg["Sim"].values
    sim_stds = std_agg["Sim"].values
    cers = agg["CER"].values
    cer_stds = std_agg["CER"].values
    
    print("Summary of Beta Ablation (6 Speakers):")
    for b, h, s, c in zip(beta_order, entropies, sims, cers):
        print(f"  {b:14s}: Entropy = {h:.3f}, Sim = {s:.4f}, CER = {c:.2f}%")
        
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

    color_ent = '#6f42c1' # Royal purple
    color_sim = '#1f77b4' # Rich corporate blue
    color_cer = '#d62728' # Deep crimson

    # ULTRA-WIDE HORIZONTAL LAYOUT (22 x 5.8 inches)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 5.8), dpi=300)
    x_indices = np.arange(len(beta_order))

    # =========================================================================
    # Panel (a): Entropy Collapse (Softmax -> Argmax Transition)
    # =========================================================================
    ax1.plot(x_indices, entropies, color=color_ent, marker='D', markersize=11, 
             linewidth=3.4, label='Weight Entropy H(w) (nats)', zorder=4)
    ax1.fill_between(x_indices, np.maximum(0, entropies - entropy_stds), entropies + entropy_stds, 
                     color=color_ent, alpha=0.15, zorder=3)
    ax1.set_ylabel('Shannon Entropy $H(w)$ [nats] (↓)', color=color_ent, fontsize=15, fontweight='bold', labelpad=12)
    ax1.tick_params(axis='y', labelcolor=color_ent, labelsize=13)
    ax1.set_ylim(-0.1, np.log(25) + 0.3)
    
    # Horizontal reference for uniform distribution H = log(K)
    ax1.axhline(np.log(25), color='#888888', linestyle=':', lw=2.0, zorder=2)
    ax1.text(0.1, np.log(25) + 0.08, 'Maximum Entropy: $\\log(K) = 3.22$ nats (Uniform)', 
             fontsize=11.5, color='#555555', fontstyle='italic')
    
    # Text annotation for Dirac collapse
    ax1.annotate('Exact Dirac Collapse\n$w_k \\to \\mathbb{I}(k = k^*)$, $H \\to 0$', 
                 (x_indices[-1], entropies[-1]), textcoords="offset points", xytext=(-65, 38),
                 ha='center', fontsize=11.5, fontweight='bold', color=color_ent,
                 arrowprops=dict(arrowstyle='->', color=color_ent, lw=1.6))

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(display_labels, fontsize=12, fontweight='medium')
    ax1.grid(True, axis='both')
    ax1.set_title('(a) Softmax Entropy Collapse as $\\beta \\to \\infty$ (Fixed $K=25$, 6 Speakers)', 
                  fontsize=15.5, fontweight='bold', pad=15, loc='left')

    # =========================================================================
    # Panel (b): Intelligibility & Similarity vs Temperature Beta (Dual Axis)
    # =========================================================================
    l1 = ax2.plot(x_indices, sims, color=color_sim, marker='o', markersize=10.5, 
                  linewidth=3.4, label='Speaker Cosine Sim (↑)', zorder=4)
    ax2.fill_between(x_indices, sims - sim_stds, sims + sim_stds, color=color_sim, alpha=0.12, zorder=3)
    ax2.set_ylabel('Speaker Cosine Similarity (↑)', color=color_sim, fontsize=15, fontweight='bold', labelpad=12)
    ax2.tick_params(axis='y', labelcolor=color_sim, labelsize=13)
    y2_min = 0.35
    y2_max = 0.78
    ax2.set_ylim(y2_min, y2_max)

    ax2_twin = ax2.twinx()
    l2 = ax2_twin.plot(x_indices, cers, color=color_cer, marker='s', markersize=9.5, 
                       linewidth=3.2, linestyle='--', label='CER % (↓)', zorder=4)
    ax2_twin.fill_between(x_indices, cers - cer_stds, cers + cer_stds, color=color_cer, alpha=0.12, zorder=3)
    ax2_twin.set_ylabel('Character Error Rate (CER %) (↓)', color=color_cer, fontsize=15, fontweight='bold', labelpad=14)
    ax2_twin.tick_params(axis='y', labelcolor=color_cer, labelsize=13)
    ax2_twin.set_ylim(0.0, max(cers + cer_stds) + 0.6)

    # Highlight Optimal Beta Sweet Spot (beta=20)
    best_b_idx = 3 # beta = 20
    ax2.axvspan(best_b_idx - 0.35, best_b_idx + 0.35, color='#2ca02c', alpha=0.16, zorder=1)
    ax2.text(best_b_idx, y2_max - 0.04, f'★ Optimal Temperature\n(Lowest CER: {cers[best_b_idx]:.2f}%)', 
             ha='center', va='center', fontsize=12, fontweight='bold', color='#155724',
             bbox=dict(boxstyle='round,pad=0.45', facecolor='#d4edda', edgecolor='#28a745', lw=1.3))

    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(display_labels, fontsize=12, fontweight='medium')
    ax2.grid(True, axis='both')
    ax2.set_title('(b) Impact of Softmax Hardness on Intelligibility & Similarity', 
                  fontsize=15.5, fontweight='bold', pad=15, loc='left')

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='lower left', bbox_to_anchor=(0.02, 0.05), 
               fontsize=12.5, framealpha=0.95, edgecolor='#c0c0c0')

    plt.tight_layout()

    out_dir = Path("/local_scratch/ssadok/un_projet_audio/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "figure_beta_ablation_6spk.png"
    pdf_path = out_dir / "figure_beta_ablation_6spk.pdf"
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to:\n  {png_path}\n  {pdf_path}")
    plt.close(fig)

if __name__ == '__main__':
    plot_beta_experiment()
