import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def plot_jacobian_ablation():
    csv_path = Path("/local_scratch/ssadok/un_projet_audio/output/tables/jacobian_functional_ablation_6spk.csv")
    if not csv_path.exists():
        print(f"Waiting for {csv_path}...")
        return
        
    df = pd.read_csv(csv_path)
    
    models_order = [
        'Global WCT (K=1)',
        'Fixed Routing (Frozen w_k)',
        'No Local Covariance (Dynamic Mean)',
        'Soft Local WCT (Full Ours)',
        'kNN-VC (1-NN Reference)'
    ]
    
    agg = df.groupby('Model', sort=False).agg({
        'Local_Covariance': 'first',
        'Dynamic_Routing': 'first',
        'CER': ['mean', 'std'],
        'Sim': ['mean', 'std'],
        'Relative_Jitter': ['mean', 'std']
    }).reindex(models_order)
    
    print("\n" + "="*85)
    print("=== LATEX TABLE FOR ICASSP PAPER: FUNCTIONAL ABLATION ===")
    print("="*85)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Functional Ablation of the Jacobian Components ($K=25$, LibriSpeech 6 Speakers).}")
    print(r"\label{tab:jacobian_ablation}")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"\textbf{Model Configuration} & \textbf{Local Cov.} & \textbf{Dyn. Route} & \textbf{CER (\%)} $\downarrow$ & \textbf{Cosine Sim.} $\uparrow$ & \textbf{Rel. Jitter} $\downarrow$ \\")
    print(r"\midrule")
    for m in models_order:
        row = agg.loc[m]
        loc_cov = row[('Local_Covariance', 'first')]
        dyn_rt = row[('Dynamic_Routing', 'first')]
        cer_m, cer_s = row[('CER', 'mean')], row[('CER', 'std')]
        sim_m, sim_s = row[('Sim', 'mean')], row[('Sim', 'std')]
        jit_m, jit_s = row[('Relative_Jitter', 'mean')], row[('Relative_Jitter', 'std')]
        bold = r"\textbf{" if "Ours" in m else ""
        ebold = r"}" if "Ours" in m else ""
        print(f"{bold}{m}{ebold} & {loc_cov} & {dyn_rt} & {bold}{cer_m:.2f} $\\pm$ {cer_s:.2f}{ebold} & {bold}{sim_m:.3f} $\\pm$ {sim_s:.3f}{ebold} & {bold}{jit_m:.2f} $\\pm$ {jit_s:.2f}{ebold} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}\n")

    # Plot 3-panel bar chart
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'axes.edgecolor': '#2b2b2b',
        'axes.linewidth': 1.6,
        'grid.color': '#e2e2e2',
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.major.width': 1.4,
        'ytick.major.width': 1.4,
        'font.size': 12.5
    })

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.5, 6.0), dpi=300)
    
    labels = [
        'Global WCT ($K=1$)',
        'Fixed Routing (Frozen $w$)',
        'Mean Only (No Loc. Cov.)',
        'Soft Local WCT (Ours)*',
        'kNN-VC ($1$-NN)'
    ]
    
    colors = ['#4a5568', '#3182ce', '#dd6b20', '#2ca02c', '#e53e3e']
    x = np.arange(len(models_order))
    bar_width = 0.58
    
    # 1. CER Bar Chart
    cers = agg[('CER', 'mean')].values
    cer_stds = agg[('CER', 'std')].values
    b1 = ax1.bar(x, cers, width=bar_width, yerr=cer_stds, capsize=5, color=colors, edgecolor='black', lw=1.5, alpha=0.9)
    ax1.set_ylabel('Character Error Rate (CER %) (↓)', fontsize=13.5, fontweight='bold')
    ax1.set_title('(a) Speech Intelligibility (CER %)', fontsize=14, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11, fontweight='medium', rotation=22, ha='right')
    ax1.set_ylim(0.0, 7.0)
    ax1.grid(True, axis='y')
    for bar, err in zip(b1, cer_stds):
        yval = bar.get_height()
        top_pos = yval + err + 0.18
        ax1.text(bar.get_x() + bar.get_width()/2.0, top_pos, f'{yval:.2f}%', 
                 ha='center', va='bottom', fontsize=11.5, fontweight='bold')

    # 2. Speaker Similarity Bar Chart
    sims = agg[('Sim', 'mean')].values
    sim_stds = agg[('Sim', 'std')].values
    b2 = ax2.bar(x, sims, width=bar_width, yerr=sim_stds, capsize=5, color=colors, edgecolor='black', lw=1.5, alpha=0.9)
    ax2.set_ylabel('Speaker Cosine Similarity (↑)', fontsize=13.5, fontweight='bold')
    ax2.set_title('(b) Speaker Biometric Similarity', fontsize=14, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11, fontweight='medium', rotation=22, ha='right')
    ax2.set_ylim(0.0, 0.90)
    ax2.grid(True, axis='y')
    for bar, err in zip(b2, sim_stds):
        yval = bar.get_height()
        top_pos = yval + err + 0.02
        ax2.text(bar.get_x() + bar.get_width()/2.0, top_pos, f'{yval:.3f}', 
                 ha='center', va='bottom', fontsize=11.5, fontweight='bold')

    # 3. Relative Trajectory Jitter Bar Chart
    jits = agg[('Relative_Jitter', 'mean')].values
    jit_stds = agg[('Relative_Jitter', 'std')].values
    b3 = ax3.bar(x, jits, width=bar_width, yerr=jit_stds, capsize=5, color=colors, edgecolor='black', lw=1.5, alpha=0.9)
    ax3.axhline(1.0, color='gray', linestyle=':', lw=2, label='Source Velocity (=1.0)')
    ax3.set_ylabel('Relative Trajectory Jitter (↓)', fontsize=13.5, fontweight='bold')
    ax3.set_title(r'(c) Temporal Jitter ($\|\Delta \hat{x}\| / \|\Delta x_{\mathrm{src}}\|$)', fontsize=14, fontweight='bold', pad=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=11, fontweight='medium', rotation=22, ha='right')
    ax3.set_ylim(0.0, 1.35)
    ax3.grid(True, axis='y')
    ax3.legend(loc='upper left', fontsize=11)
    for bar, err in zip(b3, jit_stds):
        yval = bar.get_height()
        top_pos = yval + err + 0.035
        ax3.text(bar.get_x() + bar.get_width()/2.0, top_pos, f'{yval:.2f}x', 
                 ha='center', va='bottom', fontsize=11.5, fontweight='bold')

    plt.tight_layout()
    
    out_dir = Path("/local_scratch/ssadok/un_projet_audio/output/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "figure_jacobian_functional_ablation.png"
    pdf_path = out_dir / "figure_jacobian_functional_ablation.pdf"
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved Jacobian ablation figure to:\n  {png_path}\n  {pdf_path}")
    plt.close(fig)

if __name__ == '__main__':
    plot_jacobian_ablation()
