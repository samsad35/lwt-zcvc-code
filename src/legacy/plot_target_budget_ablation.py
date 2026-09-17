import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def plot_target_budget():
    csv_path = Path("/local_scratch/ssadok/un_projet_audio/output/target_budget_ablation_6spk.csv")
    if not csv_path.exists():
        print(f"Waiting for {csv_path}...")
        return

    df = pd.read_csv(csv_path)

    # Harmonize the max budget across speakers (3701, 3945, 4000 -> 4000)
    df['Budget_Nominal'] = df['Nt'].apply(lambda x: 4000 if x >= 3500 else x)
    budget_durations = {100: 2.0, 250: 5.0, 500: 10.0, 1000: 20.0, 2000: 40.0, 4000: 80.0}
    df['Duration_sec'] = df['Budget_Nominal'].map(budget_durations)

    agg = df.groupby(['Budget_Nominal', 'Method'], as_index=False).agg({
        'Duration_sec': 'first',
        'Sim': ['mean', 'std'],
        'CER': ['mean', 'std']
    })
    
    # Flatten columns
    agg.columns = ['Budget_Nominal', 'Method', 'Duration_sec', 'Sim_mean', 'Sim_std', 'CER_mean', 'CER_std']
    
    ours_df = agg[agg['Method'] == 'Ours (Soft Local WCT)'].sort_values('Budget_Nominal')
    knn_df = agg[agg['Method'] == 'kNN-VC (1-NN)'].sort_values('Budget_Nominal')
    
    print("Ours:")
    print(ours_df[['Budget_Nominal', 'Duration_sec', 'Sim_mean', 'CER_mean']])
    print("\nkNN-VC:")
    print(knn_df[['Budget_Nominal', 'Duration_sec', 'Sim_mean', 'CER_mean']])

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.8, 6.2), dpi=300)

    # Durations formatting for X-ticks
    x_nts = ours_df['Budget_Nominal'].values
    x_labels = [f"{nt} frames\n(~{dur:.0f}s)" for nt, dur in zip(ours_df['Budget_Nominal'], ours_df['Duration_sec'])]
    x_pos = np.arange(len(x_nts))

    # =========================================================================
    # Panel (a): CER vs. Target Audio Budget (Robustness to Low-Resource Target)
    # =========================================================================
    ax1.plot(x_pos, ours_df['CER_mean'], color='#2ca02c', marker='*', markersize=14, 
             linewidth=3.2, label='Ours (Soft Local WCT)*', zorder=5)
    ax1.fill_between(x_pos, ours_df['CER_mean'] - ours_df['CER_std'], 
                     ours_df['CER_mean'] + ours_df['CER_std'], color='#2ca02c', alpha=0.15, zorder=3)

    ax1.plot(x_pos, knn_df['CER_mean'], color='#dc3545', marker='s', markersize=9, 
             linewidth=3.0, linestyle='--', label='kNN-VC (1-NN)', zorder=4)
    ax1.fill_between(x_pos, knn_df['CER_mean'] - knn_df['CER_std'], 
                     knn_df['CER_mean'] + knn_df['CER_std'], color='#dc3545', alpha=0.12, zorder=3)

    # Annotate Few-Shot Advantage at 250 frames (~5s) and 500 frames (~10s)
    gap_5s = knn_df['CER_mean'].values[1] - ours_df['CER_mean'].values[1]
    ratio_5s = knn_df['CER_mean'].values[1] / ours_df['CER_mean'].values[1]
    ax1.annotate(f'Few-Shot Advantage\nCER divided by {ratio_5s:.1f}x\n({ours_df["CER_mean"].values[1]:.1f}% vs {knn_df["CER_mean"].values[1]:.1f}%)',
                 xy=(1.0, ours_df['CER_mean'].values[1]), xytext=(1.0, ours_df['CER_mean'].values[1] + 13),
                 ha='center', fontsize=10.5, color='#155724', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#d4edda', edgecolor='#28a745', lw=1.2),
                 arrowprops=dict(arrowstyle='->', color='#28a745', lw=1.6))

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, fontsize=11.5, fontweight='medium')
    ax1.set_xlabel('Target Data Budget $N_t$ (Target Audio Duration)', fontsize=14, fontweight='bold', labelpad=12)
    ax1.set_ylabel('Character Error Rate (CER %) (↓)', fontsize=14, fontweight='bold', labelpad=12)
    ax1.set_ylim(-1, 52)
    ax1.grid(True)
    ax1.set_title('(a) Intelligibility Robustness: CER (%) vs. Target Budget', fontsize=14.5, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=12, framealpha=0.95, edgecolor='#c0c0c0')

    # =========================================================================
    # Panel (b): Speaker Cosine Similarity vs. Target Audio Budget
    # =========================================================================
    ax2.plot(x_pos, ours_df['Sim_mean'], color='#2ca02c', marker='*', markersize=14, 
             linewidth=3.2, label='Ours (Soft Local WCT)*', zorder=5)
    ax2.fill_between(x_pos, ours_df['Sim_mean'] - ours_df['Sim_std'], 
                     ours_df['Sim_mean'] + ours_df['Sim_std'], color='#2ca02c', alpha=0.15, zorder=3)

    ax2.plot(x_pos, knn_df['Sim_mean'], color='#dc3545', marker='s', markersize=9, 
             linewidth=3.0, linestyle='--', label='kNN-VC (1-NN)', zorder=4)
    ax2.fill_between(x_pos, knn_df['Sim_mean'] - knn_df['Sim_std'], 
                     knn_df['Sim_mean'] + knn_df['Sim_std'], color='#dc3545', alpha=0.12, zorder=3)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, fontsize=11.5, fontweight='medium')
    ax2.set_xlabel('Target Data Budget $N_t$ (Target Audio Duration)', fontsize=14, fontweight='bold', labelpad=12)
    ax2.set_ylabel('Speaker Cosine Similarity (↑)', fontsize=14, fontweight='bold', labelpad=12)
    ax2.set_ylim(0.38, 0.80)
    ax2.grid(True)
    ax2.set_title('(b) Speaker Similarity vs. Target Budget', fontsize=14.5, fontweight='bold', pad=15)
    ax2.legend(loc='lower right', fontsize=12, framealpha=0.95, edgecolor='#c0c0c0')

    plt.tight_layout()

    out_dir = Path("/local_scratch/ssadok/un_projet_audio/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "figure_target_budget_ablation_6spk.png"
    pdf_path = out_dir / "figure_target_budget_ablation_6spk.pdf"
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved target budget figure to:\n  {png_path}\n  {pdf_path}")
    plt.close(fig)

if __name__ == '__main__':
    plot_target_budget()
