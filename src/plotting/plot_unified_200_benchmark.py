import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

csv_summary = Path('/local_scratch/ssadok/un_projet_audio/output/tables/unified_200_benchmark_summary.csv')
if not csv_summary.exists():
    print(f"Waiting for {csv_summary}...")
    sys.exit(1)

df = pd.read_csv(csv_summary)

# Visual styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#2b2d42'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#8d99ae'
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['grid.linestyle'] = '--'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.5, 6), dpi=300)

palette = {
    'Classic WCT': '#1f77b4',
    'ICP-WCT (ours)': '#6a0dad',
    'kNN-VC (k=1)': '#ff7f0e',
    'kNN-VC (k=4)': '#d62728',
    'LinearVC': '#8c564b',
    'FreeVC': '#9c27b0',
    'FreeVC-s': '#3f51b5',
    'SoundStorm': '#00b4d8',
    'Soft Local WCT (Ours)': '#2ca02c',
    'Local Wasserstein (LWT - Ours)': '#008080',
    'Boosted LWT (alpha=1.5 - Ours)': '#e76f51'
}

markers = {
    'Classic WCT': 'o',
    'ICP-WCT (ours)': 'D',
    'kNN-VC (k=1)': 'v',
    'kNN-VC (k=4)': 's',
    'LinearVC': '^',
    'FreeVC': 'P',
    'FreeVC-s': 'X',
    'SoundStorm': 'd',
    'Soft Local WCT (Ours)': '*',
    'Local Wasserstein (LWT - Ours)': 'p',
    'Boosted LWT (alpha=1.5 - Ours)': 'h'
}

# Subplot 1: Pareto Space (CER vs Sim)
for _, row in df.iterrows():
    m = row['Method']
    c = palette.get(m, '#333333')
    mk = markers.get(m, 'o')
    msize = 14 if 'Ours' in m else 11
    
    ax1.errorbar(
        row['CER_mean'], row['Sim_mean'],
        xerr=row['CER_std'] / np.sqrt(10),
        yerr=row['Sim_std'] / np.sqrt(10),
        fmt=mk, color=c, ecolor=c,
        markersize=msize, capsize=4, elinewidth=1.2,
        label=f"{m} (RTF {row['RTF_mean']:.4f})"
    )

ax1.set_title("(a) Pareto Frontier on Unified Benchmark ($N=200$ Conversions)", fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel("Character Error Rate (CER %) $\\downarrow$ [Lower is better]", fontsize=10.5, fontweight='bold')
ax1.set_ylabel("Speaker Cosine Similarity $\\uparrow$ [Higher is better]", fontsize=10.5, fontweight='bold')
ax1.grid(True)
ax1.legend(loc='upper right', framealpha=0.92, fontsize=8.0)

# Subplot 2: Grouped Bar Chart of Multi-Metric Profiles
x = np.arange(len(df))
width = 0.18

rects1 = ax2.bar(x - 1.5*width, df['CER_mean'], width, label='CER (%) $\\downarrow$', color='#e63946', edgecolor='black', linewidth=0.6)
rects2 = ax2.bar(x - 0.5*width, df['WER_mean'], width, label='WER (%) $\\downarrow$', color='#f4a261', edgecolor='black', linewidth=0.6)
rects3 = ax2.bar(x + 0.5*width, df['Sim_mean'] * 10, width, label='Cosine Sim ($\\times 10$) $\\uparrow$', color='#1d3557', edgecolor='black', linewidth=0.6)
rects4 = ax2.bar(x + 1.5*width, df['Jitter_mean'], width, label='Rel. Jitter ($\\times$) $\\downarrow$', color='#2a9d8f', edgecolor='black', linewidth=0.6)

# Labels
short_names = [
    'Classic\nWCT', 'ICP-WCT\n(ours)', 'kNN-VC\n(k=1)', 'kNN-VC\n(k=4)', 'LinearVC',
    'FreeVC\n(DL)', 'FreeVC-s\n(DL)', 'SoundStorm\n(DL)', 'Soft Local\nWCT', 'Local Wass.\n(LWT)', 'Boosted\nLWT (1.5)'
]
ax2.set_xticks(x)
ax2.set_xticklabels(short_names, fontsize=7.0, fontweight='bold')
ax2.set_title("(b) Multi-dimensional Profile Across Voice Conversion Paradigms", fontsize=12, fontweight='bold', pad=10)
ax2.set_ylabel("Metric Values (Scaled)", fontsize=10.5, fontweight='bold')
ax2.grid(True, axis='y')
ax2.legend(loc='upper right', framealpha=0.92, fontsize=8.5)

# Values on bars
def autolabel(rects, scale=1.0, suffix=""):
    for rect in rects:
        height = rect.get_height()
        val = height / scale
        ax2.annotate(f'{val:.1f}{suffix}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=6.5, fontweight='bold')

autolabel(rects1, 1.0, "%")
autolabel(rects2, 1.0, "%")
autolabel(rects3, 10.0, "")
autolabel(rects4, 1.0, "x")

plt.tight_layout()
out_dir = Path('/local_scratch/ssadok/un_projet_audio/output/figures')
plt.savefig(out_dir / 'figure_overall_unified_200.png', dpi=300)
plt.savefig(out_dir / 'figure_overall_unified_200.pdf')
plt.close()
print("Updated figure_overall_unified_200 generated successfully!")
