import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path('/local_scratch/ssadok/un_projet_audio/output/tables/speaker_boosted_bures_6spk.csv')
if not csv_path.exists():
    print(f"Error: {csv_path} not found.")
    sys.exit(1)

df = pd.read_csv(csv_path)

summary = df.groupby('Alpha').agg({
    'CER': ['mean', 'std'],
    'Sim': ['mean', 'std'],
    'Jitter': ['mean', 'std']
}).reset_index()

summary.columns = ['Alpha', 'CER_mean', 'CER_std', 'Sim_mean', 'Sim_std', 'Jitter_mean', 'Jitter_std']
summary = summary.sort_values('Alpha').reset_index(drop=True)

print("=== Speaker Boosted Bures-Wasserstein Summary (6 Speakers) ===")
print(summary.to_string())

# Generate LaTeX Table
tex_path = Path('/local_scratch/ssadok/un_projet_audio/output/tables/tab_speaker_boosted_bures_6spk.tex')
with open(tex_path, 'w') as f:
    f.write("\\begin{table}[ht]\n")
    f.write("    \\centering\n")
    f.write("    \\caption{Effect of Speaker Covariance Component Amplification $\\alpha$ in Local Bures-Wasserstein Transport ($K=10, \\beta=20.0, T=20$, averaged across 6 LibriSpeech speakers).}\n")
    f.write("    \\label{tab:speaker_boosted_bures}\n")
    f.write("    \\resizebox{\\columnwidth}{!}{\n")
    f.write("    \\begin{tabular}{cccc}\n")
    f.write("        \\toprule\n")
    f.write("        \\textbf{Speaker Covariance Boost $\\alpha$} & \\textbf{CER (\\%)} $\\downarrow$ & \\textbf{Speaker Sim} $\\uparrow$ & \\textbf{Rel. Jitter} $\\downarrow$ \\\\\n")
    f.write("        \\midrule\n")
    for _, row in summary.iterrows():
        a = row['Alpha']
        cer = f"{row['CER_mean']:.2f} $\\pm$ {row['CER_std']:.2f}"
        sim = f"{row['Sim_mean']:.3f} $\\pm$ {row['Sim_std']:.3f}"
        jit = f"{row['Jitter_mean']:.2f}$\\times$"
        
        if a == 1.0:
            f.write(f"        {a:.1f} (Baseline LWT) & {cer} & {sim} & {jit} \\\\\n")
        elif a in [1.2, 1.5]:
            f.write(f"        \\textbf{{{a:.1f} (Optimal Boost)}} & \\textbf{{{cer}}} & \\textbf{{{sim}}} & \\textbf{{{jit}}} \\\\\n")
        else:
            f.write(f"        {a:.1f} & {cer} & {sim} & {jit} \\\\\n")
            
    f.write("        \\bottomrule\n")
    f.write("    \\end{tabular}\n")
    f.write("    }\n")
    f.write("\\end{table}\n")

print(f"\nSaved LaTeX table to {tex_path}")

# Plotting
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.9

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

alphas = summary['Alpha'].values
cer_m = summary['CER_mean'].values
cer_s = summary['CER_std'].values
sim_m = summary['Sim_mean'].values
sim_s = summary['Sim_std'].values
jit_m = summary['Jitter_mean'].values
jit_s = summary['Jitter_std'].values

# Panel (a): Dual y-axis: CER vs Sim over alpha
color_sim = '#1d3557'
color_cer = '#e63946'

ax1.set_title("(a) Evolution of Biometric Similarity & Phonetic Error vs. $\\alpha$", fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel("Speaker Covariance Amplification Factor $\\alpha$", fontsize=11, fontweight='bold')

line1 = ax1.plot(alphas, sim_m, color=color_sim, marker='o', linewidth=2.5, markersize=8, label='Speaker Cosine Sim (ECAPA)')
ax1.fill_between(alphas, sim_m - sim_s, sim_m + sim_s, color=color_sim, alpha=0.15)
ax1.set_ylabel("Speaker Cosine Similarity $\\uparrow$", color=color_sim, fontsize=11, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_sim)
ax1.axvline(x=1.0, color='gray', linestyle=':', label='$\\alpha=1.0$ (Standard LWT)')
ax1.axvline(x=1.5, color='#2a9d8f', linestyle='--', label='$\\alpha=1.5$ (Optimal Peak)')
ax1.grid(True, linestyle='--', alpha=0.5)

ax1_twin = ax1.twinx()
line2 = ax1_twin.plot(alphas, cer_m, color=color_cer, marker='s', linewidth=2.5, markersize=8, label='CER (%)')
ax1_twin.fill_between(alphas, cer_m - cer_s, cer_m + cer_s, color=color_cer, alpha=0.15)
ax1_twin.set_ylabel("Character Error Rate (CER %) $\\downarrow$", color=color_cer, fontsize=11, fontweight='bold')
ax1_twin.tick_params(axis='y', labelcolor=color_cer)

# Combined legend for ax1
lines = line1 + line2 + [ax1.get_lines()[1], ax1.get_lines()[2]]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center left', framealpha=0.92, fontsize=9.5)

# Panel (b): Pareto Trajectory in CER-Sim Space
ax2.set_title("(b) Pareto Trajectory under Covariance Direction Amplification", fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel("Character Error Rate (CER %) $\\downarrow$ [Lower is better]", fontsize=11, fontweight='bold')
ax2.set_ylabel("Speaker Cosine Similarity $\\uparrow$ [Higher is better]", fontsize=11, fontweight='bold')

# Scatter with colormap
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=min(alphas), vmax=max(alphas))

for i, a in enumerate(alphas):
    c = cmap(norm(a))
    ax2.errorbar(cer_m[i], sim_m[i], xerr=cer_s[i], yerr=sim_s[i], fmt='o', color=c, ecolor=c, alpha=0.7, capsize=4)
    # Text annotation
    offset_x = 0.05 if a != 1.0 else -0.15
    offset_y = 0.005 if a < 1.5 else -0.008
    ax2.annotate(f"$\\alpha={a:.1f}$", (cer_m[i], sim_m[i]),
                 xytext=(10, 5), textcoords='offset points',
                 fontsize=9.5, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=c))

ax2.plot(cer_m, sim_m, color='gray', linestyle='--', alpha=0.6, zorder=1)

# Colorbar for alpha
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2)
cbar.set_label("Speaker Boost $\\alpha$", fontsize=10, fontweight='bold')

ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
out_png = Path('/local_scratch/ssadok/un_projet_audio/output/figures/figure_speaker_boosted_bures.png')
out_pdf = Path('/local_scratch/ssadok/un_projet_audio/output/figures/figure_speaker_boosted_bures.pdf')
plt.savefig(out_png, dpi=300)
plt.savefig(out_pdf)
plt.close()
print(f"Generated {out_png} and {out_pdf}")
