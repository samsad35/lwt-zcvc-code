import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path('/local_scratch/ssadok/un_projet_audio/output/tables/controlled_operator_ablation_6spk.csv')
if not csv_path.exists():
    print(f"Error: {csv_path} not found.")
    sys.exit(1)

df = pd.read_csv(csv_path)

# Calculate mean and std per operator across the 6 speakers
summary = df.groupby('Operator').agg({
    'CER': ['mean', 'std'],
    'Sim': ['mean', 'std'],
    'Jitter': ['mean', 'std'],
    'Asymmetry': ['mean', 'std'],
    'RTF': ['mean', 'std']
}).reset_index()

# Flatten columns
summary.columns = ['Operator', 'CER_mean', 'CER_std', 'Sim_mean', 'Sim_std', 'Jitter_mean', 'Jitter_std', 'Asym_mean', 'Asym_std', 'RTF_mean', 'RTF_std']

# Desired order
desired_order = ['Local WCT', 'Local LS Affine', 'Local Bures-Wasserstein (LWT)']
summary['Operator'] = pd.Categorical(summary['Operator'], categories=desired_order, ordered=True)
summary = summary.sort_values('Operator').reset_index(drop=True)

print("=== Controlled Operator Ablation Summary (6 Speakers, Identical Routing) ===")
print(summary.to_string())

# Generate LaTeX Table
tex_path = Path('/local_scratch/ssadok/un_projet_audio/output/tables/tab_controlled_operator_ablation_6spk.tex')
with open(tex_path, 'w') as f:
    f.write("\\begin{table}[ht]\n")
    f.write("    \\centering\n")
    f.write("    \\caption{Controlled Operator Ablation ($K=10, \\beta=20.0, T=20$, averaged across 6 LibriSpeech speakers). Routing weights $w_k(x)$ and anchor centroids $\\mu_{X,k}, \\mu_{Y,k}$ are \\textbf{strictly identical} across all three methods.}\n")
    f.write("    \\label{tab:controlled_operator_ablation}\n")
    f.write("    \\resizebox{\\columnwidth}{!}{\n")
    f.write("    \\begin{tabular}{lcccc}\n")
    f.write("        \\toprule\n")
    f.write("        \\textbf{Local Operator $M_k$} & \\textbf{Operator Asymmetry} $\\mathcal{A}(M) \\downarrow$ & \\textbf{CER (\\%)} $\\downarrow$ & \\textbf{Sim} $\\uparrow$ & \\textbf{Rel. Jitter} $\\downarrow$ \\\\\n")
    f.write("        \\midrule\n")
    for _, row in summary.iterrows():
        op = row['Operator']
        asym = f"{row['Asym_mean']:.3f}"
        cer = f"{row['CER_mean']:.2f} $\\pm$ {row['CER_std']:.2f}"
        sim = f"{row['Sim_mean']:.3f} $\\pm$ {row['Sim_std']:.3f}"
        jit = f"{row['Jitter_mean']:.2f}$\\times$"
        
        if 'Wasserstein' in op:
            f.write(f"        \\textbf{{{op}}} & \\textbf{{{asym}}} & \\textbf{{{cer}}} & \\textbf{{{sim}}} & {jit} \\\\\n")
        else:
            f.write(f"        {op} & {asym} & {cer} & {sim} & {jit} \\\\\n")
            
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

colors = {
    'Local WCT': '#1f77b4',
    'Local LS Affine': '#ff7f0e',
    'Local Bures-Wasserstein (LWT)': '#008080'
}

markers = {
    'Local WCT': 'o',
    'Local LS Affine': 's',
    'Local Bures-Wasserstein (LWT)': 'p'
}

# Subplot 1: Pareto Space (CER vs Sim) with identical routing
for _, row in summary.iterrows():
    op = row['Operator']
    ax1.errorbar(
        row['CER_mean'], row['Sim_mean'],
        xerr=row['CER_std'], yerr=row['Sim_std'],
        fmt=markers[op], color=colors[op], ecolor=colors[op],
        markersize=14, capsize=6, capthick=1.5, elinewidth=1.5,
        alpha=0.9, label=f"{op} (Asym: {row['Asym_mean']:.3f})"
    )

# Connect with arrow or dashed line to show evolution
ax1.plot(summary['CER_mean'], summary['Sim_mean'], linestyle='--', color='gray', alpha=0.5, zorder=1)

ax1.set_title("(a) Accuracy vs. Speaker Similarity under Fixed Routing", fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel("Character Error Rate (CER %) $\\downarrow$ [Lower is better]", fontsize=11, fontweight='bold')
ax1.set_ylabel("Speaker Cosine Similarity $\\uparrow$ [Higher is better]", fontsize=11, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='lower right', framealpha=0.95, fontsize=10)

# Subplot 2: Grouped Bar Chart of Metrics (CER, Sim*10, Jitter, Asymmetry*10)
x = np.arange(len(desired_order))
width = 0.2

cer_vals = summary['CER_mean'].values
sim_vals = summary['Sim_mean'].values * 10
jit_vals = summary['Jitter_mean'].values
asym_vals = summary['Asym_mean'].values * 10

rects1 = ax2.bar(x - 1.5*width, cer_vals, width, label='CER (%) $\\downarrow$', color='#e63946', edgecolor='black', linewidth=0.8)
rects2 = ax2.bar(x - 0.5*width, sim_vals, width, label='Cosine Sim ($\\times 10$) $\\uparrow$', color='#1d3557', edgecolor='black', linewidth=0.8)
rects3 = ax2.bar(x + 0.5*width, jit_vals, width, label='Rel. Jitter ($\\times$) $\\downarrow$', color='#f4a261', edgecolor='black', linewidth=0.8)
rects4 = ax2.bar(x + 1.5*width, asym_vals, width, label='Asymmetry $\\mathcal{A}(M) (\\times 10) \\downarrow$', color='#2a9d8f', edgecolor='black', linewidth=0.8)

# Add values above bars
def autolabel(rects, scale=1.0, suffix=""):
    for rect in rects:
        height = rect.get_height()
        val = height / scale
        ax2.annotate(f'{val:.2f}{suffix}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')

autolabel(rects1, 1.0, "%")
autolabel(rects2, 10.0, "")
autolabel(rects3, 1.0, "x")
autolabel(rects4, 10.0, "")

ax2.set_xticks(x)
ax2.set_xticklabels(['Local WCT\n(Factorized)', 'Local LS Affine\n(Ridge Reg.)', 'Local Wasserstein\n(Symmetric Monge)'], fontsize=10.5, fontweight='bold')
ax2.set_title("(b) Metric Profiles & Operator Asymmetry", fontsize=13, fontweight='bold', pad=12)
ax2.set_ylabel("Metric Value (scaled)", fontsize=11, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
ax2.legend(loc='upper right', framealpha=0.95, fontsize=9.5)

plt.tight_layout()
out_png = Path('/local_scratch/ssadok/un_projet_audio/output/figures/figure_controlled_operator_ablation.png')
out_pdf = Path('/local_scratch/ssadok/un_projet_audio/output/figures/figure_controlled_operator_ablation.pdf')
plt.savefig(out_png, dpi=300)
plt.savefig(out_pdf)
plt.close()
print(f"Generated {out_png} and {out_pdf}")
