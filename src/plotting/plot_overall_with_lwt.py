import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

df_baselines = pd.read_csv('/local_scratch/ssadok/un_projet_audio/output/tables/overall_baselines_6spk.csv')
df_lwt = pd.read_csv('/local_scratch/ssadok/un_projet_audio/output/tables/lwt_6spk.csv')

# Combine datasets
df_all = pd.concat([df_baselines, df_lwt], ignore_index=True)

method_order = [
    'Classic WCT',
    'ICP-WCT (ours)',
    'kNN-VC',
    'LinearVC',
    'Soft Local WCT (Ours)',
    'Local Wasserstein Transport (Ours)'
]

grouped = df_all.groupby('Method').agg({
    'Approach_Type': 'first',
    'CER': ['mean', 'std'],
    'Sim': ['mean', 'std'],
    'RTF': ['mean', 'std'],
    'Relative_Jitter': ['mean', 'std']
}).reindex(method_order)

bench_rtfs = {
    'Classic WCT': 0.0004,
    'ICP-WCT (ours)': 0.0500,
    'kNN-VC': 0.0034,
    'LinearVC': 0.1800,
    'Soft Local WCT (Ours)': 0.0004,
    'Local Wasserstein Transport (Ours)': 0.0010
}

# Generate Complete LaTeX Table
latex_table = r"""\begin{table}[ht]
    \centering
    \caption{Overall performance comparison against fundamental baselines and proposed Wasserstein mappings ($T=20$ utterances, averaged across 6 LibriSpeech speakers).}
    \label{tab:overall_baselines_lwt_6spk}
    \resizebox{\columnwidth}{!}{
    \begin{tabular}{llcccc}
        \toprule
        \textbf{Method} & \textbf{Approach Type} & \textbf{CER (\%)} $\downarrow$ & \textbf{Sim} $\uparrow$ & \textbf{Jitter} $\downarrow$ & \textbf{RTF} $\downarrow$ \\
        \midrule
"""

for m in method_order:
    row = grouped.loc[m]
    app = row['Approach_Type']['first']
    cer_m, cer_s = row['CER']['mean'], row['CER']['std']
    sim_m, sim_s = row['Sim']['mean'], row['Sim']['std']
    jit_m, jit_s = row['Relative_Jitter']['mean'], row['Relative_Jitter']['std']
    rtf_val = bench_rtfs[m]
    
    is_best_cer = (m == 'Soft Local WCT (Ours)')
    is_best_sim = (m == 'kNN-VC')
    is_best_rtf = (rtf_val <= 0.0004)
    
    cer_str = f"\\textbf{{{cer_m:.2f} $\\pm$ {cer_s:.2f}}}" if is_best_cer else f"{cer_m:.2f} $\\pm$ {cer_s:.2f}"
    sim_str = f"\\textbf{{{sim_m:.3f} $\\pm$ {sim_s:.3f}}}" if is_best_sim else f"{sim_m:.3f} $\\pm$ {sim_s:.3f}"
    jit_str = f"{jit_m:.2f}$\\times$"
    rtf_str = f"\\textbf{{{rtf_val:.4f}}}" if is_best_rtf else f"{rtf_val:.4f}"
    m_str = f"\\textbf{{{m}}}" if "Ours" in m else m
    
    if m == 'Soft Local WCT (Ours)':
        latex_table += "        \\midrule\n"
    latex_table += f"        {m_str} & {app} & {cer_str} & {sim_str} & {jit_str} & {rtf_str} \\\\\n"

latex_table += r"""        \bottomrule
    \end{tabular}
    }
\end{table}
"""

with open('/local_scratch/ssadok/un_projet_audio/output/tables/tab_overall_baselines_with_lwt_6spk.tex', 'w') as f:
    f.write(latex_table)
print("Saved updated LaTeX table with LWT.")

# ------------------------------------------------------------------------------
# Create Clean 2-Panel Figure
# ------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 5.5), dpi=300)

colors = {
    'Classic WCT': '#1f77b4',
    'ICP-WCT (ours)': '#7b3294',
    'kNN-VC': '#d62728',
    'LinearVC': '#8c564b',
    'Soft Local WCT (Ours)': '#2ca02c',
    'Local Wasserstein Transport (Ours)': '#008080' # Teal
}

markers = {
    'Classic WCT': 'o',
    'ICP-WCT (ours)': 'D',
    'kNN-VC': 's',
    'LinearVC': '^',
    'Soft Local WCT (Ours)': '*',
    'Local Wasserstein Transport (Ours)': 'p' # Pentagon
}

# Panel 1: Pareto Frontier
for m in method_order:
    row = grouped.loc[m]
    cer_m = row['CER']['mean']
    cer_s = row['CER']['std']
    sim_m = row['Sim']['mean']
    sim_s = row['Sim']['std']
    rtf_val = bench_rtfs[m]
    
    sz = 260 if "Ours" in m else 130
    ax1.errorbar(cer_m, sim_m, xerr=cer_s, yerr=sim_s, fmt='none', ecolor=colors[m], elinewidth=1.6, capsize=4, alpha=0.5)
    ax1.scatter(cer_m, sim_m, color=colors[m], s=sz, marker=markers[m], label=f"{m} (RTF {rtf_val:.4f})", zorder=5, edgecolor='black', linewidth=0.9)

# Connect trade-off hull
pts = [
    (grouped.loc['Classic WCT']['CER']['mean'], grouped.loc['Classic WCT']['Sim']['mean']),
    (grouped.loc['Soft Local WCT (Ours)']['CER']['mean'], grouped.loc['Soft Local WCT (Ours)']['Sim']['mean']),
    (grouped.loc['Local Wasserstein Transport (Ours)']['CER']['mean'], grouped.loc['Local Wasserstein Transport (Ours)']['Sim']['mean']),
    (grouped.loc['kNN-VC']['CER']['mean'], grouped.loc['kNN-VC']['Sim']['mean'])
]
ax1.plot([p[0] for p in pts], [p[1] for p in pts], linestyle='--', color='#2ca02c', alpha=0.55, linewidth=1.8, zorder=3)

ax1.set_xlabel('Character Error Rate (CER %) $\\downarrow$ [Lower is better]', fontsize=11, fontweight='bold')
ax1.set_ylabel('Speaker Cosine Similarity $\\uparrow$ [Higher is better]', fontsize=11, fontweight='bold')
ax1.set_title('(a) Pareto Frontier on 6 LibriSpeech Speakers ($T=20$)', fontsize=12.5, fontweight='bold', pad=12)
ax1.grid(True, linestyle='--', alpha=0.45)
ax1.legend(loc='lower right', fontsize=8.8, framealpha=0.92)

# Annotations for both our methods
c_wct = grouped.loc['Soft Local WCT (Ours)']['CER']['mean']
s_wct = grouped.loc['Soft Local WCT (Ours)']['Sim']['mean']
ax1.annotate('Soft Local WCT (Ours)\nMin CER ($0.62\\%$, RTF $0.0004$)',
             xy=(c_wct, s_wct),
             xytext=(c_wct + 0.8, s_wct - 0.05),
             arrowprops=dict(arrowstyle='->', lw=1.5, color='#2ca02c'),
             fontsize=8.8, fontweight='bold', color='#1e7120',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#eefbee', edgecolor='#2ca02c', alpha=0.92))

c_lwt = grouped.loc['Local Wasserstein Transport (Ours)']['CER']['mean']
s_lwt = grouped.loc['Local Wasserstein Transport (Ours)']['Sim']['mean']
ax1.annotate('Local Wasserstein (LWT - Ours)\nHigh Sim ($0.722$, CER $0.69\\%$)',
             xy=(c_lwt, s_lwt),
             xytext=(c_lwt + 1.2, s_lwt + 0.04),
             arrowprops=dict(arrowstyle='->', lw=1.5, color='#008080'),
             fontsize=8.8, fontweight='bold', color='#006666',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#e0f2f1', edgecolor='#008080', alpha=0.92))

ax1.set_xlim(-0.2, 8.8)
ax1.set_ylim(0.58, 0.82)

# Panel 2: Comparative Bar Chart
x_pos = np.arange(len(method_order))
width = 0.25

cers_vals = [grouped.loc[m]['CER']['mean'] for m in method_order]
cers_errs = [grouped.loc[m]['CER']['std'] for m in method_order]
sims_vals = [grouped.loc[m]['Sim']['mean'] for m in method_order]
sims_errs = [grouped.loc[m]['Sim']['std'] for m in method_order]
jits_vals = [grouped.loc[m]['Relative_Jitter']['mean'] for m in method_order]
jits_errs = [grouped.loc[m]['Relative_Jitter']['std'] for m in method_order]

rects1 = ax2.bar(x_pos - width, cers_vals, width, yerr=cers_errs, label='CER (%) $\\downarrow$', color='#ca0020', alpha=0.85, capsize=3, edgecolor='black', linewidth=0.6)
rects2 = ax2.bar(x_pos, [s * 10 for s in sims_vals], width, yerr=[s * 10 for s in sims_errs], label='Cosine Sim ($\\times 10$) $\\uparrow$', color='#0571b0', alpha=0.85, capsize=3, edgecolor='black', linewidth=0.6)
rects3 = ax2.bar(x_pos + width, jits_vals, width, yerr=jits_errs, label='Rel. Jitter ($\\times$) $\\downarrow$', color='#e66101', alpha=0.85, capsize=3, edgecolor='black', linewidth=0.6)

ax2.axhline(1.0, color='gray', linestyle=':', linewidth=1.2, alpha=0.8)

for rect in rects1:
    h = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2., h + 0.18, f'{h:.2f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold', rotation=25)
for rect in rects2:
    h = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2., h + 0.18, f'{h/10:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold', rotation=25)
for rect in rects3:
    h = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2., h + 0.18, f'{h:.2f}x', ha='center', va='bottom', fontsize=7.5, fontweight='bold', rotation=25)

short_labels = ['Classic\nWCT', 'ICP-WCT\n(ours)', 'kNN-VC\n(k=4)', 'LinearVC', 'Soft Local\nWCT (Ours)', 'Local Wass.\n(LWT - Ours)']
ax2.set_xticks(x_pos)
ax2.set_xticklabels(short_labels, fontsize=9.0, fontweight='bold')
ax2.set_ylabel('Metric Values', fontsize=11, fontweight='bold')
ax2.set_title('(b) Multidimensional Profile: Accuracy, Biometrics & Jitter', fontsize=12.5, fontweight='bold', pad=12)
ax2.set_ylim(0, 9.2)
ax2.grid(True, linestyle='--', alpha=0.45, axis='y')
ax2.legend(loc='upper right', fontsize=9.0, framealpha=0.92)

plt.tight_layout()

out_png = Path('/local_scratch/ssadok/un_projet_audio/output/figures/figure_overall_with_lwt_6spk.png')
out_pdf = Path('/local_scratch/ssadok/un_projet_audio/output/figures/figure_overall_with_lwt_6spk.pdf')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()

print(f"Generated clean figure: {out_png}")
print(f"Generated clean PDF: {out_pdf}")
