import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

out_fig_dir = Path('/local_scratch/ssadok/un_projet_audio/output/figures')
out_fig_dir.mkdir(parents=True, exist_ok=True)

f_beta = Path('/local_scratch/ssadok/un_projet_audio/output/tables/lwt_ablation_beta_large_N.csv')
f_k = Path('/local_scratch/ssadok/un_projet_audio/output/tables/lwt_ablation_K_large_N.csv')
f_t = Path('/local_scratch/ssadok/un_projet_audio/output/tables/lwt_ablation_T_large_N.csv')
f_alpha = Path('/local_scratch/ssadok/un_projet_audio/output/tables/lwt_ablation_alpha_large_N.csv')

df_beta = pd.read_csv(f_beta)
df_k = pd.read_csv(f_k)
df_t = pd.read_csv(f_t)
df_alpha = pd.read_csv(f_alpha)

# Publication Styling: Clean, Modern, Compact
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#2b2d42'
plt.rcParams['axes.linewidth'] = 0.85
plt.rcParams['grid.color'] = '#8d99ae'
plt.rcParams['grid.alpha'] = 0.22
plt.rcParams['grid.linestyle'] = '--'

col_sim = '#1d3557'  # Deep Navy Blue
col_cer = '#e63946'  # Crimson Red

# Shared y-limits across all 4 subplots
sim_ylim = (0.50, 0.785)
cer_ylim = (0.35, 3.35)

fig, axes = plt.subplots(1, 4, figsize=(15.8, 3.4), dpi=300, sharey=True)

# -------------------------------------------------------------------------
# Subplot 1: Panel (a) Beta Temperature
# -------------------------------------------------------------------------
ax1 = axes[0]
x_beta = np.arange(len(df_beta))
labels_beta = [str(int(b)) if b == int(b) else str(b) for b in df_beta['Beta']]

line1_sim = ax1.plot(x_beta, df_beta['Sim'], color=col_sim, marker='o', linewidth=2.0, markersize=5.0, label='Speaker Cosine Sim $\\uparrow$')
ax1.set_ylim(sim_ylim)
ax1.set_ylabel('Speaker Cosine Sim $\\uparrow$', color=col_sim, fontsize=10.5, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=col_sim, labelsize=9.0)
ax1.set_xticks(x_beta)
ax1.set_xticklabels(labels_beta, fontsize=8.5, rotation=35)
ax1.set_xlabel(r'Routing Sharpness $\beta$', fontsize=10.0, fontweight='bold')
ax1.set_title(r'(a) Routing Sharpness $\beta$', fontsize=11, fontweight='bold', pad=8)
ax1.grid(True)

ax1_r = ax1.twinx()
line1_cer = ax1_r.plot(x_beta, df_beta['CER'], color=col_cer, marker='s', linewidth=2.0, markersize=5.0, label='CER (%) $\\downarrow$')
ax1_r.set_ylim(cer_ylim)
ax1_r.tick_params(axis='y', labelright=False, right=False)  # hide right ticks entirely

# -------------------------------------------------------------------------
# Subplot 2: Panel (b) Number of Clusters K
# -------------------------------------------------------------------------
ax2 = axes[1]
x_k = np.arange(len(df_k))
labels_k = [str(int(k)) for k in df_k['K']]

line2_sim = ax2.plot(x_k, df_k['Sim'], color=col_sim, marker='o', linewidth=2.0, markersize=5.0)
ax2.set_ylim(sim_ylim)
ax2.tick_params(axis='y', labelleft=False, left=False)  # hide left ticks entirely
ax2.set_xticks(x_k)
ax2.set_xticklabels(labels_k, fontsize=8.5, rotation=35)
ax2.set_xlabel('Clusters $K$', fontsize=10.0, fontweight='bold')
ax2.set_title('(b) Number of Clusters $K$', fontsize=11, fontweight='bold', pad=8)
ax2.grid(True)

ax2_r = ax2.twinx()
line2_cer = ax2_r.plot(x_k, df_k['CER'], color=col_cer, marker='s', linewidth=2.0, markersize=5.0)
ax2_r.set_ylim(cer_ylim)
ax2_r.tick_params(axis='y', labelright=False, right=False)  # hide right ticks entirely

# -------------------------------------------------------------------------
# Subplot 3: Panel (c) Target Budget T
# -------------------------------------------------------------------------
ax3 = axes[2]
durs = df_t['Duration_sec'].values

line3_sim = ax3.plot(durs, df_t['Sim'], color=col_sim, marker='o', linewidth=2.0, markersize=5.0)
ax3.set_ylim(sim_ylim)
ax3.tick_params(axis='y', labelleft=False, left=False)  # hide left ticks entirely
ax3.set_xlabel('Duration $T$ (seconds)', fontsize=10.0, fontweight='bold')
ax3.set_title('(c) Target Speech $T$', fontsize=11, fontweight='bold', pad=8)
ax3.tick_params(axis='x', labelsize=8.5)
ax3.grid(True)

ax3_r = ax3.twinx()
line3_cer = ax3_r.plot(durs, df_t['CER'], color=col_cer, marker='s', linewidth=2.0, markersize=5.0)
ax3_r.set_ylim(cer_ylim)
ax3_r.tick_params(axis='y', labelright=False, right=False)  # hide right ticks entirely

# -------------------------------------------------------------------------
# Subplot 4: Panel (d) Speaker Boost Alpha
# -------------------------------------------------------------------------
ax4 = axes[3]
alphas = df_alpha['Alpha'].values

line4_sim = ax4.plot(alphas, df_alpha['Sim'], color=col_sim, marker='o', linewidth=2.0, markersize=5.0)
ax4.set_ylim(sim_ylim)
ax4.tick_params(axis='y', labelleft=False, left=False)  # hide left ticks entirely
ax4.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
ax4.tick_params(axis='x', labelsize=8.5)
ax4.set_xlabel(r'Boost Factor $\alpha$', fontsize=10.0, fontweight='bold')
ax4.set_title(r'(d) Covariance Boost $\alpha$', fontsize=11, fontweight='bold', pad=8)
ax4.grid(True)

ax4_r = ax4.twinx()
line4_cer = ax4_r.plot(alphas, df_alpha['CER'], color=col_cer, marker='s', linewidth=2.0, markersize=5.0)
ax4_r.set_ylim(cer_ylim)
ax4_r.set_ylabel('CER (%) $\\downarrow$', color=col_cer, fontsize=10.5, fontweight='bold')
ax4_r.tick_params(axis='y', labelcolor=col_cer, labelsize=9.0)

# -------------------------------------------------------------------------
# Global Unified Legend (Top Center)
# -------------------------------------------------------------------------
legend_lines = [line1_sim[0], line1_cer[0]]
legend_labels = ['Speaker Cosine Sim $\\uparrow$', 'ASR CER (%) $\\downarrow$']
fig.legend(
    legend_lines, legend_labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.065),
    ncol=2,
    frameon=True,
    framealpha=0.95,
    edgecolor='#8d99ae',
    fontsize=10.5,
    handletextpad=0.6,
    columnspacing=2.5
)

plt.subplots_adjust(top=0.85, bottom=0.20, left=0.065, right=0.935, wspace=0.12)

out_1x4_png = out_fig_dir / 'figure_lwt_ablations_1x4_compact.png'
out_1x4_pdf = out_fig_dir / 'figure_lwt_ablations_1x4_compact.pdf'
plt.savefig(out_1x4_png, dpi=300, bbox_inches='tight')
plt.savefig(out_1x4_pdf, bbox_inches='tight')
plt.close()
print(f"Generated refined compact 1x4 unified ablation figure: {out_1x4_png} and {out_1x4_pdf}")
