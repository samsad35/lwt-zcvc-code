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

# Aesthetic Styling: Modern, Sleek, Purified
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#2b2d42'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#8d99ae'
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['grid.linestyle'] = '--'

col_sim = '#1d3557'  # Deep Navy
col_cer = '#e63946'  # Crimson / Coral

# 1. Unified 4-Panel Grid (2 x 2)
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), dpi=300)

# --- Panel (a): Beta Ablation ---
ax_a = axes[0, 0]
x_beta = np.arange(len(df_beta))
labels_beta = [str(int(b)) if b == int(b) else str(b) for b in df_beta['Beta']]

line_a1 = ax_a.plot(x_beta, df_beta['Sim'], color=col_sim, marker='o', linewidth=2.2, markersize=6.5, label='Speaker Sim $\\uparrow$')
ax_a.set_ylabel('Speaker Cosine Sim $\\uparrow$', color=col_sim, fontsize=10.5, fontweight='bold')
ax_a.tick_params(axis='y', labelcolor=col_sim)
ax_a.set_xticks(x_beta)
ax_a.set_xticklabels(labels_beta, fontsize=10)
ax_a.set_xlabel(r'Routing Temperature $\beta$', fontsize=10.5, fontweight='bold')
ax_a.set_title('(a) Routing Temperature Parameter $\\beta$', fontsize=11.5, fontweight='bold', pad=8)
ax_a.grid(True)

ax_a_twin = ax_a.twinx()
line_a2 = ax_a_twin.plot(x_beta, df_beta['CER'], color=col_cer, marker='s', linewidth=2.2, markersize=6.5, label='CER (%) $\\downarrow$')
ax_a_twin.set_ylabel('CER (%) $\\downarrow$', color=col_cer, fontsize=10.5, fontweight='bold')
ax_a_twin.tick_params(axis='y', labelcolor=col_cer)

lines_a = line_a1 + line_a2
ax_a.legend(lines_a, [l.get_label() for l in lines_a], loc='center left', framealpha=0.92, fontsize=9)

# --- Panel (b): K Ablation ---
ax_b = axes[0, 1]
x_k = np.arange(len(df_k))
labels_k = [str(int(k)) for k in df_k['K']]

line_b1 = ax_b.plot(x_k, df_k['Sim'], color=col_sim, marker='o', linewidth=2.2, markersize=6.5, label='Speaker Sim $\\uparrow$')
ax_b.set_ylabel('Speaker Cosine Sim $\\uparrow$', color=col_sim, fontsize=10.5, fontweight='bold')
ax_b.tick_params(axis='y', labelcolor=col_sim)
ax_b.set_xticks(x_k)
ax_b.set_xticklabels(labels_k, fontsize=10)
ax_b.set_xlabel('Cluster Cardinality $K$', fontsize=10.5, fontweight='bold')
ax_b.set_title('(b) Number of Local Clusters $K$', fontsize=11.5, fontweight='bold', pad=8)
ax_b.grid(True)

ax_b_twin = ax_b.twinx()
line_b2 = ax_b_twin.plot(x_k, df_k['CER'], color=col_cer, marker='s', linewidth=2.2, markersize=6.5, label='CER (%) $\\downarrow$')
ax_b_twin.set_ylabel('CER (%) $\\downarrow$', color=col_cer, fontsize=10.5, fontweight='bold')
ax_b_twin.tick_params(axis='y', labelcolor=col_cer)

lines_b = line_b1 + line_b2
ax_b.legend(lines_b, [l.get_label() for l in lines_b], loc='center right', framealpha=0.92, fontsize=9)

# --- Panel (c): Target Budget T Ablation ---
ax_c = axes[1, 0]
durs = df_t['Duration_sec'].values

line_c1 = ax_c.plot(durs, df_t['Sim'], color=col_sim, marker='o', linewidth=2.2, markersize=6.5, label='Speaker Sim $\\uparrow$')
ax_c.set_ylabel('Speaker Cosine Sim $\\uparrow$', color=col_sim, fontsize=10.5, fontweight='bold')
ax_c.tick_params(axis='y', labelcolor=col_sim)
ax_c.set_xlabel('Target Speech Duration $T$ (seconds)', fontsize=10.5, fontweight='bold')
ax_c.set_title('(c) Target Data Budget $T$', fontsize=11.5, fontweight='bold', pad=8)
ax_c.grid(True)

ax_c_twin = ax_c.twinx()
line_c2 = ax_c_twin.plot(durs, df_t['CER'], color=col_cer, marker='s', linewidth=2.2, markersize=6.5, label='CER (%) $\\downarrow$')
ax_c_twin.set_ylabel('CER (%) $\\downarrow$', color=col_cer, fontsize=10.5, fontweight='bold')
ax_c_twin.tick_params(axis='y', labelcolor=col_cer)

lines_c = line_c1 + line_c2
ax_c.legend(lines_c, [l.get_label() for l in lines_c], loc='center right', framealpha=0.92, fontsize=9)

# --- Panel (d): Speaker Covariance Boost Alpha ---
ax_d = axes[1, 1]
alphas = df_alpha['Alpha'].values

line_d1 = ax_d.plot(alphas, df_alpha['Sim'], color=col_sim, marker='o', linewidth=2.2, markersize=6.5, label='Speaker Sim $\\uparrow$')
ax_d.set_ylabel('Speaker Cosine Sim $\\uparrow$', color=col_sim, fontsize=10.5, fontweight='bold')
ax_d.tick_params(axis='y', labelcolor=col_sim)
ax_d.set_xlabel(r'Speaker Covariance Boost Factor $\alpha$', fontsize=10.5, fontweight='bold')
ax_d.set_title(r'(d) Speaker Covariance Boost $\alpha$ in LWT', fontsize=11.5, fontweight='bold', pad=8)
ax_d.grid(True)

ax_d_twin = ax_d.twinx()
line_d2 = ax_d_twin.plot(alphas, df_alpha['CER'], color=col_cer, marker='s', linewidth=2.2, markersize=6.5, label='CER (%) $\\downarrow$')
ax_d_twin.set_ylabel('CER (%) $\\downarrow$', color=col_cer, fontsize=10.5, fontweight='bold')
ax_d_twin.tick_params(axis='y', labelcolor=col_cer)

lines_d = line_d1 + line_d2
ax_d.legend(lines_d, [l.get_label() for l in lines_d], loc='upper left', framealpha=0.92, fontsize=9)

plt.tight_layout()
grid_png = out_fig_dir / 'figure_lwt_ablations_clean_grid.png'
grid_pdf = out_fig_dir / 'figure_lwt_ablations_clean_grid.pdf'
plt.savefig(grid_png, dpi=300)
plt.savefig(grid_pdf)
plt.close()
print(f"Generated clean LWT ablation grid: {grid_png} and {grid_pdf}")

# 2. Individual Clean Figures with customized non-overlapping legends
def save_indiv(x_vals, y_sim, y_cer, x_label, title, fname, loc_leg='best', is_xticks=False, tick_labels=None):
    fig, ax1 = plt.subplots(figsize=(6.5, 4.0), dpi=300)
    l1 = ax1.plot(x_vals, y_sim, color=col_sim, marker='o', linewidth=2.2, markersize=6.5, label='Speaker Cosine Sim $\\uparrow$')
    ax1.set_ylabel('Speaker Cosine Sim $\\uparrow$', color=col_sim, fontsize=10.5, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=col_sim)
    if is_xticks and tick_labels is not None:
        ax1.set_xticks(x_vals)
        ax1.set_xticklabels(tick_labels, fontsize=10)
    ax1.set_xlabel(x_label, fontsize=10.5, fontweight='bold')
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    l2 = ax2.plot(x_vals, y_cer, color=col_cer, marker='s', linewidth=2.2, markersize=6.5, label='CER (%) $\\downarrow$')
    ax2.set_ylabel('CER (%) $\\downarrow$', color=col_cer, fontsize=10.5, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=col_cer)
    
    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], loc=loc_leg, framealpha=0.92, fontsize=9)
    plt.title(title, fontsize=11.5, fontweight='bold', pad=8)
    plt.tight_layout()
    plt.savefig(out_fig_dir / f"{fname}.png", dpi=300)
    plt.savefig(out_fig_dir / f"{fname}.pdf")
    plt.close()

save_indiv(x_beta, df_beta['Sim'], df_beta['CER'], r'Routing Temperature $\beta$', r'LWT Routing Sharpness Ablation: $\beta$', 'figure_lwt_ablation_beta_clean', 'center left', True, labels_beta)
save_indiv(x_k, df_k['Sim'], df_k['CER'], 'Cluster Cardinality $K$', 'LWT Cluster Cardinality Ablation: $K$', 'figure_lwt_ablation_K_clean', 'center right', True, labels_k)
save_indiv(durs, df_t['Sim'], df_t['CER'], 'Target Speech Duration $T$ (seconds)', 'LWT Target Budget Adaptation: $T$', 'figure_lwt_ablation_T_clean', 'center right')
save_indiv(alphas, df_alpha['Sim'], df_alpha['CER'], r'Speaker Covariance Boost Factor $\alpha$', r'LWT Speaker Covariance Boost: $\alpha$', 'figure_lwt_ablation_alpha_clean', 'upper left')

print("All individual clean LWT ablation figures re-generated with optimal legend placements!")
