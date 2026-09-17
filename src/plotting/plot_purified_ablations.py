import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

out_fig_dir = Path('/local_scratch/ssadok/un_projet_audio/output/figures')
out_fig_dir.mkdir(parents=True, exist_ok=True)

# 1. Load Data
df_beta = pd.read_csv('/local_scratch/ssadok/un_projet_audio/output/tables/beta_ablation_6spk.csv')
df_k = pd.read_csv('/local_scratch/ssadok/un_projet_audio/output/tables/continuum_2d_grid_6spk.csv')
df_t = pd.read_csv('/local_scratch/ssadok/un_projet_audio/output/target_budget_ablation_6spk.csv')
df_alpha = pd.read_csv('/local_scratch/ssadok/un_projet_audio/output/tables/speaker_boosted_bures_6spk.csv')

# --- Compute Aggregated Stats ---

# A. Beta Ablation Stats
beta_order = [2.0, 5.0, 10.0, 20.0, 40.0, 100.0, 9999.0]
beta_labels = ['2', '5', '10', '20', '40', '100', r'$\infty$']
beta_stat = df_beta.groupby('Beta_raw').agg({
    'Sim': 'mean',
    'CER': 'mean'
}).loc[beta_order].reset_index()
beta_stat.columns = ['Beta', 'Sim_m', 'CER_m']

# B. K Ablation Stats (at Beta=20.0)
k_sub = df_k[df_k['Beta'] == 20.0]
k_order = [1, 3, 8, 25, 80, 250, 9999]
k_labels = ['1', '3', '8', '25', '80', '250', r'$N_t$']
k_stat = k_sub.groupby('K').agg({
    'Sim': 'mean',
    'CER': 'mean'
}).loc[k_order].reset_index()
k_stat.columns = ['K', 'Sim_m', 'CER_m']

# C. T Ablation Stats (Soft Local WCT vs kNN-VC)
t_ours = df_t[df_t['Method'].str.contains('Ours')].groupby('Duration_sec').agg({
    'Sim': 'mean',
    'CER': 'mean'
}).reset_index()
t_ours.columns = ['Duration', 'Sim_m', 'CER_m']

t_knn = df_t[df_t['Method'].str.contains('kNN')].groupby('Duration_sec').agg({
    'Sim': 'mean',
    'CER': 'mean'
}).reset_index()
t_knn.columns = ['Duration', 'Sim_m', 'CER_m']

# D. Alpha Ablation Stats
alpha_stat = df_alpha.groupby('Alpha').agg({
    'Sim': 'mean',
    'CER': 'mean',
    'Jitter': 'mean'
}).reset_index()
alpha_stat.columns = ['Alpha', 'Sim_m', 'CER_m', 'Jit_m']
alpha_stat = alpha_stat.sort_values('Alpha').reset_index(drop=True)

# -------------------------------------------------------------
# PLOT AESTHETICS (Pure, Minimalist, Modern Scientific Style)
# -------------------------------------------------------------
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#2b2d42'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#8d99ae'
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['grid.linestyle'] = '--'

col_sim = '#1d3557'  # Deep Navy
col_cer = '#e63946'  # Crimson / Coral
col_knn_sim = '#457b9d'
col_knn_cer = '#e76f51'

# =============================================================
# 1. UNIFIED 4-PANEL ABLATION GRID (2 x 2)
# =============================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), dpi=300)

# --- PANEL (a): BETA ABLATION ---
ax_a = axes[0, 0]
x_beta = np.arange(len(beta_labels))

line_a1 = ax_a.plot(x_beta, beta_stat['Sim_m'], color=col_sim, marker='o', linewidth=2.2, markersize=6.5, label='Speaker Sim $\\uparrow$')
ax_a.set_ylabel('Speaker Cosine Sim $\\uparrow$', color=col_sim, fontsize=10.5, fontweight='bold')
ax_a.tick_params(axis='y', labelcolor=col_sim)
ax_a.set_xticks(x_beta)
ax_a.set_xticklabels(beta_labels, fontsize=10)
ax_a.set_xlabel(r'Routing Temperature $\beta$', fontsize=10.5, fontweight='bold')
ax_a.set_title('(a) Routing Temperature Parameter $\\beta$', fontsize=11.5, fontweight='bold', pad=8)
ax_a.grid(True)

ax_a_twin = ax_a.twinx()
line_a2 = ax_a_twin.plot(x_beta, beta_stat['CER_m'], color=col_cer, marker='s', linewidth=2.2, markersize=6.5, label='CER (%) $\\downarrow$')
ax_a_twin.set_ylabel('CER (%) $\\downarrow$', color=col_cer, fontsize=10.5, fontweight='bold')
ax_a_twin.tick_params(axis='y', labelcolor=col_cer)

lines_a = line_a1 + line_a2
labels_a = [l.get_label() for l in lines_a]
ax_a.legend(lines_a, labels_a, loc='center left', framealpha=0.92, fontsize=9)

# --- PANEL (b): K ABLATION ---
ax_b = axes[0, 1]
x_k = np.arange(len(k_labels))

line_b1 = ax_b.plot(x_k, k_stat['Sim_m'], color=col_sim, marker='o', linewidth=2.2, markersize=6.5, label='Speaker Sim $\\uparrow$')
ax_b.set_ylabel('Speaker Cosine Sim $\\uparrow$', color=col_sim, fontsize=10.5, fontweight='bold')
ax_b.tick_params(axis='y', labelcolor=col_sim)
ax_b.set_xticks(x_k)
ax_b.set_xticklabels(k_labels, fontsize=10)
ax_b.set_xlabel('Cluster Cardinality $K$', fontsize=10.5, fontweight='bold')
ax_b.set_title('(b) Number of Local Clusters $K$', fontsize=11.5, fontweight='bold', pad=8)
ax_b.grid(True)

ax_b_twin = ax_b.twinx()
line_b2 = ax_b_twin.plot(x_k, k_stat['CER_m'], color=col_cer, marker='s', linewidth=2.2, markersize=6.5, label='CER (%) $\\downarrow$')
ax_b_twin.set_ylabel('CER (%) $\\downarrow$', color=col_cer, fontsize=10.5, fontweight='bold')
ax_b_twin.tick_params(axis='y', labelcolor=col_cer)

lines_b = line_b1 + line_b2
labels_b = [l.get_label() for l in lines_b]
ax_b.legend(lines_b, labels_b, loc='center left', framealpha=0.92, fontsize=9)

# --- PANEL (c): TARGET BUDGET T ABLATION ---
ax_c = axes[1, 0]
durs = t_ours['Duration'].values

line_c1 = ax_c.plot(durs, t_ours['Sim_m'], color=col_sim, marker='o', linewidth=2.2, markersize=6.5, label='Ours Sim $\\uparrow$')
line_c2 = ax_c.plot(durs, t_knn['Sim_m'], color=col_knn_sim, marker='^', linestyle='--', linewidth=1.8, markersize=6, label='kNN-VC Sim $\\uparrow$')
ax_c.set_ylabel('Speaker Cosine Sim $\\uparrow$', color=col_sim, fontsize=10.5, fontweight='bold')
ax_c.tick_params(axis='y', labelcolor=col_sim)
ax_c.set_xlabel('Target Speech Duration $T$ (seconds)', fontsize=10.5, fontweight='bold')
ax_c.set_title('(c) Target Data Budget $T$', fontsize=11.5, fontweight='bold', pad=8)
ax_c.grid(True)

ax_c_twin = ax_c.twinx()
line_c3 = ax_c_twin.plot(durs, t_ours['CER_m'], color=col_cer, marker='s', linewidth=2.2, markersize=6.5, label='Ours CER (%) $\\downarrow$')
line_c4 = ax_c_twin.plot(durs, t_knn['CER_m'], color=col_knn_cer, marker='v', linestyle='--', linewidth=1.8, markersize=6, label='kNN-VC CER (%) $\\downarrow$')
ax_c_twin.set_ylabel('CER (%) $\\downarrow$', color=col_cer, fontsize=10.5, fontweight='bold')
ax_c_twin.tick_params(axis='y', labelcolor=col_cer)

lines_c = line_c1 + line_c2 + line_c3 + line_c4
labels_c = [l.get_label() for l in lines_c]
ax_c.legend(lines_c, labels_c, loc='center right', framealpha=0.92, fontsize=8.5)

# --- PANEL (d): SPEAKER COVARIANCE BOOST ALPHA ---
ax_d = axes[1, 1]
alphas = alpha_stat['Alpha'].values

line_d1 = ax_d.plot(alphas, alpha_stat['Sim_m'], color=col_sim, marker='o', linewidth=2.2, markersize=6.5, label='Speaker Sim $\\uparrow$')
ax_d.set_ylabel('Speaker Cosine Sim $\\uparrow$', color=col_sim, fontsize=10.5, fontweight='bold')
ax_d.tick_params(axis='y', labelcolor=col_sim)
ax_d.set_xlabel(r'Speaker Covariance Boost Factor $\alpha$', fontsize=10.5, fontweight='bold')
ax_d.set_title(r'(d) Speaker Covariance Boost $\alpha$ in LWT', fontsize=11.5, fontweight='bold', pad=8)
ax_d.grid(True)

ax_d_twin = ax_d.twinx()
line_d2 = ax_d_twin.plot(alphas, alpha_stat['CER_m'], color=col_cer, marker='s', linewidth=2.2, markersize=6.5, label='CER (%) $\\downarrow$')
ax_d_twin.set_ylabel('CER (%) $\\downarrow$', color=col_cer, fontsize=10.5, fontweight='bold')
ax_d_twin.tick_params(axis='y', labelcolor=col_cer)

lines_d = line_d1 + line_d2
labels_d = [l.get_label() for l in lines_d]
ax_d.legend(lines_d, labels_d, loc='center left', framealpha=0.92, fontsize=9)

plt.tight_layout()
grid_png = out_fig_dir / 'figure_ablations_purified_grid.png'
grid_pdf = out_fig_dir / 'figure_ablations_purified_grid.pdf'
plt.savefig(grid_png, dpi=300)
plt.savefig(grid_pdf)
plt.close()
print(f"Generated purified grid: {grid_png} and {grid_pdf}")


# =============================================================
# 2. INDIVIDUAL PURIFIED FIGURES
# =============================================================

# --- Individual Beta ---
fig, ax1 = plt.subplots(figsize=(6.5, 4.0), dpi=300)
l1 = ax1.plot(x_beta, beta_stat['Sim_m'], color=col_sim, marker='o', linewidth=2.2, markersize=6.5, label='Speaker Cosine Sim $\\uparrow$')
ax1.set_ylabel('Speaker Cosine Sim $\\uparrow$', color=col_sim, fontsize=10.5, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=col_sim)
ax1.set_xticks(x_beta)
ax1.set_xticklabels(beta_labels, fontsize=10)
ax1.set_xlabel(r'Routing Temperature $\beta$', fontsize=10.5, fontweight='bold')
ax1.grid(True)

ax2 = ax1.twinx()
l2 = ax2.plot(x_beta, beta_stat['CER_m'], color=col_cer, marker='s', linewidth=2.2, markersize=6.5, label='CER (%) $\\downarrow$')
ax2.set_ylabel('CER (%) $\\downarrow$', color=col_cer, fontsize=10.5, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=col_cer)

lines = l1 + l2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center left', framealpha=0.92, fontsize=9)
plt.title(r'Routing Temperature Ablation: $\beta \in [2, \infty]$', fontsize=11.5, fontweight='bold', pad=8)
plt.tight_layout()
plt.savefig(out_fig_dir / 'figure_ablation_beta_purified.png', dpi=300)
plt.savefig(out_fig_dir / 'figure_ablation_beta_purified.pdf')
plt.close()

# --- Individual K ---
fig, ax1 = plt.subplots(figsize=(6.5, 4.0), dpi=300)
l1 = ax1.plot(x_k, k_stat['Sim_m'], color=col_sim, marker='o', linewidth=2.2, markersize=6.5, label='Speaker Cosine Sim $\\uparrow$')
ax1.set_ylabel('Speaker Cosine Sim $\\uparrow$', color=col_sim, fontsize=10.5, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=col_sim)
ax1.set_xticks(x_k)
ax1.set_xticklabels(k_labels, fontsize=10)
ax1.set_xlabel('Cluster Cardinality $K$', fontsize=10.5, fontweight='bold')
ax1.grid(True)

ax2 = ax1.twinx()
l2 = ax2.plot(x_k, k_stat['CER_m'], color=col_cer, marker='s', linewidth=2.2, markersize=6.5, label='CER (%) $\\downarrow$')
ax2.set_ylabel('CER (%) $\\downarrow$', color=col_cer, fontsize=10.5, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=col_cer)

lines = l1 + l2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center left', framealpha=0.92, fontsize=9)
plt.title('Cluster Cardinality Ablation: $K \\in [1, N_t]$', fontsize=11.5, fontweight='bold', pad=8)
plt.tight_layout()
plt.savefig(out_fig_dir / 'figure_ablation_K_purified.png', dpi=300)
plt.savefig(out_fig_dir / 'figure_ablation_K_purified.pdf')
plt.close()

# --- Individual T ---
fig, ax1 = plt.subplots(figsize=(6.5, 4.0), dpi=300)
l1 = ax1.plot(durs, t_ours['Sim_m'], color=col_sim, marker='o', linewidth=2.2, markersize=6.5, label='Ours Sim $\\uparrow$')
l2 = ax1.plot(durs, t_knn['Sim_m'], color=col_knn_sim, marker='^', linestyle='--', linewidth=1.8, markersize=6, label='kNN-VC Sim $\\uparrow$')
ax1.set_ylabel('Speaker Cosine Sim $\\uparrow$', color=col_sim, fontsize=10.5, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=col_sim)
ax1.set_xlabel('Target Speech Duration $T$ (seconds)', fontsize=10.5, fontweight='bold')
ax1.grid(True)

ax2 = ax1.twinx()
l3 = ax2.plot(durs, t_ours['CER_m'], color=col_cer, marker='s', linewidth=2.2, markersize=6.5, label='Ours CER (%) $\\downarrow$')
l4 = ax2.plot(durs, t_knn['CER_m'], color=col_knn_cer, marker='v', linestyle='--', linewidth=1.8, markersize=6, label='kNN-VC CER (%) $\\downarrow$')
ax2.set_ylabel('CER (%) $\\downarrow$', color=col_cer, fontsize=10.5, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=col_cer)

lines = l1 + l2 + l3 + l4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', framealpha=0.92, fontsize=8.5)
plt.title('Target Speech Budget Adaptation: $T \\in [2s, 80s]$', fontsize=11.5, fontweight='bold', pad=8)
plt.tight_layout()
plt.savefig(out_fig_dir / 'figure_ablation_T_purified.png', dpi=300)
plt.savefig(out_fig_dir / 'figure_ablation_T_purified.pdf')
plt.close()

# --- Individual Alpha ---
fig, ax1 = plt.subplots(figsize=(6.5, 4.0), dpi=300)
l1 = ax1.plot(alphas, alpha_stat['Sim_m'], color=col_sim, marker='o', linewidth=2.2, markersize=6.5, label='Speaker Cosine Sim $\\uparrow$')
ax1.set_ylabel('Speaker Cosine Sim $\\uparrow$', color=col_sim, fontsize=10.5, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=col_sim)
ax1.set_xlabel(r'Speaker Covariance Boost Factor $\alpha$', fontsize=10.5, fontweight='bold')
ax1.grid(True)

ax2 = ax1.twinx()
l2 = ax2.plot(alphas, alpha_stat['CER_m'], color=col_cer, marker='s', linewidth=2.2, markersize=6.5, label='CER (%) $\\downarrow$')
ax2.set_ylabel('CER (%) $\\downarrow$', color=col_cer, fontsize=10.5, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=col_cer)

lines = l1 + l2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center left', framealpha=0.92, fontsize=9)
plt.title(r'Speaker Covariance Boost in LWT: $\alpha \in [0.5, 2.0]$', fontsize=11.5, fontweight='bold', pad=8)
plt.tight_layout()
plt.savefig(out_fig_dir / 'figure_ablation_alpha_purified.png', dpi=300)
plt.savefig(out_fig_dir / 'figure_ablation_alpha_purified.pdf')
plt.close()

print("All purified, unburdened ablation figures generated successfully!")
