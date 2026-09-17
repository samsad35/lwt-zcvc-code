import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Scientific styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'

input_csv = Path('/local_scratch/ssadok/un_projet_audio/output/tables/jacobian_norms_vs_beta_6spk.csv')
if not input_csv.exists():
    print(f"File {input_csv} does not exist.")
    sys.exit(1)

df = pd.read_csv(input_csv)

num_cols = ['Beta', 'J_geom_mean', 'J_geom_std', 'J_routing_mean', 'J_routing_std', 
            'J_ratio_mean', 'Relative_Jitter', 'Entropy', 'CER', 'Sim']
mean_df = df[num_cols].groupby('Beta').mean().reset_index()
std_df = df[num_cols].groupby('Beta').std().reset_index()

betas = mean_df['Beta'].values
j_geom_m = mean_df['J_geom_mean'].values
j_geom_s = std_df['J_geom_mean'].values
j_rout_m = mean_df['J_routing_mean'].values
j_rout_s = std_df['J_routing_mean'].values
ratio_m  = mean_df['J_ratio_mean'].values
ratio_s  = std_df['J_ratio_mean'].values
jit_m    = mean_df['Relative_Jitter'].values
jit_s    = std_df['Relative_Jitter'].values
sim_m    = mean_df['Sim'].values
sim_s    = std_df['Sim'].values
cer_m    = mean_df['CER'].values
cer_s    = std_df['CER'].values
ent_m    = mean_df['Entropy'].values

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.5, 5.2), dpi=300)

# Professional publication palette
c_geom  = '#1f77b4'  # Deep Steel Blue
c_rout  = '#d62728'  # Crimson
c_ratio = '#7b3294'  # Rich Purple
c_jit   = '#e66101'  # Vibrant Amber
c_cer   = '#ca0020'  # Deep Carmine
c_sim   = '#0571b0'  # Classic Royal Blue

# ==============================================================================
# Panel 1: Theoretical Terms ||J_geom||_F vs ||J_routing||_F
# ==============================================================================
ax1.plot(betas, j_geom_m, marker='o', markersize=6.5, linewidth=2.4, color=c_geom, 
         label=r'$\|J_{\mathrm{geom}}\|_F$ (Local Covariance)')
ax1.fill_between(betas, j_geom_m - j_geom_s, j_geom_m + j_geom_s, color=c_geom, alpha=0.15)

ax1.plot(betas, j_rout_m, marker='s', markersize=6.5, linewidth=2.4, color=c_rout, 
         label=r'$\|J_{\mathrm{routing}}\|_F$ (Dynamic Routing)')
ax1.fill_between(betas, j_rout_m - j_rout_s, j_rout_m + j_rout_s, color=c_rout, alpha=0.15)

ax1_twin = ax1.twinx()
ax1_twin.plot(betas, ratio_m, marker='^', markersize=5.5, linewidth=2.0, color=c_ratio, linestyle='--',
              label=r'Ratio $\|J_{\mathrm{rout}}\| / \|J_{\mathrm{geom}}\|$')
ax1_twin.fill_between(betas, ratio_m - ratio_s, ratio_m + ratio_s, color=c_ratio, alpha=0.10)
ax1_twin.set_ylabel(r'Norm Ratio $\|J_{\mathrm{rout}}\|_F / \|J_{\mathrm{geom}}\|_F$', color=c_ratio, fontsize=11, fontweight='bold')
ax1_twin.tick_params(axis='y', labelcolor=c_ratio)
ax1_twin.set_ylim(-0.01, 0.42)

# Highlight sweet spot
ax1.axvspan(10, 25, color='#2ca02c', alpha=0.12, label=r'Optimal Zone ($\beta \in [10, 25]$)')

ax1.set_xscale('log')
ax1.set_ylim(-0.2, 9.8)
ax1.set_xlabel(r'Softmax Temperature $\beta$ (log scale)', fontsize=11, fontweight='bold')
ax1.set_ylabel(r'Frobenius Norm $\|\cdot\|_F$', fontsize=11, fontweight='bold')
ax1.set_title(r'(a) Jacobian Terms Decomposition', fontsize=12.5, fontweight='bold', pad=12)
ax1.grid(True, linestyle='--', alpha=0.45, which='both')

# Combine legends for ax1
l1, lab1 = ax1.get_legend_handles_labels()
l2, lab2 = ax1_twin.get_legend_handles_labels()
ax1.legend(l1 + l2, lab1 + lab2, loc='upper left', fontsize=9.0, framealpha=0.92)


# ==============================================================================
# Panel 2: Physical Trajectory Velocity & Continuity (Relative Jitter)
# ==============================================================================
ax2.plot(betas, jit_m, marker='o', markersize=6.5, linewidth=2.5, color=c_jit, 
         label=r'Soft Local WCT ($\frac{\|\Delta \hat{x}\|}{\|\Delta x_{\mathrm{src}}\|}$)')
ax2.fill_between(betas, jit_m - jit_s, jit_m + jit_s, color=c_jit, alpha=0.15)

# Reference horizontal lines
ax2.axhline(1.11, color='#8c564b', linestyle=':', linewidth=2.2, label=r'kNN-VC Reference ($1.11\times$)')
ax2.axhline(1.00, color='gray', linestyle='--', linewidth=1.3, alpha=0.85, label=r'Natural Source Velocity ($1.00\times$)')
ax2.axhline(0.66, color='#1f77b4', linestyle='-.', linewidth=1.3, alpha=0.85, label=r'Global WCT ($0.66\times$)')

ax2.axvspan(10, 25, color='#2ca02c', alpha=0.12, label=r'Optimal Smooth Zone ($\beta \approx 15-20$)')

# Annotations
ax2.annotate('Optimal Flow\n(Natural, $\\approx 0.96\\times$)',
             xy=(20, jit_m[np.where(betas==20)[0][0]]),
             xytext=(4.2, 0.80),
             arrowprops=dict(arrowstyle='->', lw=1.6, color=c_jit),
             fontsize=9.2, fontweight='bold', color='#b35400',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#fff5eb', edgecolor=c_jit, alpha=0.92))

ax2.annotate('Voronoi Boundary Jumps\n(Jitter Surges $> 1.19\\times$)',
             xy=(150, jit_m[np.where(betas==150)[0][0]]),
             xytext=(28, 1.25),
             arrowprops=dict(arrowstyle='->', lw=1.6, color='#8c564b'),
             fontsize=9.2, fontweight='bold', color='#8c564b',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#fbf0f0', edgecolor='#8c564b', alpha=0.92))

ax2.set_xscale('log')
ax2.set_ylim(0.30, 1.36)
ax2.set_xlabel(r'Softmax Temperature $\beta$ (log scale)', fontsize=11, fontweight='bold')
ax2.set_ylabel(r'Relative Trajectory Jitter $\frac{\|\Delta \hat{x}\|}{\|\Delta x_{\mathrm{src}}\|}$', fontsize=11, fontweight='bold')
ax2.set_title(r'(b) Physical Continuity vs. Boundary Jumps', fontsize=12.5, fontweight='bold', pad=12)
ax2.grid(True, linestyle='--', alpha=0.45, which='both')
ax2.legend(loc='lower right', fontsize=8.8, framealpha=0.92)


# ==============================================================================
# Panel 3: Perceptual Metrics (CER & Speaker Similarity)
# ==============================================================================
ax3_twin = ax3.twinx()

p_cer = ax3.plot(betas, cer_m, marker='s', markersize=6.5, linewidth=2.4, color=c_cer, label='CER (%) [Lower is better]')
ax3.fill_between(betas, cer_m - cer_s, cer_m + cer_s, color=c_cer, alpha=0.15)

p_sim = ax3_twin.plot(betas, sim_m, marker='D', markersize=5.5, linewidth=2.4, color=c_sim, label='Speaker Sim. [Higher is better]')
ax3_twin.fill_between(betas, sim_m - sim_s, sim_m + sim_s, color=c_sim, alpha=0.15)

ax3.axvspan(10, 25, color='#2ca02c', alpha=0.12)

ax3.set_xscale('log')
ax3.set_ylim(0.2, 2.7)
ax3_twin.set_ylim(0.32, 0.74)
ax3.set_xlabel(r'Softmax Temperature $\beta$ (log scale)', fontsize=11, fontweight='bold')
ax3.set_ylabel(r'Character Error Rate (CER %)', color=c_cer, fontsize=11, fontweight='bold')
ax3_twin.set_ylabel(r'Speaker Cosine Similarity', color=c_sim, fontsize=11, fontweight='bold')
ax3.tick_params(axis='y', labelcolor=c_cer)
ax3_twin.tick_params(axis='y', labelcolor=c_sim)

ax3.set_title(r'(c) Trade-off on Intelligibility & Identity', fontsize=12.5, fontweight='bold', pad=12)
ax3.grid(True, linestyle='--', alpha=0.45, which='both')

# Optimal operating point annotation cleanly positioned at x=15, y=1.25 inside open space
ax3.annotate('Pareto Sweet Spot\nCER $0.55\\%$, Sim $0.675$',
             xy=(15, cer_m[np.where(betas==15)[0][0]]),
             xytext=(25, 1.25),
             arrowprops=dict(arrowstyle='->', lw=1.6, color='#2ca02c'),
             fontsize=9.2, fontweight='bold', color='#1e7120',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#eefbee', edgecolor='#2ca02c', alpha=0.92))

# Unified legend for ax3
l3, lab3 = ax3.get_legend_handles_labels()
l4, lab4 = ax3_twin.get_legend_handles_labels()
ax3.legend(l3 + l4, lab3 + lab4, loc='upper right', fontsize=8.8, framealpha=0.92)

plt.tight_layout()

# Save outputs
fig_png = Path('/local_scratch/ssadok/un_projet_audio/output/figures/figure_jacobian_norms_vs_beta.png')
fig_pdf = Path('/local_scratch/ssadok/un_projet_audio/output/figures/figure_jacobian_norms_vs_beta.pdf')
plt.savefig(fig_png, dpi=300, bbox_inches='tight')
plt.savefig(fig_pdf, bbox_inches='tight')
plt.close()

print(f"Generated clean figure: {fig_png}")
print(f"Generated clean PDF: {fig_pdf}")
