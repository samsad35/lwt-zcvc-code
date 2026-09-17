#!/usr/bin/env python3
"""
Generate an ultra-clean, minimalist, modern publication figure:
Speaker Similarity vs Character Error Rate (CER %)
Without Deep Learning, without ICP, without title, without legend.
Solid black frame, polished scatter styling, direct typography.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# 1. Load Data
csv_path = Path('output/tables/unified_200_benchmark_summary.csv')
df = pd.read_csv(csv_path)

# Filter out Deep Learning, ICP, and Soft Local WCT (global source statistics)
exclude = ['FreeVC', 'FreeVC-s', 'SoundStorm', 'ICP-WCT (ours)', 'Soft Local WCT (Ours)']
df = df[~df['Method'].isin(exclude)].copy()
df = df.sort_values('CER_mean').reset_index(drop=True)

# 2. Modern Minimalist Typography & Styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']
plt.rcParams['mathtext.fontset'] = 'cm'

fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=300)

# Black Frame (solid, crisp)
for spine in ax.spines.values():
    spine.set_color('#000000')
    spine.set_linewidth(1.3)

# Very soft, subtle grid
ax.grid(True, linestyle=':', linewidth=0.65, color='#cbd5e1', alpha=0.5, zorder=1)
ax.set_axisbelow(True)

# Modern, harmonious palette
styles = {
    'Local WCT (Local Source Stats)': {
        'label': 'Local WCT (Ours)',
        'color': '#d97706',  # Warm Amber
        'marker': 'p',
        'size': 145,
        'zorder': 6,
        'ha': 'center',
        'va': 'top',
        'offset': (0.0, -0.013),
        'is_ours': True
    },
    'Local Wasserstein (LWT - Ours)': {
        'label': 'LWT (Ours)',
        'color': '#0284c7',  # Electric Ocean Blue
        'marker': 'o',
        'size': 160,
        'zorder': 7,
        'ha': 'right',
        'va': 'bottom',
        'offset': (-0.020, 0.007),
        'is_ours': True
    },
    'Boosted LWT (alpha=1.5 - Ours)': {
        'label': 'Boosted LWT (Ours)',
        'color': '#e11d48',  # Vivid Crimson Rose
        'marker': 's',
        'size': 150,
        'zorder': 8,
        'ha': 'center',
        'va': 'bottom',
        'offset': (0.0, 0.009),
        'is_ours': True
    },
    'kNN-VC (k=4)': {
        'label': 'kNN-VC (k=4)',
        'color': '#1e293b',  # Deep Slate
        'marker': '^',
        'size': 130,
        'zorder': 5,
        'ha': 'left',
        'va': 'bottom',
        'offset': (0.020, 0.008),
        'is_ours': False
    },
    'Classic WCT': {
        'label': 'WCT',
        'color': '#64748b',  # Muted Slate
        'marker': 'o',
        'size': 95,
        'zorder': 4,
        'ha': 'center',
        'va': 'bottom',
        'offset': (0.0, 0.008),
        'is_ours': False
    },
    'LinearVC': {
        'label': 'LinearVC',
        'color': '#475569',  # Graphite
        'marker': 'P',
        'size': 105,
        'zorder': 4,
        'ha': 'left',
        'va': 'bottom',
        'offset': (0.020, 0.007),
        'is_ours': False
    },
    'kNN-VC (k=1)': {
        'label': 'kNN-VC (k=1)',
        'color': '#94a3b8',  # Cool Gray
        'marker': 'v',
        'size': 95,
        'zorder': 4,
        'ha': 'center',
        'va': 'bottom',
        'offset': (0.0, 0.008),
        'is_ours': False
    }
}

# 3. Pareto Optimal Frontier
pareto_methods = [
    'Local Wasserstein (LWT - Ours)',
    'Boosted LWT (alpha=1.5 - Ours)'
]
pareto_df = df[df['Method'].isin(pareto_methods)].sort_values('CER_mean')

# Soft shaded region under Pareto frontier
ax.fill_between(
    pareto_df['CER_mean'],
    0.60,
    pareto_df['Sim_mean'],
    color='#f0f9ff',
    alpha=0.6,
    zorder=1
)

ax.plot(
    pareto_df['CER_mean'], pareto_df['Sim_mean'],
    linestyle='--', color='#0284c7', linewidth=1.5, alpha=0.6, zorder=2
)

# 4. Plot Points and Error Bars (SEM, N=200)
N_conversions = 200
for _, row in df.iterrows():
    m = row['Method']
    if m not in styles:
        continue
    st = styles[m]
    x = row['CER_mean']
    y = row['Sim_mean']
    x_err = row['CER_std'] / np.sqrt(N_conversions)
    y_err = row['Sim_std'] / np.sqrt(N_conversions)
    
    # Clean Error Bars
    ax.errorbar(
        x, y,
        xerr=x_err, yerr=y_err,
        fmt='none',
        ecolor=st['color'],
        elinewidth=1.6 if st['is_ours'] else 1.2,
        capsize=3.0,
        capthick=1.2,
        alpha=0.85,
        zorder=st['zorder'] - 1
    )
    
    # Dual-ring Scatter Point (High-end publication look)
    ax.scatter(
        x, y,
        c=st['color'],
        s=st['size'],
        marker=st['marker'],
        edgecolors='#ffffff',
        linewidths=2.0 if st['is_ours'] else 1.2,
        zorder=st['zorder']
    )
    
    # Direct Typography Label
    dx, dy = st['offset']
    weight = 'bold' if st['is_ours'] else 'semibold'
    fsize = 9.5 if st['is_ours'] else 8.5
    fcolor = '#0f172a' if st['is_ours'] else '#475569'
    
    import matplotlib.patheffects as pe
    ax.annotate(
        st['label'],
        xy=(x, y),
        xytext=(x + dx, y + dy),
        ha=st.get('ha', 'left'),
        va=st.get('va', 'center'),
        fontsize=fsize,
        fontweight=weight,
        color=fcolor,
        zorder=10,
        path_effects=[pe.withStroke(linewidth=2.5, foreground='white')]
    )

# 5. Minimalist "Ideal" Direction Marker
ax.annotate(
    'Ideal Region (High Sim, Low CER) ↖',
    xy=(0.40, 0.772),
    xytext=(0.40, 0.772),
    fontsize=9.0,
    fontstyle='italic',
    fontweight='bold',
    color='#0284c7',
    zorder=11
)

# 6. Axis Ranges & Labels (NO Title, NO Legend)
ax.set_xlim(0.35, 1.95)
ax.set_ylim(0.615, 0.785)

ax.set_xlabel('Character Error Rate (CER %) $\\leftarrow$ [Lower is better]', fontsize=11, fontweight='bold', color='#000000', labelpad=10)
ax.set_ylabel('Speaker Similarity (ECAPA-TDNN) $\\rightarrow$ [Higher is better]', fontsize=11, fontweight='bold', color='#000000', labelpad=10)

# Ticks styling with black frame
ax.tick_params(axis='both', which='major', colors='#000000', labelsize=10, width=1.1, length=4.5)
ax.tick_params(axis='both', which='minor', colors='#000000', width=0.8, length=2.5)

plt.tight_layout()

# Save
out_dir = Path('output/figures')
out_dir.mkdir(parents=True, exist_ok=True)
png_path = out_dir / 'figure_pareto_cer_vs_sim_clean.png'
pdf_path = out_dir / 'figure_pareto_cer_vs_sim_clean.pdf'

plt.savefig(png_path, dpi=300)
plt.savefig(pdf_path)
plt.close()

print(f"Generated ultra-clean Pareto figure:\n  - {png_path}\n  - {pdf_path}")
