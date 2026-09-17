#!/usr/bin/env python3
"""
Generate ultra-clean, minimalist, modern publication figures:
1. Speaker Similarity (ECAPA-TDNN) vs Whisper-Small CER (%) [Ground-Truth Text Reference]
2. Speaker Similarity (ECAPA-TDNN) vs Whisper-Small CER (%) [Source Audio ASR Reference]
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

# Filter out Deep Learning and ICP
# Filter out Deep Learning, ICP, and Soft Local WCT (global source statistics)
exclude = ['FreeVC', 'FreeVC-s', 'SoundStorm', 'ICP-WCT (ours)', 'Soft Local WCT (Ours)']
df = df[~df['Method'].isin(exclude)].copy()

# Modern Minimalist Typography & Styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']
plt.rcParams['mathtext.fontset'] = 'cm'

N_conversions = 200
out_dir = Path('output/figures')
out_dir.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# FIGURE 1: WHISPER CER (GROUND-TRUTH LIBRISPEECH REFERENCE)
# ==============================================================================
fig, ax = plt.subplots(figsize=(8.8, 5.8), dpi=300)

for spine in ax.spines.values():
    spine.set_color('#000000')
    spine.set_linewidth(1.3)

ax.grid(True, linestyle=':', linewidth=0.65, color='#cbd5e1', alpha=0.5, zorder=1)
ax.set_axisbelow(True)

styles_gt = {
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
        'va': 'top',
        'offset': (0.022, -0.006),
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
        'offset': (0.0, 0.009),
        'is_ours': False
    }
}

pareto_df_gt = df[df['Method'].isin(['Classic WCT', 'Local Wasserstein (LWT - Ours)', 'Boosted LWT (alpha=1.5 - Ours)'])].sort_values('CER_Whisper_mean')

ax.fill_between(
    pareto_df_gt['CER_Whisper_mean'],
    0.60,
    pareto_df_gt['Sim_mean'],
    color='#f0f9ff',
    alpha=0.6,
    zorder=1
)

ax.plot(
    pareto_df_gt['CER_Whisper_mean'], pareto_df_gt['Sim_mean'],
    linestyle='--', color='#0284c7', linewidth=1.5, alpha=0.6, zorder=2
)

for _, row in df.iterrows():
    m = row['Method']
    if m not in styles_gt: continue
    st = styles_gt[m]
    x = row['CER_Whisper_mean']
    y = row['Sim_mean']
    x_err = row['CER_Whisper_std'] / np.sqrt(N_conversions)
    y_err = row['Sim_std'] / np.sqrt(N_conversions)
    
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
    
    ax.scatter(
        x, y,
        c=st['color'],
        s=st['size'],
        marker=st['marker'],
        edgecolors='#ffffff',
        linewidths=2.0 if st['is_ours'] else 1.2,
        zorder=st['zorder']
    )
    
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

ax.annotate(
    'Ideal Region (High Sim, Low CER) ↖',
    xy=(1.30, 0.772),
    xytext=(1.30, 0.772),
    fontsize=9.0,
    fontstyle='italic',
    fontweight='bold',
    color='#0284c7',
    zorder=11
)

ax.set_xlim(1.25, 2.55)
ax.set_ylim(0.615, 0.785)

ax.set_xlabel('Character Error Rate (Whisper-Small CER %) $\\leftarrow$ [Lower is better]', fontsize=11, fontweight='bold', color='#000000', labelpad=10)
ax.set_ylabel('Speaker Similarity (ECAPA-TDNN) $\\rightarrow$ [Higher is better]', fontsize=11, fontweight='bold', color='#000000', labelpad=10)

ax.tick_params(axis='both', which='major', colors='#000000', labelsize=10, width=1.1, length=4.5)
ax.tick_params(axis='both', which='minor', colors='#000000', width=0.8, length=2.5)

plt.tight_layout()
png_gt = out_dir / 'figure_pareto_whisper_cer_vs_sim_clean.png'
pdf_gt = out_dir / 'figure_pareto_whisper_cer_vs_sim_clean.pdf'
plt.savefig(png_gt, dpi=300)
plt.savefig(pdf_gt)
plt.close()
print(f"Generated GT figure: {png_gt}")

# ==============================================================================
# FIGURE 2: WHISPER CER (SOURCE ASR REFERENCE)
# ==============================================================================
fig, ax = plt.subplots(figsize=(8.8, 5.8), dpi=300)

for spine in ax.spines.values():
    spine.set_color('#000000')
    spine.set_linewidth(1.3)

ax.grid(True, linestyle=':', linewidth=0.65, color='#cbd5e1', alpha=0.5, zorder=1)
ax.set_axisbelow(True)

styles_src = {
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
        'offset': (0.020, 0.008),
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
        'offset': (0.0, 0.009),
        'is_ours': False
    }
}

pareto_df_src = df[df['Method'].isin(['Local Wasserstein (LWT - Ours)', 'Boosted LWT (alpha=1.5 - Ours)'])].sort_values('CER_Whisper_src_mean')

ax.fill_between(
    pareto_df_src['CER_Whisper_src_mean'],
    0.60,
    pareto_df_src['Sim_mean'],
    color='#f0f9ff',
    alpha=0.6,
    zorder=1
)

ax.plot(
    pareto_df_src['CER_Whisper_src_mean'], pareto_df_src['Sim_mean'],
    linestyle='--', color='#0284c7', linewidth=1.5, alpha=0.6, zorder=2
)

for _, row in df.iterrows():
    m = row['Method']
    if m not in styles_src: continue
    st = styles_src[m]
    x = row['CER_Whisper_src_mean']
    y = row['Sim_mean']
    x_err = row['CER_Whisper_src_std'] / np.sqrt(N_conversions)
    y_err = row['Sim_std'] / np.sqrt(N_conversions)
    
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
    
    ax.scatter(
        x, y,
        c=st['color'],
        s=st['size'],
        marker=st['marker'],
        edgecolors='#ffffff',
        linewidths=2.0 if st['is_ours'] else 1.2,
        zorder=st['zorder']
    )
    
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

ax.annotate(
    'Ideal Region (High Sim, Low CER) ↖',
    xy=(0.42, 0.772),
    xytext=(0.42, 0.772),
    fontsize=9.0,
    fontstyle='italic',
    fontweight='bold',
    color='#0284c7',
    zorder=11
)

ax.set_xlim(0.38, 1.80)
ax.set_ylim(0.615, 0.785)

ax.set_xlabel('Character Error Rate (Whisper-Small vs Source CER %) $\\leftarrow$ [Lower is better]', fontsize=11, fontweight='bold', color='#000000', labelpad=10)
ax.set_ylabel('Speaker Similarity (ECAPA-TDNN) $\\rightarrow$ [Higher is better]', fontsize=11, fontweight='bold', color='#000000', labelpad=10)

ax.tick_params(axis='both', which='major', colors='#000000', labelsize=10, width=1.1, length=4.5)
ax.tick_params(axis='both', which='minor', colors='#000000', width=0.8, length=2.5)

plt.tight_layout()
png_src = out_dir / 'figure_pareto_whisper_src_cer_vs_sim_clean.png'
pdf_src = out_dir / 'figure_pareto_whisper_src_cer_vs_sim_clean.pdf'
plt.savefig(png_src, dpi=300)
plt.savefig(pdf_src)
plt.close()
print(f"Generated Source-Ref figure: {png_src}")
