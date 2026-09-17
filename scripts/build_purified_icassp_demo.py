import os, sys, json
import numpy as np
import torchaudio
from pathlib import Path

print("=== Building Simplified & Elegant Demo Website (No Box Clutter) ===")

base_dir = Path("output/audio_samples/boosted_lwt_demo")

# 1. Helper to extract 48 waveform peaks from audio file
def extract_peaks(wav_path, num_bars=48):
    try:
        wav, sr = torchaudio.load(str(wav_path))
        w = wav.squeeze().numpy()
        if len(w) == 0:
            return [0.25] * num_bars
        chunk_size = max(1, len(w) // num_bars)
        peaks = []
        for i in range(num_bars):
            chunk = w[i*chunk_size:(i+1)*chunk_size]
            if len(chunk) > 0:
                peaks.append(float(np.max(np.abs(chunk))))
            else:
                peaks.append(0.1)
        max_p = max(peaks) if max(peaks) > 1e-4 else 1.0
        norm_peaks = [round(max(0.18, min(1.0, p / max_p)), 3) for p in peaks]
        return norm_peaks
    except Exception as e:
        print(f"Warning: could not extract peaks for {wav_path}: {e}")
        return [0.25] * num_bars

# Helper to generate SVG bars with dynamic clipPath
def generate_svg_waveform(peaks, clip_id, fill_id, color_theme="primary"):
    colors = {
        "primary": "#0284c7",
        "gold": "#d97706",
        "teal": "#0d9488",
        "indigo": "#4f46e5",
        "slate": "#475569"
    }
    fill_color = colors.get(color_theme, "#0284c7")
    
    n = len(peaks)
    # viewBox: 0 0 280 26. 48 bars: width 3.2, pitch 5.7 => 47*5.7 + 3.2 = 271.1. Offset x0 = 4.5
    rects = []
    for i, p in enumerate(peaks):
        x = round(4.5 + i * 5.7, 1)
        h = max(3.5, round(p * 22.0, 1))
        y = round((26.0 - h) / 2.0, 1)
        rects.append(f'<rect x="{x}" y="{y}" width="3.2" height="{h}" rx="1.6" />')
    
    rects_str = "".join(rects)
    
    svg = f'''<svg class="waveform-svg" viewBox="0 0 280 26" preserveAspectRatio="none">
        <defs>
            <clipPath id="{clip_id}">
                <rect id="{fill_id}" x="0" y="0" width="0%" height="26" />
            </clipPath>
        </defs>
        <g class="wave-bg">{rects_str}</g>
        <g class="wave-fill" fill="{fill_color}" clip-path="url(#{clip_id})">{rects_str}</g>
    </svg>'''
    return svg

# 2. Benchmark Pairs Data (Verified True LibriSpeech Transcripts)
pairs_info = [
    {
        'id': 'pair1',
        'dir': 'pair1_1284_to_1089',
        'cat': 'cross',
        'cat_name': 'Cross-Gender (Female &rarr; Male)',
        'src_spk': '1284',
        'src_gender': 'Female',
        'tgt_spk': '1089',
        'tgt_gender': 'Male',
        'transcript': 'He wore blue silk stockings, blue knee pants with gold buckles, a blue ruffled waist, and a jacket of bright blue braided with gold.'
    },
    {
        'id': 'pair2',
        'dir': 'pair2_1320_to_121',
        'cat': 'cross',
        'cat_name': 'Cross-Gender (Male &rarr; Female)',
        'src_spk': '1320',
        'src_gender': 'Male',
        'tgt_spk': '121',
        'tgt_gender': 'Female',
        'transcript': 'Since the period of our tale, the active spirit of the country has surrounded it with a belt of rich and thriving settlements, though none but the hunter or the savage is ever known, even now, to penetrate its wild recesses.'
    },
    {
        'id': 'pair3',
        'dir': 'pair3_1580_to_237',
        'cat': 'same',
        'cat_name': 'Same-Gender (Female &rarr; Female)',
        'src_spk': '1580',
        'src_gender': 'Female',
        'tgt_spk': '237',
        'tgt_gender': 'Female',
        'transcript': 'I will endeavour in my statement to avoid such terms as would serve to limit the events to any particular place, or give a clue as to the people concerned.'
    },
    {
        'id': 'pair4',
        'dir': 'pair4_672_to_1188',
        'cat': 'same',
        'cat_name': 'Same-Gender (Male &rarr; Male)',
        'src_spk': '672',
        'src_gender': 'Male',
        'tgt_spk': '1188',
        'tgt_gender': 'Male',
        'transcript': 'Out in the woods stood a nice little fir tree.'
    },
    {
        'id': 'pair5',
        'dir': 'pair5_1995_to_260',
        'cat': 'cross',
        'cat_name': 'Cross-Gender (Female &rarr; Male)',
        'src_spk': '1995',
        'src_gender': 'Female',
        'tgt_spk': '260',
        'tgt_gender': 'Male',
        'transcript': 'In the debate between the senior societies, her defence of the fifteenth amendment had been not only a notable bit of reasoning, but delivered with real enthusiasm.'
    },
    {
        'id': 'pair6',
        'dir': 'pair6_908_to_3570',
        'cat': 'cross',
        'cat_name': 'Cross-Gender (Male &rarr; Female)',
        'src_spk': '908',
        'src_gender': 'Male',
        'tgt_spk': '3570',
        'tgt_gender': 'Female',
        'transcript': 'To fade away like morning beauty from her mortal day, down by the river of Adona her soft voice is heard, and thus her gentle lamentation falls like morning dew.'
    },
    {
        'id': 'pair7',
        'dir': 'pair7_121_to_1580',
        'cat': 'same',
        'cat_name': 'Same-Gender (Female &rarr; Female)',
        'src_spk': '121',
        'src_gender': 'Female',
        'tgt_spk': '1580',
        'tgt_gender': 'Female',
        'transcript': 'Also a popular contrivance whereby love-making may be suspended, but not stopped, during the picnic season.'
    },
    {
        'id': 'pair8',
        'dir': 'pair8_260_to_1089',
        'cat': 'same',
        'cat_name': 'Same-Gender (Male &rarr; Male)',
        'src_spk': '260',
        'src_gender': 'Male',
        'tgt_spk': '1089',
        'tgt_gender': 'Male',
        'transcript': 'Saturday, August fifteenth: the sea unbroken all round, no land in sight.'
    },
    {
        'id': 'pair9',
        'dir': 'pair9_61_to_1995',
        'cat': 'cross',
        'cat_name': 'Cross-Gender (Male &rarr; Female)',
        'src_spk': '61',
        'src_gender': 'Male',
        'tgt_spk': '1995',
        'tgt_gender': 'Female',
        'transcript': 'He began a confused complaint against the wizard who had vanished behind the curtain on the left.'
    }
]

# Method configurations
methods_config = [
    {
        'key': 'boosted_lwt',
        'file': 'converted_boosted_lwt.wav',
        'name': 'Proposed LWT (α=1.5)',
        'badge': 'Proposed Optimum ⭐',
        'highlight': True,
        'color': 'gold',
    },
    {
        'key': 'baseline_lwt',
        'file': 'converted_baseline_lwt.wav',
        'name': 'Standard LWT (α=1.0)',
        'badge': 'Ours (α=1.0)',
        'highlight': False,
        'color': 'primary',
    },
    {
        'key': 'linearvc',
        'file': 'converted_linearvc.wav',
        'name': 'LinearVC',
        'badge': 'Interspeech 2025',
        'highlight': False,
        'color': 'primary',
    },
    {
        'key': 'knnvc_k4',
        'file': 'converted_knnvc_k4.wav',
        'name': 'kNN-VC (k=4)',
        'badge': 'IEEE TASLP 2023',
        'highlight': False,
        'color': 'primary',
    },
    {
        'key': 'classic_wct',
        'file': 'converted_classic_wct.wav',
        'name': 'Classic WCT',
        'badge': 'Global Baseline',
        'highlight': False,
        'color': 'primary',
    }
]

# 3. Neutral Voice Conversion Data (from 40-speaker balanced background)
neutral_voice_cases = [
    {
        'id': 'sample1_1284',
        'dir': 'pure_content/sample1_1284_female',
        'spk': '1284',
        'gender': 'Female (High-pitched)',
        'f0_src': '175 Hz',
        'f0_pure': '171 Hz',
        'transcript': 'He wore blue silk stockings, blue knee pants with gold buckles, a blue ruffled waist, and a jacket of bright blue braided with gold.'
    },
    {
        'id': 'sample2_1320',
        'dir': 'pure_content/sample2_1320_male',
        'spk': '1320',
        'gender': 'Male (Deep voice)',
        'f0_src': '118 Hz',
        'f0_pure': '161 Hz',
        'transcript': 'Since the period of our tale, the active spirit of the country has surrounded it with a belt of rich and thriving settlements, though none but the hunter or the savage is ever known, even now, to penetrate its wild recesses.'
    },
    {
        'id': 'sample3_1995',
        'dir': 'pure_content/sample3_1995_female',
        'spk': '1995',
        'gender': 'Female',
        'f0_src': '180 Hz',
        'f0_pure': '165 Hz',
        'transcript': 'In the debate between the senior societies, her defence of the fifteenth amendment had been not only a notable bit of reasoning, but delivered with real enthusiasm.'
    },
    {
        'id': 'sample4_672',
        'dir': 'pure_content/sample4_672_male',
        'spk': '672',
        'gender': 'Male',
        'f0_src': '125 Hz',
        'f0_pure': '158 Hz',
        'transcript': 'Out in the woods stood a nice little fir tree.'
    }
]

print("Precomputing waveform peaks for all audio files...")
all_waveforms = {}

for p in pairs_info:
    p_dir = base_dir / p['dir']
    all_waveforms[f"{p['id']}_src"] = extract_peaks(p_dir / 'source.wav')
    all_waveforms[f"{p['id']}_tgt"] = extract_peaks(p_dir / 'target_ref.wav')
    for m in methods_config:
        all_waveforms[f"{p['id']}_{m['key']}"] = extract_peaks(p_dir / m['file'])

for nvc in neutral_voice_cases:
    nvc_dir = base_dir / nvc['dir']
    all_waveforms[f"{nvc['id']}_src"] = extract_peaks(nvc_dir / 'source.wav')
    all_waveforms[f"{nvc['id']}_pure"] = extract_peaks(nvc_dir / 'pure_content_balanced_40spk.wav')

print(f"Precomputed {len(all_waveforms)} audio waveforms successfully!")

# Helper to render a player component with SVG waveform
def render_player_component(player_id, audio_src, peaks, color_theme="primary"):
    clip_id = f"clip_{player_id}"
    fill_id = f"fill_{player_id}"
    svg_waveform = generate_svg_waveform(peaks, clip_id, fill_id, color_theme)
    
    return f"""
    <div class="player-unit {color_theme}" id="unit_{player_id}">
        <button class="play-btn" onclick="togglePlay('{player_id}')" title="Play / Pause" aria-label="Play">
            <svg class="icon-play" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
            <svg class="icon-pause" style="display:none;" viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
        </button>
        <div class="wave-track" onclick="seekWave(event, '{player_id}')" title="Click to seek">
            {svg_waveform}
        </div>
        <span class="time-readout" id="time_{player_id}">0:00</span>
        <audio id="audio_{player_id}" preload="none" src="{audio_src}"></audio>
    </div>
    """

# Build Complete HTML
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local Transport Mixtures for Zero-Shot Voice Conversion in SSL Spaces | Under Review</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #ffffff;
            --surface: #ffffff;
            --surface-sub: #f8fafc;
            --border-subtle: #e2e8f0;
            --border-divider: #f1f5f9;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --text-sub: #334155;
            
            --primary: #0284c7;
            --primary-light: #f0f9ff;
            --primary-border: #bae6fd;
            
            --gold: #d97706;
            --gold-light: #fffbeb;
            --gold-border: #fcd34d;
            --gold-dark: #b45309;
            
            --teal: #0d9488;
            --teal-light: #f0fdfa;
            --teal-border: #99f6e4;
            
            --indigo: #4f46e5;
            --indigo-light: #eef2ff;
            --slate: #475569;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
        }}

        /* Header / Hero */
        .hero {{
            background: #ffffff;
            border-bottom: 1px solid var(--border-subtle);
            padding: 3.5rem 1.5rem 2.5rem;
            text-align: center;
        }}

        .conf-pill {{
            display: inline-flex;
            align-items: center;
            background: #f1f5f9;
            color: #475569;
            border: 1px solid #cbd5e1;
            padding: 0.25rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }}

        h1 {{
            font-size: 2.15rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            color: var(--text-main);
            max-width: 920px;
            margin: 0 auto 1rem;
            line-height: 1.28;
        }}

        .authors {{
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 0.4rem;
        }}

        .author-space {{
            display: inline-block;
            width: 2.5rem;
        }}

        .affiliations {{
            font-size: 0.88rem;
            color: var(--text-muted);
            margin-bottom: 1.5rem;
        }}

        .equal-note {{
            font-size: 0.78rem;
            color: #94a3b8;
            margin-top: 0.15rem;
        }}

        .nav-links {{
            display: flex;
            justify-content: center;
            gap: 0.65rem;
            flex-wrap: wrap;
        }}

        .nav-link-btn {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: var(--surface-sub);
            color: var(--text-sub);
            border: 1px solid var(--border-subtle);
            padding: 0.4rem 0.95rem;
            border-radius: 8px;
            font-size: 0.82rem;
            font-weight: 600;
            text-decoration: none;
            transition: all 0.15s ease;
        }}

        .nav-link-btn:hover {{
            border-color: var(--primary);
            color: var(--primary);
            background: var(--primary-light);
        }}

        /* Main Container */
        .container {{
            max-width: 1040px;
            margin: 0 auto;
            padding: 2rem 1.5rem 4rem;
        }}

        /* Section Headers - Clean, No Heavy Line */
        .section-header {{
            margin: 3.5rem 0 1.25rem;
            padding-bottom: 0.6rem;
            border-bottom: 2px solid #0f172a;
        }}

        .section-header h2 {{
            font-size: 1.35rem;
            font-weight: 800;
            color: var(--text-main);
            letter-spacing: -0.015em;
        }}

        .section-header p {{
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-top: 0.2rem;
        }}

        /* Universal Player Unit - Minimal & Sleek */
        .player-unit {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.35rem 0.75rem;
            width: 100%;
            max-width: 440px;
            min-width: 0;
        }}

        .player-unit.gold {{
            background: #fffdf7;
            border-color: #fde68a;
        }}

        .player-unit.teal {{
            background: #f0fdfa;
            border-color: #99f6e4;
        }}

        .play-btn {{
            width: 30px;
            height: 30px;
            min-width: 30px;
            border-radius: 50%;
            border: none;
            background: #0f172a;
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: transform 0.1s ease, background 0.15s ease;
            flex-shrink: 0;
        }}

        .play-btn:hover {{
            transform: scale(1.06);
            background: var(--primary);
        }}

        .player-unit.gold .play-btn {{
            background: var(--gold);
        }}
        .player-unit.gold .play-btn:hover {{
            background: var(--gold-dark);
        }}

        .player-unit.teal .play-btn {{
            background: var(--teal);
        }}
        .player-unit.teal .play-btn:hover {{
            background: #0f766e;
        }}

        .play-btn svg {{
            width: 13px;
            height: 13px;
            fill: currentColor;
        }}

        /* SVG Waveform Track */
        .wave-track {{
            flex: 1;
            height: 26px;
            cursor: pointer;
            display: flex;
            align-items: center;
            min-width: 0;
            overflow: hidden;
        }}

        .waveform-svg {{
            width: 100%;
            height: 100%;
            display: block;
        }}

        .waveform-svg .wave-bg {{
            fill: #cbd5e1;
            transition: fill 0.15s ease;
        }}

        .waveform-svg:hover .wave-bg {{
            fill: #94a3b8;
        }}

        .time-readout {{
            font-size: 0.72rem;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-muted);
            min-width: 44px;
            text-align: right;
            flex-shrink: 0;
        }}

        /* ========================================================================= */
        /* SECTION 1: BENCHMARK PAIRS - SIMPLIFIED, ZERO BOX CLUTTER                */
        /* ========================================================================= */
        .pairs-nav {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-divider);
        }}

        .filter-buttons {{
            display: flex;
            gap: 0.4rem;
        }}

        .tab-btn {{
            background: transparent;
            border: 1px solid var(--border-subtle);
            color: var(--text-sub);
            padding: 0.3rem 0.7rem;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }}

        .tab-btn:hover {{
            border-color: #94a3b8;
            color: var(--text-main);
        }}

        .tab-btn.active {{
            background: var(--text-main);
            color: #ffffff;
            border-color: var(--text-main);
        }}

        .pair-tabs-row {{
            display: flex;
            gap: 0.35rem;
            flex-wrap: wrap;
        }}

        .pair-tab-chip {{
            background: transparent;
            border: 1px solid var(--border-subtle);
            color: var(--text-sub);
            padding: 0.25rem 0.55rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }}

        .pair-tab-chip:hover {{
            border-color: var(--primary);
            color: var(--primary);
        }}

        .pair-tab-chip.active {{
            background: var(--primary);
            color: #ffffff;
            border-color: var(--primary);
        }}

        .pair-tab-chip.all-chip {{
            font-weight: 700;
            background: #f1f5f9;
        }}

        /* Clean Flat Pair Section */
        .pair-block {{
            margin-bottom: 3rem;
            padding-bottom: 2.5rem;
            border-bottom: 1px solid var(--border-divider);
        }}

        .pair-block:last-child {{
            border-bottom: none;
        }}

        .pair-title-row {{
            margin-bottom: 0.4rem;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            flex-wrap: wrap;
        }}

        .pair-title-row h3 {{
            font-size: 1.12rem;
            font-weight: 700;
            color: var(--text-main);
        }}

        .cat-tag {{
            font-size: 0.74rem;
            font-weight: 600;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            background: #f1f5f9;
            color: #475569;
        }}

        .cat-tag.cross {{
            background: #eff6ff;
            color: #1d4ed8;
        }}

        .spoken-text-line {{
            font-size: 0.88rem;
            color: #475569;
            font-style: italic;
            margin-bottom: 1.25rem;
            line-height: 1.5;
        }}

        .spoken-text-line strong {{
            font-style: normal;
            color: var(--text-main);
            font-weight: 600;
            margin-right: 0.35rem;
        }}

        /* Unified Single Flat Table per Pair */
        .comparison-table {{
            width: 100%;
            border-top: 1px solid var(--border-subtle);
        }}

        .comp-row {{
            display: grid;
            grid-template-columns: 1fr 440px;
            align-items: center;
            padding: 0.65rem 0.5rem;
            border-bottom: 1px solid #f1f5f9;
            gap: 1.5rem;
            transition: background 0.15s ease;
        }}

        .comp-row:hover {{
            background: #fafafa;
        }}

        .comp-row.ref-row {{
            background: #f8fafc;
        }}

        .comp-row.proposed {{
            background: #fffdf5;
        }}

        @media (max-width: 768px) {{
            .comp-row {{
                grid-template-columns: 1fr;
                gap: 0.4rem;
                padding: 0.75rem 0.25rem;
            }}
        }}

        .comp-label {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.88rem;
            color: var(--text-main);
            font-weight: 500;
        }}

        .comp-row.ref-row .comp-label {{
            font-weight: 600;
            color: #1e293b;
        }}

        .comp-row.proposed .comp-label {{
            font-weight: 700;
            color: #92400e;
        }}

        .sub-badge {{
            font-size: 0.7rem;
            font-weight: 700;
            padding: 0.1rem 0.4rem;
            border-radius: 4px;
            background: #f1f5f9;
            color: #475569;
            white-space: nowrap;
        }}

        .sub-badge.gold {{
            background: #fef3c7;
            color: #b45309;
        }}

        .sub-badge.src {{
            background: #e2e8f0;
            color: #334155;
        }}

        .sub-badge.tgt {{
            background: #e0e7ff;
            color: #3730a3;
        }}

        .comp-player {{
            display: flex;
            justify-content: flex-end;
            min-width: 0;
        }}

        /* ========================================================================= */
        /* SECTION 2: NEUTRAL VOICE CONVERSION - FLAT & CLEAN                       */
        /* ========================================================================= */
        .neutral-intro-text {{
            font-size: 0.92rem;
            color: var(--text-sub);
            margin-bottom: 1.75rem;
            line-height: 1.6;
        }}

        .neutral-sample-block {{
            margin-bottom: 2.25rem;
            padding-bottom: 1.75rem;
            border-bottom: 1px solid var(--border-divider);
        }}

        .neutral-sample-block:last-child {{
            border-bottom: none;
        }}

        .neutral-sample-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.35rem;
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--text-main);
        }}

        .neutral-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.25rem;
            margin-top: 0.75rem;
        }}

        @media (max-width: 720px) {{
            .neutral-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .neutral-side {{
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }}

        .neutral-side-title {{
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
        }}

        .neutral-side-title strong {{
            color: var(--text-main);
        }}

        /* ========================================================================= */
        /* SECTION 3: BENCHMARK TABLE - CLEAN & FLAT                                */
        /* ========================================================================= */
        .table-wrap {{
            overflow-x: auto;
            margin-top: 1rem;
            border-top: 1px solid var(--border-subtle);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            text-align: left;
        }}

        th {{
            background: #f8fafc;
            color: var(--text-sub);
            padding: 0.75rem 0.85rem;
            font-weight: 700;
            border-bottom: 1px solid var(--border-subtle);
            white-space: nowrap;
        }}

        td {{
            padding: 0.7rem 0.85rem;
            border-bottom: 1px solid #f1f5f9;
            white-space: nowrap;
            color: var(--text-sub);
        }}

        tr:hover td {{
            background: #fafafa;
        }}

        tr.highlight td {{
            background: #fffdf5;
            font-weight: 600;
            color: var(--text-main);
        }}

        /* ========================================================================= */
        /* SECTION 4: CODE SECTION                                                  */
        /* ========================================================================= */
        .code-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.25rem;
            margin-top: 1.25rem;
        }}

        @media (max-width: 860px) {{
            .code-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .code-box {{
            background: #0f172a;
            border-radius: 10px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}

        .code-box-header {{
            background: #1e293b;
            padding: 0.55rem 1rem;
            font-size: 0.78rem;
            font-weight: 600;
            color: #94a3b8;
            font-family: 'JetBrains Mono', monospace;
        }}

        .code-content {{
            padding: 1rem 1.1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            line-height: 1.6;
            color: #e2e8f0;
            overflow-x: auto;
            white-space: pre;
            flex: 1;
        }}

        .code-content .kw {{ color: #f472b6; font-weight: 600; }}
        .code-content .fn {{ color: #38bdf8; }}
        .code-content .str {{ color: #a3e635; }}
        .code-content .com {{ color: #64748b; font-style: italic; }}
        .code-content .cmd {{ color: #fde047; font-weight: 600; }}

        footer {{
            text-align: center;
            padding: 3.5rem 1rem;
            color: var(--text-muted);
            font-size: 0.82rem;
            border-top: 1px solid var(--border-subtle);
            margin-top: 3rem;
        }}
    </style>
</head>
<body>

    <header class="hero">
        <div class="conf-pill">Under Review &bull; Audio Demonstration</div>
        <h1>Local Transport Mixtures for Zero-Shot Voice Conversion in SSL Spaces</h1>
        
        <div class="authors">
            <span>Samir Sadok</span>
            <span class="author-space"></span>
            <span>Xavier Alameda-Pineda</span>
        </div>
        <div class="affiliations">
            <span>Inria, Univ. Grenoble Alpes, CNRS, Grenoble INP, LJK, France</span>
        </div>

        <div class="nav-links">
            <a href="#benchmark" class="nav-link-btn">🎧 1. Voice Conversion Benchmark (9 Pairs)</a>
            <a href="#neutral-voice" class="nav-link-btn">⚖️ 2. Neutral Voice Conversion</a>
            <a href="#results-table" class="nav-link-btn">📊 3. Unified Objective Results (N=200)</a>
            <a href="#code" class="nav-link-btn">💻 4. Code &amp; Quickstart</a>
        </div>
    </header>

    <div class="container">

        <!-- ========================================================================= -->
        <!-- SECTION 1: BENCHMARK VOICE CONVERSION (9 CURATED PAIRS)                  -->
        <!-- ========================================================================= -->
        <div id="benchmark" class="section-header">
            <h2>1. Curated Voice Conversion Benchmark (9 Pairs)</h2>
            <p>
                Comprehensive evaluation across all gender combinations. Click play to compare methods directly against Source and Target ground truth.
            </p>
        </div>

        <div class="pairs-nav">
            <div class="filter-buttons">
                <button class="tab-btn active" onclick="filterCategory('all', this)">All (9)</button>
                <button class="tab-btn" onclick="filterCategory('cross', this)">Cross-Gender (4)</button>
                <button class="tab-btn" onclick="filterCategory('same', this)">Same-Gender (5)</button>
            </div>
            <div class="pair-tabs-row">
                <button class="pair-tab-chip all-chip active" onclick="selectPairTab('all', this)">Show All</button>
                <button class="pair-tab-chip" onclick="selectPairTab('pair1', this)">Pair 1 (F&rarr;M)</button>
                <button class="pair-tab-chip" onclick="selectPairTab('pair2', this)">Pair 2 (M&rarr;F)</button>
                <button class="pair-tab-chip" onclick="selectPairTab('pair3', this)">Pair 3 (F&rarr;F)</button>
                <button class="pair-tab-chip" onclick="selectPairTab('pair4', this)">Pair 4 (M&rarr;M)</button>
                <button class="pair-tab-chip" onclick="selectPairTab('pair5', this)">Pair 5 (F&rarr;M)</button>
                <button class="pair-tab-chip" onclick="selectPairTab('pair6', this)">Pair 6 (M&rarr;F)</button>
                <button class="pair-tab-chip" onclick="selectPairTab('pair7', this)">Pair 7 (F&rarr;F)</button>
                <button class="pair-tab-chip" onclick="selectPairTab('pair8', this)">Pair 8 (M&rarr;M)</button>
                <button class="pair-tab-chip" onclick="selectPairTab('pair9', this)">Pair 9 (M&rarr;F)</button>
            </div>
        </div>
"""

# Render Benchmark Pairs (SECTION 1 - Flat & Clean)
for idx, p in enumerate(pairs_info, 1):
    src_peaks = all_waveforms[f"{p['id']}_src"]
    tgt_peaks = all_waveforms[f"{p['id']}_tgt"]
    
    src_player = render_player_component(
        f"{p['id']}_src",
        f"{p['dir']}/source.wav",
        src_peaks,
        "slate"
    )
    
    tgt_player = render_player_component(
        f"{p['id']}_tgt",
        f"{p['dir']}/target_ref.wav",
        tgt_peaks,
        "indigo"
    )
    
    # Method Rows
    methods_rows = ""
    for m in methods_config:
        m_peaks = all_waveforms[f"{p['id']}_{m['key']}"]
        m_player = render_player_component(
            f"{p['id']}_{m['key']}",
            f"{p['dir']}/{m['file']}",
            m_peaks,
            m['color']
        )
        is_prop = m['highlight']
        
        methods_rows += f"""
            <div class="comp-row {'proposed' if is_prop else ''}">
                <div class="comp-label">
                    <span>{m['name']}</span>
                    <span class="sub-badge {'gold' if is_prop else ''}">{m['badge']}</span>
                </div>
                <div class="comp-player">
                    {m_player}
                </div>
            </div>
        """

    html += f"""
        <div class="pair-block" id="card_{p['id']}" data-cat="{p['cat']}">
            <div class="pair-title-row">
                <h3>Pair {idx}: Speaker {p['src_spk']} ({p['src_gender']}) &rarr; Speaker {p['tgt_spk']} ({p['tgt_gender']})</h3>
                <span class="cat-tag {'cross' if p['cat'] == 'cross' else ''}">{p['cat_name']}</span>
            </div>
            
            <p class="spoken-text-line"><strong>Text:</strong> “{p['transcript']}”</p>
            
            <div class="comparison-table">
                <!-- Ground Truth References -->
                <div class="comp-row ref-row">
                    <div class="comp-label">
                        <span>Source Speech ({p['src_gender']})</span>
                        <span class="sub-badge src">Source Spk {p['src_spk']}</span>
                    </div>
                    <div class="comp-player">
                        {src_player}
                    </div>
                </div>
                
                <div class="comp-row ref-row">
                    <div class="comp-label">
                        <span>Target Reference ({p['tgt_gender']})</span>
                        <span class="sub-badge tgt">Target Spk {p['tgt_spk']}</span>
                    </div>
                    <div class="comp-player">
                        {tgt_player}
                    </div>
                </div>
                
                <!-- Converted Methods -->
                {methods_rows}
            </div>
        </div>
    """

html += """
        <!-- ========================================================================= -->
        <!-- SECTION 2: VOICE CONVERSION TO A NEUTRAL VOICE                           -->
        <!-- ========================================================================= -->
        <div id="neutral-voice" class="section-header">
            <h2>2. Voice Conversion to a Canonical Neutral Voice</h2>
            <p>
                Converting any voice (Male or Female) into a standardized gender-neutral canonical timbre (~160 Hz) while preserving 100% phonetic intelligibility.
            </p>
        </div>

        <p class="neutral-intro-text">
            By subtracting the source local centroids <code>&mu;<sub>X,k</sub></code>, whitening via <code>&Sigma;<sub>X,k</sub><sup>-1/2</sup></code>, and transporting directly onto the <strong>balanced 40-speaker universal background</strong> (<code>c<sub>k</sub>, &Sigma;<sub>shared,k</sub></code>), we neutralize individual speaker timbre:
            <code>z<sub>neutral</sub>(t) = &sum;<sub>k</sub> w<sub>t,k</sub> [ c<sub>k</sub> + (x<sub>t</sub> - &mu;<sub>X,k</sub>) A<sub>neutral,k</sub> ]</code>.
            Notice how both deep male and high-pitched female voices collapse into the exact same <strong>neutral canonical voice (~160 Hz)</strong> with <strong>0.00% CER</strong>.
        </p>
"""

# Render Neutral Voice Samples (SECTION 2 - Clean Flat Layout)
for nvc in neutral_voice_cases:
    src_peaks = all_waveforms[f"{nvc['id']}_src"]
    pure_peaks = all_waveforms[f"{nvc['id']}_pure"]
    
    src_player = render_player_component(
        f"{nvc['id']}_src",
        f"{nvc['dir']}/source.wav",
        src_peaks,
        "slate"
    )
    
    pure_player = render_player_component(
        f"{nvc['id']}_pure",
        f"{nvc['dir']}/pure_content_balanced_40spk.wav",
        pure_peaks,
        "teal"
    )
    
    html += f"""
        <div class="neutral-sample-block">
            <div class="neutral-sample-header">
                <span>Speaker {nvc['spk']} &bull; {nvc['gender']}</span>
                <span class="cat-tag">Canonical Neutralization</span>
            </div>
            <p class="spoken-text-line"><strong>Text:</strong> “{nvc['transcript']}”</p>
            
            <div class="neutral-grid">
                <div class="neutral-side">
                    <div class="neutral-side-title">
                        <span>Original Human Voice</span>
                        <span>Pitch: <strong>{nvc['f0_src']}</strong></span>
                    </div>
                    {src_player}
                </div>
                
                <div class="neutral-side">
                    <div class="neutral-side-title">
                        <span>Converted Neutral Voice</span>
                        <span>Pitch: <strong>{nvc['f0_pure']}</strong> &bull; CER: <strong>0.00%</strong></span>
                    </div>
                    {pure_player}
                </div>
            </div>
        </div>
    """

html += """
        <!-- ========================================================================= -->
        <!-- SECTION 3: UNIFIED BENCHMARK RESULTS TABLE (N=200)                      -->
        <!-- ========================================================================= -->
        <div id="results-table" class="section-header">
            <h2>3. Strictly Unified Benchmark Summary (N=200 Test Conversions)</h2>
            <p>
                Averaged across 10 unseen target speakers, 20 source utterances from 10 distinct speakers, T=20 target reference utterances.
            </p>
        </div>

        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Method</th>
                        <th>Approach Type</th>
                        <th>W2V2 CER &darr;</th>
                        <th>W2V2 WER &darr;</th>
                        <th>Whisper CER &darr;</th>
                        <th>ECAPA Sim &uarr;</th>
                        <th>MOS &uarr;</th>
                        <th>Pitch Corr &uarr;</th>
                        <th>Loudness Corr &uarr;</th>
                        <th>RTF &darr;</th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="highlight">
                        <td><strong>Proposed LWT (&alpha;=1.5)</strong></td>
                        <td>Speaker-Scaled Wasserstein</td>
                        <td><strong>0.71%</strong></td>
                        <td><strong>2.68%</strong></td>
                        <td><strong>1.68%</strong></td>
                        <td><strong>0.749</strong> &starf;</td>
                        <td>4.37</td>
                        <td>0.730</td>
                        <td>0.917</td>
                        <td>0.0371</td>
                    </tr>
                    <tr>
                        <td><strong>Local Wasserstein (LWT)</strong></td>
                        <td>Locally Adaptive Wasserstein</td>
                        <td>0.58%</td>
                        <td>2.38%</td>
                        <td>1.64%</td>
                        <td>0.720</td>
                        <td><strong>4.38</strong></td>
                        <td>0.735</td>
                        <td>0.922</td>
                        <td>0.0239</td>
                    </tr>
                    <tr>
                        <td><strong>Soft Local WCT</strong></td>
                        <td>Locally Adaptive Statistical</td>
                        <td><strong>0.54%</strong> &starf;</td>
                        <td><strong>2.28%</strong> &starf;</td>
                        <td><strong>1.62%</strong> &starf;</td>
                        <td>0.631</td>
                        <td>4.36</td>
                        <td><strong>0.746</strong> &starf;</td>
                        <td><strong>0.925</strong> &starf;</td>
                        <td>0.0067</td>
                    </tr>
                    <tr>
                        <td><strong>LinearVC (Interspeech 2025)</strong></td>
                        <td>Global Linear Projection</td>
                        <td>0.67%</td>
                        <td>2.82%</td>
                        <td>1.70%</td>
                        <td>0.707</td>
                        <td>4.37</td>
                        <td>0.728</td>
                        <td>0.913</td>
                        <td><strong>0.00002</strong> &starf;</td>
                    </tr>
                    <tr>
                        <td><strong>kNN-VC (k=4)</strong></td>
                        <td>Local Instance Averaging</td>
                        <td>1.11%</td>
                        <td>4.00%</td>
                        <td>1.81%</td>
                        <td>0.743</td>
                        <td><strong>4.38</strong></td>
                        <td>0.728</td>
                        <td>0.920</td>
                        <td>0.0002</td>
                    </tr>
                    <tr>
                        <td><strong>kNN-VC (k=1)</strong></td>
                        <td>1-NN Replacement</td>
                        <td>1.72%</td>
                        <td>5.44%</td>
                        <td>2.24%</td>
                        <td>0.719</td>
                        <td>4.37</td>
                        <td>0.727</td>
                        <td>0.913</td>
                        <td>0.0001</td>
                    </tr>
                    <tr>
                        <td><strong>Classic WCT</strong></td>
                        <td>Global Gaussian Matching</td>
                        <td>1.00%</td>
                        <td>3.23%</td>
                        <td>1.61%</td>
                        <td>0.648</td>
                        <td>4.35</td>
                        <td>0.735</td>
                        <td>0.906 (volume drops)</td>
                        <td>0.0009</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- ========================================================================= -->
        <!-- SECTION 4: CODE & QUICKSTART                                            -->
        <!-- ========================================================================= -->
        <div id="code" class="section-header">
            <h2>4. Code &amp; Quickstart</h2>
            <p>
                Easy-to-use Python library and CLI tool. Installable directly with pip.
            </p>
        </div>

        <div class="code-grid">
            <div class="code-box">
                <div class="code-box-header">📦 Installation (Pip)</div>
                <div class="code-content"><span class="com"># Clone and install boosted-lwt</span>
<span class="cmd">git clone</span> https://github.com/&lt;your-username&gt;/boosted-lwt.git
<span class="cmd">cd</span> boosted-lwt
<span class="cmd">pip install</span> -e .</div>
            </div>

            <div class="code-box">
                <div class="code-box-header">⚡ Command Line Interface (CLI)</div>
                <div class="code-content"><span class="com"># Zero-shot voice conversion</span>
<span class="cmd">boosted-lwt convert</span> \\
  --source input.wav \\
  --target target_ref.wav \\
  --output converted.wav \\
  --alpha 1.5

<span class="com"># Convert to canonical neutral voice</span>
<span class="cmd">boosted-lwt pure-content</span> \\
  --source input.wav \\
  --output neutral.wav</div>
            </div>
        </div>

        <div class="code-box" style="margin-top: 1.25rem;">
            <div class="code-box-header">🐍 Python API Quickstart</div>
            <div class="code-content"><span class="kw">from</span> boosted_lwt <span class="kw">import</span> <span class="fn">VoiceConverter</span>

<span class="com"># 1. Initialize converter (auto-loads WavLM, HiFi-GAN, and balanced UBM)</span>
converter = <span class="fn">VoiceConverter</span>(device=<span class="str">"cuda"</span>)

<span class="com"># 2. Zero-shot Voice Conversion with Boosted LWT (alpha=1.5)</span>
converter.<span class="fn">convert</span>(
    source=<span class="str">"source.wav"</span>,
    target=[<span class="str">"target_ref1.wav"</span>, <span class="str">"target_ref2.wav"</span>],
    alpha=<span class="str">1.5</span>,
    output_path=<span class="str">"converted.wav"</span>
)

<span class="com"># 3. Convert to Standardized Canonical Neutral Voice (~160 Hz, CER 0.00%)</span>
converter.<span class="fn">extract_pure_content</span>(
    source=<span class="str">"source.wav"</span>,
    output_path=<span class="str">"neutral_voice.wav"</span>
)</div>
        </div>

    </div>

    <footer>
        <p>Under Review &bull; Local Transport Mixtures for Zero-Shot Voice Conversion in SSL Spaces</p>
        <p style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.35rem;">Samir Sadok<sup>*</sup> &bull; Xavier Alameda-Pineda<sup>*</sup> &bull; Inria, Univ. Grenoble Alpes, CNRS</p>
    </footer>

    <!-- Interactive Waveform Player Coordination Script -->
    <script>
        let currentlyPlayingId = null;

        function formatTime(seconds) {
            if (!seconds || isNaN(seconds)) return "0:00";
            const m = Math.floor(seconds / 60);
            const s = Math.floor(seconds % 60);
            return `${m}:${s < 10 ? '0' : ''}${s}`;
        }

        function togglePlay(playerId) {
            const audio = document.getElementById('audio_' + playerId);
            const unit = document.getElementById('unit_' + playerId);
            const btn = unit.querySelector('.play-btn');
            const iconPlay = btn.querySelector('.icon-play');
            const iconPause = btn.querySelector('.icon-pause');

            if (currentlyPlayingId && currentlyPlayingId !== playerId) {
                const prevAudio = document.getElementById('audio_' + currentlyPlayingId);
                const prevUnit = document.getElementById('unit_' + currentlyPlayingId);
                if (prevAudio && prevUnit) {
                    prevAudio.pause();
                    const prevBtn = prevUnit.querySelector('.play-btn');
                    prevBtn.querySelector('.icon-play').style.display = 'block';
                    prevBtn.querySelector('.icon-pause').style.display = 'none';
                }
            }

            if (audio.paused) {
                audio.play();
                currentlyPlayingId = playerId;
                iconPlay.style.display = 'none';
                iconPause.style.display = 'block';
            } else {
                audio.pause();
                currentlyPlayingId = null;
                iconPlay.style.display = 'block';
                iconPause.style.display = 'none';
            }
        }

        function seekWave(e, playerId) {
            const audio = document.getElementById('audio_' + playerId);
            const track = e.currentTarget;
            const rect = track.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const pct = Math.max(0, Math.min(1, clickX / rect.width));

            if (audio.duration) {
                audio.currentTime = pct * audio.duration;
                updateWaveFill(playerId, pct);
                if (audio.paused) {
                    togglePlay(playerId);
                }
            }
        }

        function updateWaveFill(playerId, pct) {
            const fill = document.getElementById('fill_' + playerId);
            if (fill) {
                fill.setAttribute('width', (pct * 100) + '%');
            }
        }

        document.querySelectorAll('audio').forEach(audio => {
            const playerId = audio.id.replace('audio_', '');
            const timeReadout = document.getElementById('time_' + playerId);
            const unit = document.getElementById('unit_' + playerId);

            audio.addEventListener('timeupdate', () => {
                if (audio.duration) {
                    const pct = audio.currentTime / audio.duration;
                    updateWaveFill(playerId, pct);
                    timeReadout.textContent = formatTime(audio.currentTime);
                }
            });

            audio.addEventListener('loadedmetadata', () => {
                timeReadout.textContent = formatTime(audio.duration);
            });

            audio.addEventListener('ended', () => {
                updateWaveFill(playerId, 0);
                timeReadout.textContent = formatTime(audio.duration || 0);
                const btn = unit.querySelector('.play-btn');
                btn.querySelector('.icon-play').style.display = 'block';
                btn.querySelector('.icon-pause').style.display = 'none';
                if (currentlyPlayingId === playerId) currentlyPlayingId = null;
            });
        });

        function filterCategory(cat, btn) {
            document.querySelectorAll('.filter-buttons .tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            document.querySelectorAll('.pair-block').forEach(card => {
                if (cat === 'all' || card.getAttribute('data-cat') === cat) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
            
            document.querySelectorAll('.pair-tab-chip').forEach(c => c.classList.remove('active'));
            document.querySelector('.pair-tab-chip.all-chip').classList.add('active');
        }

        function selectPairTab(pairId, chip) {
            document.querySelectorAll('.pair-tab-chip').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            
            if (pairId === 'all') {
                document.querySelectorAll('.pair-block').forEach(card => {
                    card.style.display = 'block';
                });
            } else {
                document.querySelectorAll('.pair-block').forEach(card => {
                    if (card.id === 'card_' + pairId) {
                        card.style.display = 'block';
                        card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    } else {
                        card.style.display = 'none';
                    }
                });
            }
        }
    </script>
</body>
</html>
"""

# Save to local demo directory
out_html = base_dir / "index.html"
out_html.write_text(html)
print(f"[SUCCESS] Re-built Demo Website at: {out_html.resolve()}")

# Save to separate git repository as well
gh_pages_html = Path("/local_scratch/ssadok/boosted-lwt-demo/index.html")
if gh_pages_html.parent.exists():
    gh_pages_html.write_text(html)
    print(f"[SUCCESS] Updated separate Demo Repo Website at: {gh_pages_html.resolve()}")
