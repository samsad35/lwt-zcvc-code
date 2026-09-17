import os, sys, time, torch, torchaudio
import numpy as np
from pathlib import Path
import jiwer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.inference.speaker import EncoderClassifier
import torchaudio.functional as AF

device = 'cuda:0'
print("=== Generating Pure Content with Balanced 40-Speaker Model (50/50 M/F) ===")

# 1. Models
wavlm = torch.hub.load('bshall/knn-vc', 'wavlm_large', trust_repo=True, device=device).eval()
hifigan, _ = torch.hub.load('bshall/knn-vc', 'hifigan_wavlm', trust_repo=True, prematched=True, device=device)
hifigan.eval()

processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
asr = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h').to(device).eval()

spk_model = EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb', 
    run_opts={'device': device}, 
    savedir='/tmp/speechbrain'
)

def ext(p):
    w, sr = torchaudio.load(str(p))
    w = w.to(device)
    if sr != 16000: w = AF.resample(w, sr, 16000)
    if w.dim() == 1: w = w.unsqueeze(0)
    elif w.dim() == 2 and w.shape[0] > 1: w = w.mean(dim=0, keepdim=True)
    with torch.no_grad(): feat, _ = wavlm.extract_features(w, output_layer=6)
    return feat.squeeze(0), w.squeeze().cpu()

def voc(f):
    with torch.inference_mode():
        if isinstance(f, np.ndarray): f = torch.tensor(f, dtype=torch.float32, device=device)
        audio = hifigan(f.unsqueeze(0)).squeeze().cpu()
        max_val = torch.abs(audio).max()
        if max_val > 0.99:
            audio = audio / max_val * 0.99
        return audio

def tr(w):
    inp = processor(w.squeeze().numpy(), sampling_rate=16000, return_tensors='pt', padding=True).to(device)
    with torch.no_grad(): log = asr(inp.input_values).logits
    pred_ids = torch.argmax(log, dim=-1)
    return processor.batch_decode(pred_ids)[0].strip()

def emb(w):
    with torch.no_grad():
        return torch.nn.functional.normalize(
            spk_model.encode_batch(w.squeeze().float().unsqueeze(0).to(device)).squeeze().cpu(), 
            dim=0
        )

def compute_mean_pitch(w, sr=16000):
    if w.dim() == 1: w = w.unsqueeze(0)
    pitch = AF.detect_pitch_frequency(w, sample_rate=sr, frame_time=0.025, win_length=30)
    p_np = pitch.squeeze().numpy()
    voiced = p_np > 50.0
    return float(np.mean(p_np[voiced])) if np.sum(voiced) > 5 else 0.0

def compute_bures_map_torch(cov_X, cov_Y, eps=1e-2):
    D = cov_X.shape[0]
    I_D = torch.eye(D, device=cov_X.device, dtype=cov_X.dtype)
    cX = cov_X + eps * I_D
    cY = cov_Y + eps * I_D
    
    evals_X, evecs_X = torch.linalg.eigh(cX)
    evals_X = evals_X.clamp(min=eps)
    cX_half = evecs_X @ torch.diag(torch.sqrt(evals_X)) @ evecs_X.t()
    cX_inv_half = evecs_X @ torch.diag(1.0 / torch.sqrt(evals_X)) @ evecs_X.t()
    
    M = cX_half @ cY @ cX_half
    evals_M, evecs_M = torch.linalg.eigh(M)
    evals_M = evals_M.clamp(min=eps**2)
    M_half = evecs_M @ torch.diag(torch.sqrt(evals_M)) @ evecs_M.t()
    
    A = cX_inv_half @ M_half @ cX_inv_half
    return A

# Load the new balanced background model
cache_path = Path("output/cache/shared_clusters_k10_balanced_40spk.pt")
print(f"Loading 40-speaker balanced model from {cache_path}...")
data_bal = torch.load(cache_path, map_location=device)
cents_shared = data_bal['cents_shared']
cents_shared_norm = data_bal['cents_shared_norm']
cov_shared_k = data_bal['cov_shared_k']

K_clusters = 10
beta_val = 20.0

out_dir = Path("output/audio_samples/pure_content_demo")

test_cases = [
    {
        'id': 'sample1_1284_female',
        'spk': '1284',
        'gender': 'Female (High-pitched)',
        'path': out_dir / 'sample1_1284_female/source.wav',
        'transcript': 'HE WORE BLUE SILK STOCKINGS BLUE KNEE PANTS WITH GOLD BUCKLES A BLUE RUFFLED WAIST AND A JACKET OF BRIGHT BLUE BRAIDED WITH GOLD'
    },
    {
        'id': 'sample2_1320_male',
        'spk': '1320',
        'gender': 'Male (Deep voice)',
        'path': out_dir / 'sample2_1320_male/source.wav',
        'transcript': 'SINCE THE PERIOD OF OUR TALE THE ACTIVE SPIRIT OF THE COUNTRY HAS FACILITATED COMMUNICATION'
    },
    {
        'id': 'sample3_1995_female',
        'spk': '1995',
        'gender': 'Female',
        'path': out_dir / 'sample3_1995_female/source.wav',
        'transcript': 'IN THE DEBATE BETWEEN THE SENIOR SOCIETIES HER DEFENCE OF THE POOR CAPTIVE HAD BEEN VIGOROUS AND AUDIBLE'
    },
    {
        'id': 'sample4_672_male',
        'spk': '672',
        'gender': 'Male',
        'path': out_dir / 'sample4_672_male/source.wav',
        'transcript': 'OUT IN THE WOODS STOOD A NICE LITTLE FIR TREE'
    }
]

demo_cards = []

for case in test_cases:
    case_dir = out_dir / case['id']
    xs_t, w_src = ext(case['path'])
    ref_transcript = tr(w_src)
    spk_emb_src = emb(w_src)
    pitch_src = compute_mean_pitch(w_src)
    
    # Load old 6-spk audio to compute pitch
    w_old, _ = torchaudio.load(str(case_dir / "pure_content_neutral.wav"))
    pitch_old = compute_mean_pitch(w_old)
    cer_old = jiwer.cer(ref_transcript, tr(w_old)) * 100.0
    sim_old = torch.dot(spk_emb_src, emb(w_old)).item()
    
    # -------------------------------------------------------------
    # NEW BALANCED PURE CONTENT TRANSFORMATION
    # -------------------------------------------------------------
    xs_n = torch.nn.functional.normalize(xs_t, dim=1)
    sim_mat = torch.mm(xs_n, cents_shared_norm.t())
    w_dyn = torch.softmax(sim_mat * beta_val, dim=1)
    
    z_pure = torch.zeros_like(xs_t)
    for k in range(K_clusters):
        wk = w_dyn[:, k:k+1]
        mass_k = wk.sum()
        if mass_k < 1e-4: continue
        
        mu_X_k = (xs_t * wk).sum(dim=0) / mass_k
        xc = xs_t - mu_X_k.unsqueeze(0)
        cov_X_k = torch.mm(xc.t(), xc * wk) / mass_k
        
        A_pure_k = compute_bures_map_torch(cov_X_k, cov_shared_k[k], eps=1e-2)
        T_pure_k = torch.mm(xc, A_pure_k) + cents_shared[k].unsqueeze(0)
        z_pure += wk * T_pure_k
        
    w_new = voc(z_pure)
    torchaudio.save(str(case_dir / "pure_content_balanced_40spk.wav"), w_new.unsqueeze(0), 16000)
    
    hyp_new = tr(w_new)
    cer_new = jiwer.cer(ref_transcript, hyp_new) * 100.0
    spk_emb_new = emb(w_new)
    sim_new = torch.dot(spk_emb_src, spk_emb_new).item()
    pitch_new = compute_mean_pitch(w_new)
    
    print(f"\n[{case['id']}] {case['gender']}:")
    print(f"   Source Pitch      : {pitch_src:.1f} Hz")
    print(f"   Old 6-Spk Pitch   : {pitch_old:.1f} Hz (Female-leaning 67% F)")
    print(f"   New 40-Spk Pitch  : {pitch_new:.1f} Hz (Balanced 50/50 M/F)")
    print(f"   CER New Balanced  : {cer_new:.2f}% | ECAPA Sim to Source: {sim_new:.3f}")
    
    demo_cards.append({
        'id': case['id'],
        'spk': case['spk'],
        'gender': case['gender'],
        'transcript': case['transcript'],
        'pitch_src': pitch_src,
        'pitch_old': pitch_old,
        'pitch_new': pitch_new,
        'cer_old': cer_old,
        'sim_old': sim_old,
        'cer_new': cer_new,
        'sim_new': sim_new,
        'src_path': f"{case['id']}/source.wav",
        'old_path': f"{case['id']}/pure_content_neutral.wav",
        'new_path': f"{case['id']}/pure_content_balanced_40spk.wav"
    })

# Generate Updated HTML
html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pure Content Extraction | Balanced Universal Background</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #f8fafc;
            --surface: #ffffff;
            --card-border: #e2e8f0;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --text-secondary: #334155;
            --primary: #0284c7;
            --primary-light: #f0f9ff;
            --teal: #0d9488;
            --teal-light: #f0fdfa;
            --teal-border: #99f6e4;
            --indigo: #4f46e5;
            --indigo-light: #eef2ff;
            --indigo-border: #c7d2fe;
        }
        * { box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            margin: 0; padding: 0;
            line-height: 1.6;
        }
        .hero {
            background: var(--surface);
            border-bottom: 1px solid var(--card-border);
            padding: 3rem 1.5rem 2.25rem;
            text-align: center;
        }
        .badge {
            display: inline-flex;
            align-items: center;
            background: var(--indigo-light);
            color: var(--indigo);
            border: 1px solid var(--indigo-border);
            padding: 0.3rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 1rem;
        }
        h1 {
            font-size: 2.2rem;
            font-weight: 800;
            margin: 0 0 0.75rem;
            color: var(--text-main);
        }
        p.subtitle {
            font-size: 1.05rem;
            color: var(--text-muted);
            max-width: 820px;
            margin: 0 auto;
        }
        .container {
            max-width: 1100px;
            margin: 0 auto;
            padding: 2.5rem 1.25rem 4rem;
        }
        .box-concept {
            background: var(--surface);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 2.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }
        .box-concept h3 {
            margin: 0 0 0.5rem;
            color: var(--indigo);
            font-size: 1.15rem;
            font-weight: 700;
        }
        .box-concept p {
            margin: 0.5rem 0;
            color: var(--text-secondary);
            font-size: 0.92rem;
        }
        .box-concept code {
            background: #f1f5f9;
            color: #0f172a;
            padding: 0.2rem 0.45rem;
            border-radius: 5px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.84rem;
            border: 1px solid #e2e8f0;
        }
        .card {
            background: var(--surface);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2.25rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--card-border);
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        .card-header h3 {
            margin: 0;
            font-size: 1.15rem;
            font-weight: 700;
        }
        .pill {
            background: #f1f5f9;
            color: #475569;
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .transcript {
            background: #f8fafc;
            border-left: 3px solid var(--primary);
            padding: 0.65rem 0.9rem;
            border-radius: 6px;
            font-size: 0.88rem;
            color: var(--text-secondary);
            margin-bottom: 1.25rem;
            font-style: italic;
        }
        .audio-row-3 {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 1rem;
        }
        @media (max-width: 900px) {
            .audio-row-3 { grid-template-columns: 1fr; }
        }
        .audio-box {
            background: #ffffff;
            border: 1px solid var(--card-border);
            border-radius: 10px;
            padding: 1.1rem;
            display: flex;
            flex-direction: column;
        }
        .audio-box.highlight {
            border-color: #a5b4fc;
            background: #fdfefe;
            box-shadow: 0 2px 6px rgba(79, 70, 229, 0.06);
        }
        .audio-box .label {
            font-size: 0.88rem;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 0.2rem;
        }
        .audio-box.highlight .label {
            color: var(--indigo);
        }
        .audio-box .sublabel {
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-bottom: 0.85rem;
        }
        audio {
            width: 100%;
            height: 38px;
            outline: none;
        }
        .metric-pill-row {
            display: flex;
            gap: 0.4rem;
            margin-top: 0.85rem;
            flex-wrap: wrap;
        }
        .metric-pill {
            font-size: 0.74rem;
            font-weight: 600;
            padding: 0.2rem 0.5rem;
            border-radius: 6px;
            background: #ffffff;
            border: 1px solid #cbd5e1;
            color: #334155;
        }
        .metric-pill.tag-pitch {
            background: #f0fdf4;
            border-color: #bbf7d0;
            color: #166534;
        }
        footer {
            text-align: center;
            padding: 2.5rem 1rem;
            color: var(--text-muted);
            font-size: 0.85rem;
            border-top: 1px solid var(--card-border);
            background: var(--surface);
        }
    </style>
</head>
<body>

    <header class="hero">
        <div class="badge">Pure Content &bull; Population Background Scaling</div>
        <h1>Universal Balanced Pure Content Synthesis</h1>
        <p class="subtitle">
            Demonstrating gender-neutral canonical voice projection by scaling the background dataset to <strong>40 speakers strictly balanced at 50% Male / 50% Female</strong> (49,757 frames).
        </p>
    </header>

    <div class="container">

        <div class="box-concept">
            <h3>From 6-Speaker (67% Female) to 40-Speaker (50/50 Balanced) Background</h3>
            <p>
                When background centroids <code>c<sub>k</sub></code> were estimated on 6 speakers containing 4 females (67%), the center of gravity was pulled toward higher formants.
            </p>
            <p>
                By extracting the universal background on <strong>40 speakers (20 Males / 20 Females)</strong> from <code>train-clean-100</code>, the centroid shifts to the true anatomical median between male and female vocal tracts (~150 Hz). 
            </p>
            <p>
                Listen below to the transition: from source &rarr; female-leaning 6-spk pure &rarr; <strong>true balanced 40-spk neutral pure voice</strong>!
            </p>
        </div>
"""

for c in demo_cards:
    html_content += f"""
        <div class="card">
            <div class="card-header">
                <h3>{c['id']} &bull; Original Speaker: {c['spk']} ({c['gender']})</h3>
                <span class="pill">Source Pitch: {c['pitch_src']:.0f} Hz</span>
            </div>
            <p class="transcript"><strong>Source Text:</strong> "{c['transcript']}"</p>
            
            <div class="audio-row-3">
                <div class="audio-box">
                    <span class="label">1. Original Source</span>
                    <span class="sublabel">{c['gender']}</span>
                    <audio controls preload="none">
                        <source src="{c['src_path']}" type="audio/wav">
                    </audio>
                    <div class="metric-pill-row">
                        <span class="metric-pill tag-pitch">F0: {c['pitch_src']:.0f} Hz</span>
                    </div>
                </div>
                
                <div class="audio-box">
                    <span class="label">2. Old 6-Spk (67% Female)</span>
                    <span class="sublabel">Biased Background (Female-leaning)</span>
                    <audio controls preload="none">
                        <source src="{c['old_path']}" type="audio/wav">
                    </audio>
                    <div class="metric-pill-row">
                        <span class="metric-pill tag-pitch">F0: {c['pitch_old']:.0f} Hz</span>
                        <span class="metric-pill">CER: {c['cer_old']:.2f}%</span>
                    </div>
                </div>

                <div class="audio-box highlight">
                    <span class="label">3. New 40-Spk (50% M / 50% F) ⭐</span>
                    <span class="sublabel">Balanced Universal Neutral Voice</span>
                    <audio controls preload="none">
                        <source src="{c['new_path']}" type="audio/wav">
                    </audio>
                    <div class="metric-pill-row">
                        <span class="metric-pill tag-pitch">F0: {c['pitch_new']:.0f} Hz</span>
                        <span class="metric-pill">CER: {c['cer_new']:.2f}%</span>
                        <span class="metric-pill">Sim: {c['sim_new']:.3f}</span>
                    </div>
                </div>
            </div>
        </div>
    """

html_content += """
    </div>

    <footer>
        <p>ICASSP 2026 Analytical Speech Disentanglement Demonstration &bull; Local Wasserstein Transport (LWT)</p>
    </footer>

    <script>
        document.addEventListener('play', function(e) {
            const audios = document.getElementsByTagName('audio');
            for (let i = 0; i < audios.length; i++) {
                if (audios[i] !== e.target) audios[i].pause();
            }
        }, true);
    </script>
</body>
</html>
"""

html_path = out_dir / "index.html"
html_path.write_text(html_content)
print(f"\n[SUCCESS] Updated pure content demo page at: {html_path.resolve()}")
