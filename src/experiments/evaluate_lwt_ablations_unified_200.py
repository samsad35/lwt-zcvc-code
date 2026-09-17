import os, sys, time, torch, torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
import jiwer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.inference.speaker import EncoderClassifier
import torchaudio.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.core.models import compute_trajectory_jitter

device = 'cuda:0'

print("================================================================================")
print("=== LAUNCHING UNIFIED LWT ABLATIONS (N=200, 10 TARGET SPEAKERS, STRICT MATCH) ===")
print("================================================================================")

# 1. Load Pretrained Models
wavlm = torch.hub.load('bshall/knn-vc', 'wavlm_large', trust_repo=True, device=device).eval()
hifigan, _ = torch.hub.load('bshall/knn-vc', 'hifigan_wavlm', trust_repo=True, prematched=True, device=device)
hifigan.eval()

processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
asr = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h').to(device).eval()
spk_model = EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb', 
    run_opts={'device': 'cuda:0'}, 
    savedir='/tmp/speechbrain'
)

def ext(p):
    w, sr = torchaudio.load(str(p))
    w = w.to(device)
    if sr != 16000: w = F.resample(w, sr, 16000)
    if w.dim() == 1: w = w.unsqueeze(0)
    with torch.no_grad(): feat, _ = wavlm.extract_features(w, output_layer=6)
    return feat.squeeze(0).cpu().numpy(), w.squeeze().cpu()

def voc(f):
    with torch.inference_mode():
        if isinstance(f, np.ndarray): f = torch.tensor(f, dtype=torch.float32, device=device)
        return hifigan(f.unsqueeze(0)).squeeze(0).cpu()

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

def project_psd(M, eps=1e-3):
    evals, evecs = torch.linalg.eigh(M)
    evals = torch.clamp(evals, min=eps)
    return evecs @ torch.diag(evals) @ evecs.t()

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

root = Path('/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean')

# EXACT SAME 10 Target Speakers as Benchmark
target_speakers = [
    '121', '1089', '1221', '3570', '3729', # Female
    '237', '260', '1188', '4446', '4507'   # Male
]

# EXACT SAME 20 Source Utterances from 10 distinct speakers
src_spks = ['1284', '1320', '1580', '1995', '2094', '2300', '2830', '2961', '3575', '4077']
src_files = []
for s in src_spks:
    f_list = sorted(list((root / s).rglob('*.flac')))
    if len(f_list) >= 2:
        src_files.extend(f_list[:2])
    else:
        src_files.extend(f_list)
src_files = src_files[:20]

print(f"Loaded {len(target_speakers)} target speakers and {len(src_files)} source utterances.")
print(f"Total conversions per condition = {len(target_speakers)} x {len(src_files)} = 200 conversions.")

print("Pre-extracting source features and references...")
src_data = []
for p in src_files:
    feat, wav = ext(p)
    ref = tr(wav)
    src_data.append({
        'path': p,
        'spk': p.parent.parent.name,
        'feat': feat,
        'ref': ref,
        'duration': len(feat) * 0.02
    })

# Precompute Population Shared Phonetic Features
bg_spks = ['4970', '4992', '5142', '5639', '5683', '61']
bg_feats = []
for b_spk in bg_spks:
    b_files = sorted(list((root / b_spk).rglob('*.flac')))[:6]
    for bf in b_files:
        x, _ = ext(bf)
        bg_feats.append(x)
bg_Y = np.concatenate(bg_feats, axis=0)

# Pre-load target speech files up to 25 utterances
target_files_dict = {}
for spk in target_speakers:
    target_files_dict[spk] = sorted(list((root / spk).rglob('*.flac')))[:25]

# Pre-extract target utterances cache
print("Caching target features and speaker embeddings...")
target_cache = {}
for spk in target_speakers:
    feats, wavs = [], []
    for f in target_files_dict[spk]:
        x, w = ext(f)
        feats.append(x)
        wavs.append(w)
    prof = torch.stack([emb(w) for w in wavs[:10]]).mean(dim=0)
    prof = torch.nn.functional.normalize(prof, dim=0)
    target_cache[spk] = {
        'feats': feats,
        'wavs': wavs,
        'prof': prof
    }

# Helper function to run LWT evaluation across the 200 conversions
def eval_lwt_suite(cents_norm, cl_mu_Y, cl_cov_Y, beta=20.0):
    cers, sims, jitters = [], [], []
    for tgt_spk in target_speakers:
        prof = target_cache[tgt_spk]['prof']
        mu_k_list = cl_mu_Y[tgt_spk]
        cov_k_list = cl_cov_Y[tgt_spk]
        cents = cents_norm[tgt_spk] if isinstance(cents_norm, dict) else cents_norm
        K = len(mu_k_list)
        
        for s in src_data:
            xs = s['feat']
            ref = s['ref']
            xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
            xs_n = torch.nn.functional.normalize(xs_t, dim=1)
            
            sim_mat = xs_n @ cents.t()
            w_dyn = torch.softmax(sim_mat * beta, dim=1)
            
            x_hat = torch.zeros_like(xs_t)
            for k in range(K):
                wk = w_dyn[:, k:k+1]
                mass_k = wk.sum()
                if mass_k < 1e-4: continue
                mu_X_k = (xs_t * wk).sum(dim=0) / mass_k
                xc = xs_t - mu_X_k.unsqueeze(0)
                cov_X_k = torch.mm(xc.t(), xc * wk) / mass_k
                A_k = compute_bures_map_torch(cov_X_k, cov_k_list[k], eps=1e-2)
                T_k = torch.mm(xc, A_k) + mu_k_list[k].unsqueeze(0)
                x_hat += wk * T_k
                
            x_np = x_hat.cpu().numpy()
            _, rjit = compute_trajectory_jitter(x_np, xs)
            wc = voc(x_np)
            hyp = tr(wc)
            cer = jiwer.cer(ref, hyp) * 100.0
            sim = torch.dot(prof, emb(wc)).item()
            
            cers.append(cer)
            sims.append(sim)
            jitters.append(rjit)
            
    return float(np.mean(cers)), float(np.mean(sims)), float(np.mean(jitters))

# ==============================================================================
# ABLATION 4 (FIRST): ALPHA SPEAKER COVARIANCE BOOST (N=200, 10 Targets, T=20)
# ==============================================================================
print("\n================================================================================")
print("=== ABLATION 4: ALPHA SPEAKER BOOST (K=10, Beta=20, T=20 utts, N=200) ===")
print("================================================================================")
K_fixed = 10
km_shared = KMeans(n_clusters=K_fixed, random_state=42, n_init=1).fit(bg_Y)
cents_sh = torch.tensor(km_shared.cluster_centers_, dtype=torch.float32, device=device)
cents_sh_norm = torch.nn.functional.normalize(cents_sh, dim=1)

cov_shared_k = []
for k in range(K_fixed):
    idx_k = np.where(km_shared.labels_ == k)[0]
    Yk = bg_Y[idx_k]
    muk = np.mean(Yk, axis=0)
    covk = np.cov(Yk - muk, rowvar=False) if len(Yk) > 1 else np.zeros((1024, 1024))
    cov_shared_k.append(torch.tensor(covk, dtype=torch.float32, device=device))

# Build target representation on shared clusters with strictly T=20 utterances
cl_mu_shared = {}
cl_cov_shared = {}
delta_cov = {}
for spk in target_speakers:
    Y = np.concatenate(target_cache[spk]['feats'][:20], axis=0) # STRICT T=20
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    Y_n = torch.nn.functional.normalize(Y_t, dim=1)
    sim_Y = Y_n @ cents_sh_norm.t()
    w_Y = torch.softmax(sim_Y * 20.0, dim=1)
    
    m_list, c_list, d_list = [], [], []
    for k in range(K_fixed):
        wk = w_Y[:, k:k+1]
        mass_k = wk.sum()
        muk = (Y_t * wk).sum(dim=0) / (mass_k + 1e-8)
        yc = Y_t - muk.unsqueeze(0)
        cov_Y_k = torch.mm(yc.t(), yc * wk) / (mass_k + 1e-8)
        m_list.append(muk)
        c_list.append(cov_Y_k)
        d_list.append(cov_Y_k - cov_shared_k[k])
    cl_mu_shared[spk] = m_list
    cl_cov_shared[spk] = c_list
    delta_cov[spk] = d_list

alphas = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
records_alpha = []
for a in alphas:
    t0 = time.time()
    boosted_cov = {}
    for spk in target_speakers:
        b_list = []
        for k in range(K_fixed):
            cov_b = cov_shared_k[k] + a * delta_cov[spk][k]
            cov_b = project_psd(cov_b, eps=1e-3)
            b_list.append(cov_b)
        boosted_cov[spk] = b_list
        
    cer, sim, jit = eval_lwt_suite(cents_sh_norm, cl_mu_shared, boosted_cov, beta=20.0)
    records_alpha.append({'Alpha': a, 'CER': cer, 'Sim': sim, 'Jitter': jit})
    print(f"  Alpha = {a:4.1f} | CER: {cer:5.2f}% | Sim: {sim:.4f} | Jitter: {jit:.3f} | ({time.time()-t0:.1f}s)")

out_alpha_csv = Path('/local_scratch/ssadok/un_projet_audio/output/tables/lwt_ablation_alpha_large_N.csv')
pd.DataFrame(records_alpha).to_csv(out_alpha_csv, index=False)
print(f"Saved harmonized Alpha ablation to {out_alpha_csv}")

# ==============================================================================
# ABLATION 1: BETA TEMPERATURE ON LWT (N=200, 10 Targets, T=20, Alpha=1.0)
# ==============================================================================
print("\n================================================================================")
print("=== ABLATION 1: BETA TEMPERATURE (K=10, T=20 utts, Alpha=1.0, N=200) ===")
print("================================================================================")
betas = [2.0, 5.0, 10.0, 20.0, 40.0, 100.0]
records_beta = []
for b in betas:
    t0 = time.time()
    cer, sim, jit = eval_lwt_suite(cents_sh_norm, cl_mu_shared, cl_cov_shared, beta=b)
    records_beta.append({'Beta': b, 'CER': cer, 'Sim': sim, 'Jitter': jit})
    print(f"  Beta = {b:5.1f} | CER: {cer:5.2f}% | Sim: {sim:.4f} | Jitter: {jit:.3f} | ({time.time()-t0:.1f}s)")

out_beta_csv = Path('/local_scratch/ssadok/un_projet_audio/output/tables/lwt_ablation_beta_large_N.csv')
pd.DataFrame(records_beta).to_csv(out_beta_csv, index=False)
print(f"Saved harmonized Beta ablation to {out_beta_csv}")

# ==============================================================================
# ABLATION 2: NUMBER OF CLUSTERS K ON LWT (N=200, 10 Targets, T=20, Beta=20)
# ==============================================================================
print("\n================================================================================")
print("=== ABLATION 2: NUMBER OF CLUSTERS K (Beta=20, T=20 utts, N=200) ===")
print("================================================================================")
k_values = [1, 2, 4, 8, 10, 16, 24, 32]
records_k = []
for k_val in k_values:
    t0 = time.time()
    if k_val == 1:
        muk_sh = torch.tensor(np.mean(bg_Y, axis=0), dtype=torch.float32, device=device).unsqueeze(0)
        cents_k_norm = torch.nn.functional.normalize(muk_sh, dim=1)
        cl_mu_k = {}
        cl_cov_k = {}
        for spk in target_speakers:
            Y = np.concatenate(target_cache[spk]['feats'][:20], axis=0)
            muk = torch.tensor(np.mean(Y, axis=0), dtype=torch.float32, device=device)
            covk = torch.tensor(np.cov(Y - muk.cpu().numpy(), rowvar=False), dtype=torch.float32, device=device)
            cl_mu_k[spk] = [muk]
            cl_cov_k[spk] = [project_psd(covk, eps=1e-3)]
    else:
        km_k = KMeans(n_clusters=k_val, random_state=42, n_init=1).fit(bg_Y)
        cents_k_raw = torch.tensor(km_k.cluster_centers_, dtype=torch.float32, device=device)
        cents_k_norm = torch.nn.functional.normalize(cents_k_raw, dim=1)
        
        cl_mu_k = {}
        cl_cov_k = {}
        for spk in target_speakers:
            Y = np.concatenate(target_cache[spk]['feats'][:20], axis=0)
            Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
            Y_n = torch.nn.functional.normalize(Y_t, dim=1)
            sim_Y = Y_n @ cents_k_norm.t()
            w_Y = torch.softmax(sim_Y * 20.0, dim=1)
            m_list, c_list = [], []
            for ki in range(k_val):
                wk = w_Y[:, ki:ki+1]
                mass_k = wk.sum()
                muk = (Y_t * wk).sum(dim=0) / (mass_k + 1e-8)
                yc = Y_t - muk.unsqueeze(0)
                cov_Y_ki = torch.mm(yc.t(), yc * wk) / (mass_k + 1e-8)
                m_list.append(muk)
                c_list.append(project_psd(cov_Y_ki, eps=1e-3))
            cl_mu_k[spk] = m_list
            cl_cov_k[spk] = c_list
            
    cer, sim, jit = eval_lwt_suite(cents_k_norm, cl_mu_k, cl_cov_k, beta=20.0)
    records_k.append({'K': k_val, 'CER': cer, 'Sim': sim, 'Jitter': jit})
    print(f"  K = {k_val:3d} | CER: {cer:5.2f}% | Sim: {sim:.4f} | Jitter: {jit:.3f} | ({time.time()-t0:.1f}s)")

out_k_csv = Path('/local_scratch/ssadok/un_projet_audio/output/tables/lwt_ablation_K_large_N.csv')
pd.DataFrame(records_k).to_csv(out_k_csv, index=False)
print(f"Saved harmonized K ablation to {out_k_csv}")

# ==============================================================================
# ABLATION 3: TARGET ADAPTATION BUDGET T (N=200, 10 Targets, K=10, Beta=20)
# ==============================================================================
print("\n================================================================================")
print("=== ABLATION 3: TARGET ADAPTATION BUDGET T (K=10, Beta=20, N=200) ===")
print("================================================================================")
t_utts = [1, 2, 5, 10, 15, 20, 25]
records_t = []
for n_u in t_utts:
    t0 = time.time()
    cl_mu_t = {}
    cl_cov_t = {}
    total_sec = []
    for spk in target_speakers:
        feats_sub = target_cache[spk]['feats'][:n_u]
        Y = np.concatenate(feats_sub, axis=0)
        total_sec.append(len(Y) * 0.02)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
        Y_n = torch.nn.functional.normalize(Y_t, dim=1)
        sim_Y = Y_n @ cents_sh_norm.t()
        w_Y = torch.softmax(sim_Y * 20.0, dim=1)
        m_list, c_list = [], []
        for k in range(K_fixed):
            wk = w_Y[:, k:k+1]
            mass_k = wk.sum()
            muk = (Y_t * wk).sum(dim=0) / (mass_k + 1e-8)
            yc = Y_t - muk.unsqueeze(0)
            cov_Y_k = torch.mm(yc.t(), yc * wk) / (mass_k + 1e-8)
            m_list.append(muk)
            c_list.append(project_psd(cov_Y_k, eps=1e-3))
        cl_mu_t[spk] = m_list
        cl_cov_t[spk] = c_list
        
    avg_sec = float(np.mean(total_sec))
    cer, sim, jit = eval_lwt_suite(cents_sh_norm, cl_mu_t, cl_cov_t, beta=20.0)
    records_t.append({'Utterances': n_u, 'Duration_sec': avg_sec, 'CER': cer, 'Sim': sim, 'Jitter': jit})
    print(f"  T = {n_u:2d} utts (~{avg_sec:4.1f}s) | CER: {cer:5.2f}% | Sim: {sim:.4f} | Jitter: {jit:.3f} | ({time.time()-t0:.1f}s)")

out_t_csv = Path('/local_scratch/ssadok/un_projet_audio/output/tables/lwt_ablation_T_large_N.csv')
pd.DataFrame(records_t).to_csv(out_t_csv, index=False)
print(f"Saved harmonized T ablation to {out_t_csv}")

print("\n[ALL COMPLETE] All 4 ablations successfully harmonized with the benchmark N=200 protocol!")
