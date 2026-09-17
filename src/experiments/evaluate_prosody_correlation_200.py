import os, sys, time, torch, torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.stats as stats
from sklearn.cluster import KMeans
import torchaudio.functional as AF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.core.models import get_wct

device = 'cuda:0'

print("================================================================================")
print("=== EVALUATING PITCH & LOUDNESS CORRELATION ON UNIFIED 200 BENCHMARK ===")
print("================================================================================")

# 1. Models
wavlm = torch.hub.load('bshall/knn-vc', 'wavlm_large', trust_repo=True, device=device).eval()
hifigan, _ = torch.hub.load('bshall/knn-vc', 'hifigan_wavlm', trust_repo=True, prematched=True, device=device)
hifigan.eval()

def ext(p):
    w, sr = torchaudio.load(str(p))
    w = w.to(device)
    if sr != 16000: w = AF.resample(w, sr, 16000)
    if w.dim() == 1: w = w.unsqueeze(0)
    elif w.dim() == 2 and w.shape[0] > 1: w = w.mean(dim=0, keepdim=True)
    with torch.no_grad(): feat, _ = wavlm.extract_features(w, output_layer=6)
    return feat.squeeze(0).cpu().numpy(), w.squeeze().cpu()

def voc(f):
    with torch.inference_mode():
        if isinstance(f, np.ndarray): f = torch.tensor(f, dtype=torch.float32, device=device)
        return hifigan(f.unsqueeze(0)).squeeze().cpu()

def compute_energy(wav, frame_length=400, hop_length=160):
    wav = wav.squeeze()
    if wav.dim() == 0: return np.array([])
    unfolded = wav.unfold(0, frame_length, hop_length)
    rms = torch.sqrt(torch.mean(unfolded**2, dim=-1) + 1e-8)
    energy_db = 20 * torch.log10(rms + 1e-4)
    return energy_db.numpy()

def compute_pitch(wav, sr=16000):
    if wav.dim() == 1: wav = wav.unsqueeze(0)
    pitch = AF.detect_pitch_frequency(wav, sample_rate=sr, frame_time=0.025, win_length=30)
    pitch_np = pitch.squeeze().numpy()
    voiced = pitch_np > 50.0 # voiced threshold in Hz
    return pitch_np, voiced

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

# 10 Target Speakers
target_speakers = [
    '121', '1089', '1221', '3570', '3729', # Female
    '237', '260', '1188', '4446', '4507'   # Male
]

# 20 Source Utterances from 10 distinct speakers
src_spks = ['1284', '1320', '1580', '1995', '2094', '2300', '2830', '2961', '3575', '4077']
src_files = []
for s in src_spks:
    f_list = sorted(list((root / s).rglob('*.flac')))
    if len(f_list) >= 2: src_files.extend(f_list[:2])
    else: src_files.extend(f_list)
src_files = src_files[:20]

print("Pre-extracting source features and prosody references...", flush=True)
src_data = []
for p in src_files:
    feat, wav = ext(p)
    mu_X, W_X, _ = get_wct(feat, min(128, len(feat)-1))
    xw = (feat - mu_X) @ W_X
    feat_norm = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
    pitch_src, voiced_src = compute_pitch(wav)
    energy_src = compute_energy(wav)
    src_data.append({
        'path': p,
        'spk': p.parent.parent.name,
        'feat': feat,
        'feat_norm': feat_norm,
        'xw': xw,
        'wav': wav,
        'pitch': pitch_src,
        'voiced': voiced_src,
        'energy': energy_src,
        'duration': len(feat) * 0.02
    })

# Shared background clusters
bg_spks = ['4970', '4992', '5142', '5639', '5683', '61']
bg_feats = []
for b_spk in bg_spks:
    b_files = sorted(list((root / b_spk).rglob('*.flac')))[:6]
    for bf in b_files:
        x, _ = ext(bf); bg_feats.append(x)
bg_Y = np.concatenate(bg_feats, axis=0)

K_clusters = 10
beta_val = 20.0
km_shared = KMeans(n_clusters=K_clusters, random_state=42, n_init=1).fit(bg_Y)
cents_shared = torch.tensor(km_shared.cluster_centers_, dtype=torch.float32, device=device)
cents_shared_norm = torch.nn.functional.normalize(cents_shared, dim=1)

cov_shared_k = []
for k in range(K_clusters):
    idx_k = np.where(km_shared.labels_ == k)[0]
    Yk = bg_Y[idx_k]
    muk = np.mean(Yk, axis=0)
    covk = np.cov(Yk - muk, rowvar=False) if len(Yk) > 1 else np.zeros((1024, 1024))
    cov_shared_k.append(torch.tensor(covk, dtype=torch.float32, device=device))

print("Extracting target models (T=20 utterances per speaker)...", flush=True)
target_models = {}
for spk in target_speakers:
    files = sorted(list((root / spk).rglob('*.flac')))[:20]
    X_list = []
    for p in files:
        x, _ = ext(p); X_list.append(x)
    Y = np.concatenate(X_list, axis=0)
    mu_Y, W_Y, C_Y = get_wct(Y, 128)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    Y_norm = torch.nn.functional.normalize(Y_t, dim=1)
    
    km_tgt = KMeans(n_clusters=K_clusters, random_state=42, n_init=1).fit(Y)
    cents_tgt = torch.tensor(km_tgt.cluster_centers_, dtype=torch.float32, device=device)
    cents_tgt_norm = torch.nn.functional.normalize(cents_tgt, dim=1)
    cl_wct = []
    for k in range(K_clusters):
        idx_k = np.where(km_tgt.labels_ == k)[0]
        Yk = Y[idx_k]
        if len(Yk) > 128:
            muk, _, Ck = get_wct(Yk, 128)
        else:
            muk = np.mean(Yk, axis=0) if len(Yk) > 0 else np.zeros(1024)
            Ck = np.eye(1024, dtype=np.float32)
        cl_wct.append((muk, Ck))
        
    sim_Y = Y_norm @ cents_shared_norm.t()
    w_Y = torch.softmax(sim_Y * beta_val, dim=1)
    cl_mu_Y_shared = []
    cl_cov_Y_shared = []
    delta_cov_speaker = []
    for k in range(K_clusters):
        wk = w_Y[:, k:k+1]
        mass_k = wk.sum()
        muk = (Y_t * wk).sum(dim=0) / (mass_k + 1e-8)
        yc = Y_t - muk.unsqueeze(0)
        cov_Y_k = torch.mm(yc.t(), yc * wk) / (mass_k + 1e-8)
        cl_mu_Y_shared.append(muk)
        cl_cov_Y_shared.append(cov_Y_k)
        delta_cov_speaker.append(cov_Y_k - cov_shared_k[k])
        
    target_models[spk] = {
        'Y_t': Y_t,
        'Y_norm': Y_norm,
        'mu_Y': mu_Y,
        'C_Y': C_Y,
        'cents_tgt_norm': cents_tgt_norm,
        'cl_wct': cl_wct,
        'cl_mu_Y_shared': cl_mu_Y_shared,
        'cl_cov_Y_shared': cl_cov_Y_shared,
        'delta_cov_speaker': delta_cov_speaker
    }

methods = [
    'Classic WCT',
    'kNN-VC (k=1)',
    'kNN-VC (k=4)',
    'LinearVC',
    'Soft Local WCT (Ours)',
    'Local Wasserstein (LWT - Ours)',
    'Boosted LWT (alpha=1.5 - Ours)'
]

records = []
t_start = time.time()
print(f"\nRunning 7 methods across 200 conversions...", flush=True)

for m_idx, method_name in enumerate(methods):
    t_m = time.time()
    count = 0
    for tgt_spk, tm in target_models.items():
        for s_item in src_data:
            xs = s_item['feat']
            xw = s_item['xw']
            Ns = len(xs)
            xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
            xs_n = torch.nn.functional.normalize(xs_t, dim=1)
            
            if method_name == 'Classic WCT':
                x_hat_np = xw @ tm['C_Y'].T + tm['mu_Y']
                
            elif method_name == 'kNN-VC (k=1)':
                sim_mat = torch.mm(xs_n, tm['Y_norm'].t())
                best_idx = torch.argmax(sim_mat, dim=1)
                x_hat_np = tm['Y_t'][best_idx].cpu().numpy()
                
            elif method_name == 'kNN-VC (k=4)':
                sim_mat = torch.mm(xs_n, tm['Y_norm'].t())
                top4_idx = torch.topk(sim_mat, k=4, dim=1).indices
                x_hat = torch.mean(tm['Y_t'][top4_idx], dim=1)
                x_hat_np = x_hat.cpu().numpy()
                
            elif method_name == 'LinearVC':
                sim_mat = torch.mm(xs_n, tm['Y_norm'].t())
                best_idx = torch.argmax(sim_mat, dim=1)
                Y_matched = tm['Y_t'][best_idx]
                ones = torch.ones((Ns, 1), dtype=torch.float32, device=device)
                X_bias = torch.cat([xs_t, ones], dim=1)
                W_lin = torch.linalg.lstsq(X_bias, Y_matched).solution
                x_hat = torch.mm(X_bias, W_lin)
                x_hat_np = x_hat.cpu().numpy()
                
            elif method_name == 'Soft Local WCT (Ours)':
                sim_mat = xs_n @ tm['cents_tgt_norm'].t()
                w_dyn = torch.softmax(sim_mat * beta_val, dim=1).cpu().numpy()
                x_hat_np = np.zeros_like(xs)
                for k in range(K_clusters):
                    muk, Ck = tm['cl_wct'][k]
                    x_hat_np += w_dyn[:, k:k+1] * (xw @ Ck.T + muk)
                    
            elif method_name == 'Local Wasserstein (LWT - Ours)':
                sim_mat = xs_n @ cents_shared_norm.t()
                w_dyn = torch.softmax(sim_mat * beta_val, dim=1)
                x_hat = torch.zeros_like(xs_t)
                for k in range(K_clusters):
                    wk = w_dyn[:, k:k+1]
                    mass_k = wk.sum()
                    if mass_k < 1e-4: continue
                    mu_X_k = (xs_t * wk).sum(dim=0) / mass_k
                    xc = xs_t - mu_X_k.unsqueeze(0)
                    cov_X_k = torch.mm(xc.t(), xc * wk) / mass_k
                    A_k = compute_bures_map_torch(cov_X_k, tm['cl_cov_Y_shared'][k], eps=1e-2)
                    T_k = torch.mm(xc, A_k) + tm['cl_mu_Y_shared'][k].unsqueeze(0)
                    x_hat += wk * T_k
                x_hat_np = x_hat.cpu().numpy()
                
            elif method_name == 'Boosted LWT (alpha=1.5 - Ours)':
                sim_mat = xs_n @ cents_shared_norm.t()
                w_dyn = torch.softmax(sim_mat * beta_val, dim=1)
                x_hat = torch.zeros_like(xs_t)
                for k in range(K_clusters):
                    wk = w_dyn[:, k:k+1]
                    mass_k = wk.sum()
                    if mass_k < 1e-4: continue
                    mu_X_k = (xs_t * wk).sum(dim=0) / mass_k
                    xc = xs_t - mu_X_k.unsqueeze(0)
                    cov_X_k = torch.mm(xc.t(), xc * wk) / mass_k
                    cov_boosted = cov_shared_k[k] + 1.5 * tm['delta_cov_speaker'][k]
                    cov_boosted = project_psd(cov_boosted, eps=1e-3)
                    A_k = compute_bures_map_torch(cov_X_k, cov_boosted, eps=1e-2)
                    T_k = torch.mm(xc, A_k) + tm['cl_mu_Y_shared'][k].unsqueeze(0)
                    x_hat += wk * T_k
                x_hat_np = x_hat.cpu().numpy()

            # Vocoder
            wc = voc(x_hat_np)
            
            # Loudness correlation
            e_src = s_item['energy']
            e_c = compute_energy(wc)
            min_len_e = min(len(e_src), len(e_c))
            r_loud, _ = stats.pearsonr(e_src[:min_len_e], e_c[:min_len_e])
            
            # Pitch correlation (log F0 on co-voiced frames)
            p_src, v_src = s_item['pitch'], s_item['voiced']
            p_c, v_c = compute_pitch(wc)
            min_len_p = min(len(p_src), len(p_c))
            co_v = (v_src[:min_len_p]) & (v_c[:min_len_p])
            if co_v.sum() > 10:
                log_p_src = np.log(p_src[:min_len_p][co_v])
                log_p_c = np.log(p_c[:min_len_p][co_v])
                r_pitch, _ = stats.pearsonr(log_p_src, log_p_c)
            else:
                r_pitch = np.nan
                
            records.append({
                'Method': method_name,
                'Target_Speaker': tgt_spk,
                'Source_Speaker': s_item['spk'],
                'r_pitch': r_pitch,
                'r_loudness': r_loud
            })
            count += 1
            if count % 50 == 0:
                print(f"    [{m_idx+1}/7] {method_name}: {count}/200 done...", flush=True)
                
    print(f"  [{m_idx+1}/7] {method_name} evaluated in {time.time()-t_m:.1f}s.", flush=True)

df = pd.DataFrame(records)
df.to_csv('output/tables/prosody_correlation_unified_200.csv', index=False)

summary = df.groupby('Method').agg({
    'r_pitch': ['mean', 'std'],
    'r_loudness': ['mean', 'std']
}).reset_index()

print("\n================================================================================")
print("=== FINAL PROSODY PRESERVATION BENCHMARK RESULTS (N=200 CONVERSIONS) ===")
print("================================================================================")
for idx, r in summary.iterrows():
    m = r[('Method', '')]
    rp_m, rp_s = r[('r_pitch', 'mean')], r[('r_pitch', 'std')]
    rl_m, rl_s = r[('r_loudness', 'mean')], r[('r_loudness', 'std')]
    print(f"{m:32s} | r_pitch (F0) = {rp_m:.4f} ± {rp_s:.4f} | r_loudness (Energy) = {rl_m:.4f} ± {rl_s:.4f}")
print("================================================================================")
