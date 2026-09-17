import os, sys, torch, torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
import jiwer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.inference.speaker import EncoderClassifier
import torchaudio.functional as F

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.core.models import get_wct, compute_trajectory_jitter

device = 'cuda:0'

print("=== Jacobian Functional Ablation: Local Covariance vs. Dynamic Routing ===")
print("Loading models on device:", device)

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
    with torch.no_grad():
        feat, _ = wavlm.extract_features(w, output_layer=6)
    return feat.squeeze(0).cpu().numpy(), w.squeeze().cpu()

def voc(f):
    with torch.inference_mode():
        if isinstance(f, np.ndarray):
            f = torch.tensor(f, dtype=torch.float32, device=device)
        elif f.device != torch.device(device):
            f = f.to(device)
        return hifigan(f.unsqueeze(0)).squeeze(0).cpu()

def tr(w):
    inp = processor(w.squeeze().numpy(), sampling_rate=16000, return_tensors='pt', padding=True).to(device)
    with torch.no_grad(): log = asr(inp.input_values).logits
    return processor.batch_decode(torch.argmax(log, dim=-1))[0]

def emb(w):
    with torch.no_grad():
        return torch.nn.functional.normalize(spk_model.encode_batch(w.squeeze().float().unsqueeze(0).to(device)).squeeze().cpu(), dim=0)

root = Path('/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean')

# 6 balanced target speakers (3F, 3M)
target_speakers = ['121', '237', '1221', '260', '1089', '1188']
# 6 diverse source test sentences
src_spks = ['1284', '1320', '1580', '1995', '2300', '2830']
src_files = [list((root / s).rglob('*.flac'))[0] for s in src_spks]

print(f"Pre-extracting {len(src_files)} source test utterances...")
src_data = [(ext(p)[0], tr(ext(p)[1])) for p in src_files]

records = []

K_clusters = 25
beta_val = 20.0

for spk in target_speakers:
    print(f"\n=======================================================")
    print(f"--> Target Speaker: {spk}")
    print(f"=======================================================")
    files = sorted(list((root / spk).rglob('*.flac')))[:15]
    X_list, wavs = [], []
    for p in files:
        x, w = ext(p); X_list.append(x); wavs.append(w)
    Y = np.concatenate(X_list, axis=0) # [Nt, 1024]
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    Y_norm = torch.nn.functional.normalize(Y_t, dim=1)
    
    prof = torch.stack([emb(w) for w in wavs[:5]]).mean(dim=0)
    prof = torch.nn.functional.normalize(prof, dim=0)
    
    # 1. Global statistics
    mu_Y, W_Y, C_Y = get_wct(Y, 128)
    
    # 2. Fit K-Means for local components
    km = KMeans(n_clusters=K_clusters, random_state=42, n_init=1).fit(Y)
    cents = km.cluster_centers_
    cents_norm = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-8)
    
    cl = []
    for k in range(K_clusters):
        Yk = Y[km.labels_ == k]
        muk = np.mean(Yk, axis=0)
        if len(Yk) > 128:
            _, _, Ck = get_wct(Yk, 128)
        else:
            cov_k = np.cov(Yk - muk, rowvar=False) if len(Yk) > 1 else np.zeros((1024, 1024))
            evals, evecs = np.linalg.eigh(cov_k)
            evals = np.maximum(evals, 1e-5)
            Ck = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
        cl.append((muk, Ck))

    # Pre-whiten source data
    whitened_src = []
    for xs, ref in src_data:
        mu_X, W_X, _ = get_wct(xs, min(128, len(xs)-1))
        xw = (xs - mu_X) @ W_X
        xs_norm = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-8)
        whitened_src.append((xs, xw, xs_norm, ref))

    # Define the 5 configurations to evaluate
    configs = [
        ('Global WCT (K=1)', False, False),
        ('Fixed Routing (Frozen w_k)', True, False),
        ('No Local Covariance (Dynamic Mean)', False, True),
        ('Soft Local WCT (Full Ours)', True, True),
        ('kNN-VC (1-NN Reference)', '0th-order', 'Hard Voronoi')
    ]

    for name, has_local_cov, has_dyn_routing in configs:
        cers, sims, jitters, rel_jitters = [], [], [], []

        for xs, xw, xs_norm, ref in whitened_src:
            Ns = len(xs)

            if name == 'Global WCT (K=1)':
                x_hat = xw @ C_Y.T + mu_Y

            elif name == 'kNN-VC (1-NN Reference)':
                xs_t = torch.tensor(xs, dtype=torch.float32, device=device)
                xs_n = torch.nn.functional.normalize(xs_t, dim=1)
                sim_mat = xs_n @ Y_norm.T
                idx = torch.argmax(sim_mat, dim=1)
                x_hat = Y_t[idx].cpu().numpy()

            else:
                # Compute dynamic weights w_k(x_t)
                sim_mat = xs_norm @ cents_norm.T # [Ns, K]
                logits = sim_mat * beta_val
                logits -= np.max(logits, axis=1, keepdims=True)
                w_dynamic = np.exp(logits)
                w_dynamic /= w_dynamic.sum(axis=1, keepdims=True)

                if name == 'Fixed Routing (Frozen w_k)':
                    # Frozen weights: average routing over the utterance (w_k(x0) = \bar{w}_k)
                    # This strictly sets \nabla w_k = 0 across time!
                    w_fixed = np.mean(w_dynamic, axis=0, keepdims=True) # [1, K]
                    w_use = np.repeat(w_fixed, Ns, axis=0) # [Ns, K]
                    x_hat = np.zeros_like(xs)
                    for k in range(K_clusters):
                        muk, Ck = cl[k]
                        x_hat += w_use[:, k:k+1] * (xw @ Ck.T + muk)

                elif name == 'No Local Covariance (Dynamic Mean)':
                    # Dynamic routing is active (\nabla w_k != 0), but local covariance is eliminated!
                    # We use the global covariance C_Y instead of cluster-specific C_k:
                    # x_hat = sum_k w_k(x) ( C_Y xw + mu_k )
                    x_hat = np.zeros_like(xs)
                    global_trans = xw @ C_Y.T
                    for k in range(K_clusters):
                        muk, _ = cl[k]
                        x_hat += w_dynamic[:, k:k+1] * (global_trans + muk)

                elif name == 'Soft Local WCT (Full Ours)':
                    # Both mechanisms active: dynamic routing AND specialized local covariances
                    x_hat = np.zeros_like(xs)
                    for k in range(K_clusters):
                        muk, Ck = cl[k]
                        x_hat += w_dynamic[:, k:k+1] * (xw @ Ck.T + muk)

            # Compute trajectory jitter
            jit, rel_jit = compute_trajectory_jitter(x_hat, xs)
            jitters.append(jit)
            rel_jitters.append(rel_jit)

            # Synthesize audio and evaluate
            wc = voc(x_hat)
            cers.append(jiwer.cer(ref, tr(wc)))
            sims.append(torch.dot(prof, emb(wc)).item())

        mean_sim = np.mean(sims)
        mean_cer = np.mean(cers) * 100
        mean_jit = np.mean(jitters)
        mean_rel_jit = np.mean(rel_jitters)

        records.append({
            'Speaker': spk,
            'Model': name,
            'Local_Covariance': '✓' if has_local_cov is True else ('✗' if has_local_cov is False else str(has_local_cov)),
            'Dynamic_Routing': '✓' if has_dyn_routing is True else ('✗' if has_dyn_routing is False else str(has_dyn_routing)),
            'Sim': mean_sim,
            'CER': mean_cer,
            'Jitter': mean_jit,
            'Relative_Jitter': mean_rel_jit
        })
        print(f"  {name:35s} | Sim: {mean_sim:.4f} | CER: {mean_cer:5.2f}% | RelJitter: {mean_rel_jit:.3f}")

df = pd.DataFrame(records)
out_csv = Path('/local_scratch/ssadok/un_projet_audio/output/tables/jacobian_functional_ablation_6spk.csv')
out_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)
print(f"\nSaved results to {out_csv}")

summary = df.groupby(['Model', 'Local_Covariance', 'Dynamic_Routing'], sort=False).agg({
    'CER': ['mean', 'std'],
    'Sim': ['mean', 'std'],
    'Relative_Jitter': ['mean', 'std']
})
print("\n" + "="*80)
print("=== JACOBIAN FUNCTIONAL ABLATION SUMMARY ACROSS 6 SPEAKERS ===")
print("="*80)
print(summary)
