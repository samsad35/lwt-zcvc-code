import os, sys, torch, torchaudio
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import jiwer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.inference.speaker import EncoderClassifier
import torchaudio.functional as F

device = 'cuda:0'

print("Loading models...")
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
        return hifigan(torch.tensor(f, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).cpu()

def get_wct(X, k=128):
    mu = np.mean(X, axis=0)
    cov = np.cov(X - mu, rowvar=False)
    e_val, e_vec = np.linalg.eigh(cov)
    idx = np.argsort(e_val)[::-1]
    e_val, e_vec = e_val[idx], e_vec[:, idx]
    k = min(k, len(e_val))
    e_val = np.maximum(e_val[:k], 1e-5)
    e_vec = e_vec[:, :k]
    W = e_vec @ np.diag(1.0 / np.sqrt(e_val)) @ e_vec.T
    C = e_vec @ np.diag(np.sqrt(e_val)) @ e_vec.T
    return mu, W, C

def tr(w):
    inp = processor(w.numpy(), sampling_rate=16000, return_tensors='pt', padding=True).to(device)
    with torch.no_grad(): log = asr(inp.input_values).logits
    return processor.batch_decode(torch.argmax(log, dim=-1))[0]

def emb(w):
    with torch.no_grad():
        if torch.is_tensor(w):
            w_1d = w.squeeze().float()
        else:
            w_1d = torch.tensor(w).squeeze().float()
        return torch.nn.functional.normalize(spk_model.encode_batch(w_1d.unsqueeze(0).to(device)).squeeze().cpu(), dim=0)

root = Path('/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean')

# Test on 2 target speakers: 121 (F) and 260 (M)
for tgt_spk in ['121', '260']:
    print(f"\n==========================================")
    print(f"Testing Target Speaker: {tgt_spk}")
    print(f"==========================================")
    tgt_files = sorted(list((root / tgt_spk).rglob('*.flac')))[:15]
    X_tgt_list, tgt_wavs = [], []
    for p in tgt_files:
        x, w = ext(p); X_tgt_list.append(x); tgt_wavs.append(w)
    X_tgt = np.concatenate(X_tgt_list, axis=0)
    tgt_prof = torch.stack([emb(w) for w in tgt_wavs[:5]]).mean(dim=0)
    tgt_prof = torch.nn.functional.normalize(tgt_prof, dim=0)

    # 5 diverse source files
    src_files = [list((root / s).rglob('*.flac'))[0] for s in ['1320', '2300', '2830', '1284', '1995']]
    src_data = [(ext(p)[0], tr(ext(p)[1])) for p in src_files]

    # K=1
    mu_t, _, C_t = get_wct(X_tgt)
    cers_1, sims_1 = [], []
    for xs, ref in src_data:
        mu_s, W_s, _ = get_wct(xs, min(128, len(xs)-1))
        xc = (xs - mu_s) @ W_s @ C_t.T + mu_t
        w_c = voc(xc)
        cers_1.append(jiwer.cer(ref, tr(w_c)))
        sims_1.append(torch.dot(tgt_prof, emb(w_c)).item())
    print(f'K=1: Sim={np.mean(sims_1):.4f}, CER={np.mean(cers_1)*100:.2f}%')

    # K=3, 5, 8 with cosine softmax beta=20.0
    for K in [3, 5, 8]:
        km = KMeans(n_clusters=K, random_state=42, n_init=5).fit(X_tgt)
        cl = [get_wct(X_tgt[km.labels_ == k], 128) for k in range(K)]
        cent_norm = km.cluster_centers_ / (np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True) + 1e-8)
        cers_k, sims_k = [], []
        for xs, ref in src_data:
            mu_s, W_s, _ = get_wct(xs, min(128, len(xs)-1))
            xw = (xs - mu_s) @ W_s
            xs_norm = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-8)
            logits = (xs_norm @ cent_norm.T) * 20.0
            w = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            w = w / w.sum(axis=1, keepdims=True)
            xc = np.zeros_like(xs)
            for k in range(K):
                mu_k, _, C_k = cl[k]
                xc += w[:, k:k+1] * (xw @ C_k.T + mu_k)
            w_c = voc(xc)
            cers_k.append(jiwer.cer(ref, tr(w_c)))
            sims_k.append(torch.dot(tgt_prof, emb(w_c)).item())
        print(f'K={K}: Sim={np.mean(sims_k):.4f}, CER={np.mean(cers_k)*100:.2f}%')

    # 1-NN and 4-NN
    X_tgt_t = torch.tensor(X_tgt).float().to(device)
    tgt_norm = torch.nn.functional.normalize(X_tgt_t, dim=1)
    for k_knn in [1, 4]:
        cers_knn, sims_knn = [], []
        for xs, ref in src_data:
            xs_t = torch.tensor(xs).float().to(device)
            xs_n = torch.nn.functional.normalize(xs_t, dim=1)
            sim_mat = xs_n @ tgt_norm.T
            topk = torch.topk(sim_mat, k=k_knn, dim=1)[1]
            x_out = X_tgt_t[topk.squeeze(1)] if k_knn==1 else torch.stack([X_tgt_t[idx].mean(dim=0) for idx in topk])
            w_c = voc(x_out.cpu().numpy())
            cers_knn.append(jiwer.cer(ref, tr(w_c)))
            sims_knn.append(torch.dot(tgt_prof, emb(w_c)).item())
        print(f'kNN={k_knn}: Sim={np.mean(sims_knn):.4f}, CER={np.mean(cers_knn)*100:.2f}%')
