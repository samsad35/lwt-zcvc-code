import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
import sys
import ppgs
from scipy.spatial.distance import cosine

# Ajouter src/ au PYTHONPATH pour pouvoir importer 'une'
sys.path.append(str(Path(__file__).parent.parent))

from une import LibriSpeech
from une import WavLM

def get_speaker_id(path_str):
    return Path(path_str).parts[-3]

def extract_features_all(wavlm_target, wavlm_source, audio_paths, l_target, l_source, desc_msg, gpu=0):
    latents_target = []
    latents_source = []
    labels_ppg = []
    speaker_ids = []
    
    with torch.no_grad():
        for path in tqdm(audio_paths, desc=desc_msg):
            h_t = wavlm_target(str(path))[l_target].squeeze(0).cpu().numpy()
            h_s = wavlm_source(str(path))[l_source].squeeze(0).cpu().numpy()
            
            # Extract PPGs
            audio = ppgs.load.audio(str(path))
            p = ppgs.from_audio(audio, ppgs.SAMPLE_RATE, gpu=gpu).squeeze(0)
            
            p = p[:, ::2]
            p_class = torch.argmax(p, dim=0).cpu().numpy()
            
            min_T = min(h_t.shape[0], h_s.shape[0], p_class.shape[0])
            
            latents_target.append(h_t[:min_T])
            latents_source.append(h_s[:min_T])
            labels_ppg.append(p_class[:min_T])
            speaker_ids.append(get_speaker_id(path))
            
    return latents_target, latents_source, labels_ppg, speaker_ids

def learn_cca_mapping(X_list, Y_list, device, dim=64):
    X_frames = np.concatenate(X_list, axis=0)
    Y_frames = np.concatenate(Y_list, axis=0)
    
    X_mean = np.mean(X_frames, axis=0)
    Y_mean = np.mean(Y_frames, axis=0)
    
    Xt = torch.tensor(X_frames - X_mean, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y_frames - Y_mean, dtype=torch.float32, device=device)
    
    Q_x, R_x = torch.linalg.qr(Xt)
    Q_y, R_y = torch.linalg.qr(Yt)
    
    C = torch.matmul(Q_x.T, Q_y)
    U, S, Vh = torch.linalg.svd(C)
    
    Wx = torch.linalg.solve(R_x, U)
    Wy = torch.linalg.solve(R_y, Vh.T)
    
    Wy_sub = Wy[:, :dim]
    
    C_y_train = torch.matmul(Yt, Wy_sub)
    M = torch.linalg.lstsq(C_y_train, Xt).solution
    
    return X_mean, Y_mean, Wy_sub, M

def project_list(Y_list, X_mean, Y_mean, Wy_sub, M, device):
    X_recon_list = []
    for Y_utt in Y_list:
        Y_c = torch.tensor(Y_utt - Y_mean, dtype=torch.float32, device=device)
        C_y = torch.matmul(Y_c, Wy_sub)
        X_pred_c = torch.matmul(C_y, M)
        X_recon_list.append(X_pred_c.cpu().numpy() + X_mean)
    return X_recon_list

def prepare_data_zspeaker(X_raw_list, X_recon_list):
    # Z_speaker = Mean(X_raw - X_recon) per utterance
    z_speaker_list = []
    for raw, recon in zip(X_raw_list, X_recon_list):
        residual = raw - recon
        z_spk = np.mean(residual, axis=0)
        z_speaker_list.append(z_spk)
    return z_speaker_list

def prepare_probing(z_speaker_list, X_recon_list, labels_ppg_list, speaker_ids_list):
    # Speaker Probing using Z_speaker directly
    y_spk = np.array(speaker_ids_list)
    X_spk = np.array(z_speaker_list)
    
    # PPG Probing using repeated Z_speaker and Z_content
    X_ppg_frames_spk = []
    X_ppg_frames_content = []
    y_ppg_frames = []
    
    np.random.seed(42)
    for z_spk, x_recon, p_class in zip(z_speaker_list, X_recon_list, labels_ppg_list):
        T = p_class.shape[0]
        if T > 100:
            idx = np.random.choice(T, size=100, replace=False)
        else:
            idx = np.arange(T)
            
        z_spk_repeated = np.tile(z_spk, (len(idx), 1))
        
        X_ppg_frames_spk.append(z_spk_repeated)
        X_ppg_frames_content.append(x_recon[idx])
        y_ppg_frames.append(p_class[idx])
        
    X_ppg_spk = np.concatenate(X_ppg_frames_spk, axis=0)
    X_ppg_content = np.concatenate(X_ppg_frames_content, axis=0)
    y_ppg = np.concatenate(y_ppg_frames, axis=0)
    
    return X_spk, y_spk, X_ppg_spk, X_ppg_content, y_ppg

def probe(X_tr, y_tr, X_te, y_te, max_iter=1000):
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=max_iter, random_state=42, n_jobs=-1).fit(scaler.transform(X_tr), y_tr)
    acc = accuracy_score(y_te, clf.predict(scaler.transform(X_te)))
    return acc

def compute_reconstruction_metrics(X_raw_list, X_recon_list, z_speaker_list):
    mse_content = []
    cos_content = []
    mse_combined = []
    cos_combined = []
    
    for raw, recon, z_spk in zip(X_raw_list, X_recon_list, z_speaker_list):
        # Combined = Z_content + Z_speaker
        combined = recon + z_spk
        
        # Flatten for metrics
        raw_f = raw.flatten()
        recon_f = recon.flatten()
        comb_f = combined.flatten()
        
        mse_content.append(np.mean((raw_f - recon_f)**2))
        mse_combined.append(np.mean((raw_f - comb_f)**2))
        
        # Cosine similarity is 1 - cosine distance
        cos_content.append(1 - cosine(raw_f, recon_f))
        cos_combined.append(1 - cosine(raw_f, comb_f))
        
    return np.mean(mse_content), np.mean(mse_combined), np.mean(cos_content), np.mean(cos_combined)

def main():
    warnings.filterwarnings("ignore")
    dataset_path = "/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean"
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    librispeech = LibriSpeech(root=Path(dataset_path), ext="flac")
    librispeech.generate_table()
    df_all = librispeech.table
    
    df_cca = df_all.sample(n=50, random_state=42)
    df_remaining = df_all.drop(df_cca.index)
    
    df_probe_train = df_remaining.sample(n=200, random_state=42)
    df_probe_test = df_remaining.drop(df_probe_train.index).sample(n=100, random_state=42)
    
    wavlm_base = WavLM(model_name="microsoft/wavlm-base", device=device)
    wavlm_asr = WavLM(model_name="patrickvonplaten/wavlm-libri-clean-100h-base-plus", device=device)
    L_TARGET = 8   
    L_SOURCE = 12  
    gpu_id = 0 if device == "cuda" else None
    
    print("\n1. Extraction des features...")
    X_cca, Y_cca, _, _ = extract_features_all(wavlm_base, wavlm_asr, df_cca['path'].tolist(), L_TARGET, L_SOURCE, "CCA Data", gpu=gpu_id)
    X_train_raw, Y_train, ppg_train, spk_train = extract_features_all(wavlm_base, wavlm_asr, df_probe_train['path'].tolist(), L_TARGET, L_SOURCE, "Train Data", gpu=gpu_id)
    X_test_raw, Y_test, ppg_test, spk_test = extract_features_all(wavlm_base, wavlm_asr, df_probe_test['path'].tolist(), L_TARGET, L_SOURCE, "Test Data", gpu=gpu_id)
    
    del wavlm_base, wavlm_asr
    torch.cuda.empty_cache()
    
    print("\n2. Calcul de la CCA et Séparation Contenu / Speaker...")
    DIM = 64
    xM, yM, Wy, M = learn_cca_mapping(X_cca, Y_cca, device, dim=DIM)
    
    # Z_content
    Z_content_train = project_list(Y_train, xM, yM, Wy, M, device)
    Z_content_test = project_list(Y_test, xM, yM, Wy, M, device)
    
    # Z_speaker = Mean(X_raw - Z_content)
    Z_speaker_train = prepare_data_zspeaker(X_train_raw, Z_content_train)
    Z_speaker_test = prepare_data_zspeaker(X_test_raw, Z_content_test)
    
    print("\n3. Validation du Z_speaker (Probing)...")
    spk_X_tr, spk_y_tr, ppg_X_tr_spk, ppg_X_tr_content, ppg_y_tr = prepare_probing(Z_speaker_train, Z_content_train, ppg_train, spk_train)
    spk_X_te, spk_y_te, ppg_X_te_spk, ppg_X_te_content, ppg_y_te = prepare_probing(Z_speaker_test, Z_content_test, ppg_test, spk_test)
    
    acc_spk = probe(spk_X_tr, spk_y_tr, spk_X_te, spk_y_te)
    
    acc_ppg_content = probe(ppg_X_tr_content, ppg_y_tr, ppg_X_te_content, ppg_y_te)
    acc_ppg_spk = probe(ppg_X_tr_spk, ppg_y_tr, ppg_X_te_spk, ppg_y_te)
    
    print("\n4. Validation de la Recombinaison (MSE / COS)...")
    mse_c, mse_comb, cos_c, cos_comb = compute_reconstruction_metrics(X_test_raw, Z_content_test, Z_speaker_test)
    
    print(f"\n--- RÉSULTATS DU DISENTANGLEMENT ---")
    print(f"[Z_speaker] Précision Identité Locuteur : {acc_spk*100:.2f}% (Idéal: Proche de 100%)")
    print(f"[Z_speaker] Précision Contenu Phonétique : {acc_ppg_spk*100:.2f}% (Idéal: Proche du hasard ~5-10%)")
    print(f"[Z_content] Précision Contenu Phonétique : {acc_ppg_content*100:.2f}%")
    
    print(f"\n--- RÉSULTATS DE RECONSTRUCTION DE WavLM RAW ---")
    print(f"MSE avec Z_content seul     : {mse_c:.4f}")
    print(f"MSE avec (Z_content + Z_spk): {mse_comb:.4f}  (Amélioration !)")
    print(f"COS avec Z_content seul     : {cos_c:.4f}")
    print(f"COS avec (Z_content + Z_spk): {cos_comb:.4f}  (Amélioration !)")

    with open(output_dir / "final_disentanglement_results.txt", "w") as f:
        f.write("--- DISENTANGLEMENT (Z_speaker = Mean(X_raw - Z_content)) ---\n")
        f.write(f"Speaker ID Acc: {acc_spk*100:.2f}%\n")
        f.write(f"PPG Acc: {acc_ppg_spk*100:.2f}%\n")
        f.write(f"\n--- RECONSTRUCTION ---\n")
        f.write(f"MSE (Content Only): {mse_c:.4f}\n")
        f.write(f"MSE (Combined): {mse_comb:.4f}\n")
        f.write(f"COS (Content Only): {cos_c:.4f}\n")
        f.write(f"COS (Combined): {cos_comb:.4f}\n")

if __name__ == "__main__":
    main()
