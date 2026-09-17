import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import anderson, normaltest, shapiro

# On réutilise ton loader local pour LibriSpeech et ta classe de tokenization
from une import LibriSpeech
from une import MySpeechTokenizer 

def test_gaussianity_projections(X, num_projections=5000, sample_size=250):
    """
    Évalue la gaussianité de la matrice latente X via des projections 1D aléatoires (méthode UNE).
    """
    N, D = X.shape
    
    # Normalisation des features (centrage et réduction)
    X_centered = X - np.mean(X, axis=0)
    X_scaled = X_centered / (np.std(X_centered, axis=0) + 1e-8)
    
    ad_passes = 0
    dp_passes = 0
    sw_passes = 0
    
    print(f"\n--- RÉSULTATS DU TEST DE GAUSSIANITÉ ({num_projections} projections) ---")
    print(f"Calcul sur {N} frames latentes (Dimension d'origine: {D})...")
    
    for _ in tqdm(range(num_projections), desc="Calcul des projections aléatoires"):
        # 1. Générer une direction de projection aléatoire sur la sphère unité
        direction = np.random.normal(0, 1, size=(D,))
        direction /= np.linalg.norm(direction)
        
        # 2. Projeter les données en 1D
        projection = X_scaled @ direction
        
        # 3. Sous-échantillonner aléatoirement (requis pour Shapiro-Wilk et stabilité AD)
        sub_sample = np.random.choice(projection, size=sample_size, replace=False)
        
        # --- TEST 1 : Anderson-Darling ---
        res_ad = anderson(sub_sample, dist='norm')
        if res_ad.statistic < res_ad.critical_values[2]: # Seuil de significativité à 5%
            ad_passes += 1
            
        # --- TEST 2 : D'Agostino-Pearson ---
        stat_dp, p_val_dp = normaltest(sub_sample)
        if p_val_dp > 0.05:
            dp_passes += 1
            
        # --- TEST 3 : Shapiro-Wilk ---
        stat_sw, p_val_sw = shapiro(sub_sample)
        if p_val_sw > 0.05:
            sw_passes += 1

    print("\n" + "="*60)
    print(f"📊 TAUX DE GAUSSIANITÉ GLOBALE (Espace 'e' SpeechTokenizer) :")
    print(f"   -> Anderson-Darling (AD)    : {ad_passes / num_projections * 100:.2f}% (Objectif papier: ~90-95%)")
    print(f"   -> D'Agostino-Pearson (DP)  : {dp_passes / num_projections * 100:.2f}%")
    print(f"   -> Shapiro-Wilk (SW)        : {sw_passes / num_projections * 100:.2f}%")
    print("="*60)
    
    return {
        "ad_pct": ad_passes / num_projections,
        "dp_pct": dp_passes / num_projections,
        "sw_pct": sw_passes / num_projections
    }

def test_speechtokenizer_gaussianity(dataset_root: str, num_audios: int = 50, device: str = "cuda"):
    print("--- 1. Initialisation de MySpeechTokenizer ---")
    torch_device = torch.device(device)
    tokenizer = MySpeechTokenizer(device=torch_device)

    print("\n--- 2. Indexation locale du dataset LibriSpeech ---")
    librispeech = LibriSpeech(root=Path(dataset_root), ext="flac")
    librispeech.generate_table()
    
    # Échantillonnage identique à tes runs précédents
    df_sampled = librispeech.table.sample(n=min(num_audios, len(librispeech.table)), random_state=42)
    paths = df_sampled['path'].tolist()
    
    all_latents = []

    print("\n--- 3. Extraction de l'espace latent continu (e) ---")
    with torch.no_grad():
        for p in tqdm(paths, desc="Extraction via SpeechTokenizer"):
            # Pré-traitement, chargement et resampling de l'audio
            wav = tokenizer.preprocess(path_wav=str(p)).to(torch_device)
            wav = wav.unsqueeze(0) # (1, 1, T)
            
            # Extraction à la sortie du bloc de convolutions, JUSTE AVANT la RVQ
            e = tokenizer.model.encoder(wav) # Shape: (1, D_latent, T_frames)
            
            # Reshape en (T_frames, D_latent) et conversion numpy
            latents_reshaped = e.squeeze(0).transpose(0, 1).cpu().numpy()
            all_latents.append(latents_reshaped)

    # Concaténation globale au niveau des frames temporelles
    Z = np.concatenate(all_latents, axis=0)
    print(f"\n👉 Espace latent 'e' extrait. Nombre total de frames : {Z.shape[0]}, Dimensions (D) : {Z.shape[1]}")

    # 4. Évaluation par projections aléatoires multivariées
    test_gaussianity_projections(Z, num_projections=5000, sample_size=250)
    

if __name__ == "__main__":
    DATASET_PATH = "/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    test_speechtokenizer_gaussianity(dataset_root=DATASET_PATH, num_audios=50, device=DEVICE)