import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Imports from our une module
from une import WavLM, WhisperEncoderWrapper, LibriSpeech

def extract_aligned_latents(model_x, model_y, audio_paths):
    """
    Extrait les représentations des deux modèles et aligne leur dimension temporelle.
    Essentiel car Whisper padde ses entrées à 30 secondes (1500 frames),
    alors que WavLM renvoie le nombre exact de frames de l'audio.
    """
    with torch.no_grad():
        hx = model_x(str(audio_paths[0]))
        hy = model_y(str(audio_paths[0]))
        num_layers_x = len(hx)
        num_layers_y = len(hy)
        
    latents_x = {l: [] for l in range(num_layers_x)}
    latents_y = {l: [] for l in range(num_layers_y)}
    
    with torch.no_grad():
        for path in tqdm(audio_paths, desc="Extraction WavLM & Whisper"):
            hx = model_x(str(path))
            hy = model_y(str(path))
            
            # Alignement temporel: on tronque le padding de Whisper pour matcher WavLM
            T_x = hx[0].shape[1]
            T_y = hy[0].shape[1]
            min_T = min(T_x, T_y)
            
            for l in range(num_layers_x):
                latents_x[l].append(hx[l].squeeze(0)[:min_T].cpu().numpy())
            for l in range(num_layers_y):
                latents_y[l].append(hy[l].squeeze(0)[:min_T].cpu().numpy())
                
    flat_x = [np.concatenate(latents_x[l], axis=0) for l in range(num_layers_x)]
    flat_y = [np.concatenate(latents_y[l], axis=0) for l in range(num_layers_y)]
    
    return flat_x, flat_y

def run_gcca_whisper():
    dataset_path = "/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean"
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"1. Utilisation du device: {device}")
    
    librispeech = LibriSpeech(root=Path(dataset_path), ext="flac")
    librispeech.generate_table()
    
    num_audios = 400
    df_sampled = librispeech.table.sample(n=num_audios, random_state=42)
    audio_paths = df_sampled['path'].tolist()
    print(f"-> {num_audios} fichiers sélectionnés.")

    print("\n2. Chargement des modèles...")
    wavlm = WavLM(model_name="microsoft/wavlm-base", device=device)
    whisper = WhisperEncoderWrapper(model_name="openai/whisper-base", device=device)
    
    print("\n3. Extraction et Alignement temporel des représentations...")
    wavlm_layers, whisper_layers = extract_aligned_latents(wavlm, whisper, audio_paths)
    
    num_layers_x = len(wavlm_layers)
    num_layers_y = len(whisper_layers)
    total_frames = wavlm_layers[0].shape[0]
    print(f"\n-> Extraction terminée ! {total_frames} frames totales alignées.")
    
    print(f"-> WavLM: {num_layers_x} couches | Whisper: {num_layers_y} couches")

    print("\n4. Calcul de la matrice GCCA asymétrique...")
    cca_matrix = np.zeros((num_layers_x, num_layers_y))
    n_components = 64 
    
    for i in tqdm(range(num_layers_x), desc="WavLM Layers (Rows)"):
        X = wavlm_layers[i]
        for j in range(num_layers_y):
            Y = whisper_layers[j]
            
            X_c = X - np.mean(X, axis=0)
            Y_c = Y - np.mean(Y, axis=0)
            
            X_t = torch.tensor(X_c, dtype=torch.float32, device=device)
            Y_t = torch.tensor(Y_c, dtype=torch.float32, device=device)
            
            Q_X, _ = torch.linalg.qr(X_t)
            Q_Y, _ = torch.linalg.qr(Y_t)
            
            C = torch.matmul(Q_X.T, Q_Y)
            S = torch.linalg.svdvals(C)
            
            cca_scores = S.cpu().numpy()
            
            mean_score = np.mean(cca_scores[:n_components])
            cca_matrix[i, j] = mean_score
            
    print("\n5. Génération de la Heatmap...")
    sns.set_theme(style="white")
    plt.figure(figsize=(10, 10))
    
    # Echelle adaptée (0.5 - 1.0) comme pour les autres modèles Top 64
    sns.heatmap(
        cca_matrix, 
        annot=True,     # Ajout des valeurs pour plus de précision sur la petite matrice
        fmt=".2f",
        cmap="viridis", 
        vmin=0.5,
        vmax=1.0,
        cbar_kws={'label': f'Mean CCA Score (Top {n_components})'}
    )
    plt.title(f"GCCA Similarity: WavLM vs Whisper ({num_audios} audios)")
    plt.xlabel("Whisper Layers")
    plt.ylabel("WavLM Layers")
    
    output_file = output_dir / "gcca_wavlm_whisper_matrix.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🎉 Matrice générée et sauvegardée dans : {output_file}")

if __name__ == '__main__':
    run_gcca_whisper()
