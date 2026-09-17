import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Imports from our une module
from une import WavLM, Wav2Vec2Encoder, LibriSpeech

def extract_all_layers_latents(model_wrapper, audio_paths, desc_msg):
    """
    Extrait et concatène toutes les frames pour toutes les couches d'un modèle.
    """
    # Initialize a list to hold the concatenated representations for each layer
    all_layers_latents = None
    
    for path in tqdm(audio_paths, desc=desc_msg):
        # The wrapper returns a tuple of tensors, one for each layer
        # Shape of each layer tensor: (batch, seq_len, hidden_size)
        hidden_states = model_wrapper(str(path))
        
        # Squeeze batch dimension and convert to numpy
        layer_frames = [layer.squeeze(0).cpu().numpy() for layer in hidden_states]
        
        if all_layers_latents is None:
            # Initialize empty lists for each layer
            all_layers_latents = [[] for _ in range(len(layer_frames))]
            
        for layer_idx, frames in enumerate(layer_frames):
            all_layers_latents[layer_idx].append(frames)
            
    # Concatenate frames along the time axis
    return [np.concatenate(layer_list, axis=0) for layer_list in all_layers_latents]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Utilisation de {device} ---")
    
    print("1. Initialisation du dataset LibriSpeech...")
    dataset_path = "/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean"
    librispeech = LibriSpeech(root=Path(dataset_path), ext="flac")
    librispeech.generate_table()
    
    # Échantillonnage de 400 fichiers audio (même config que le benchmark de gaussianité)
    num_audios = 400
    df_sampled = librispeech.table.sample(n=num_audios, random_state=42)
    audio_paths = df_sampled['path'].tolist()
    print(f"-> {num_audios} fichiers sélectionnés.")

    print("\n2. Chargement des modèles...")
    wavlm = WavLM(model_name="microsoft/wavlm-base", device=device)
    wav2vec2 = Wav2Vec2Encoder(model_name="facebook/wav2vec2-base", device=device)
    
    print("\n3. Extraction des représentations (ça peut prendre quelques minutes)...")
    wavlm_layers = extract_all_layers_latents(wavlm, audio_paths, "Extraction WavLM")
    wav2vec_layers = extract_all_layers_latents(wav2vec2, audio_paths, "Extraction Wav2Vec2")
    
    num_layers = len(wavlm_layers)
    total_frames = wavlm_layers[0].shape[0]
    print(f"\n-> Extraction terminée ! {total_frames} frames totales extraites pour {num_layers} couches.")
    
    print("\n4. Calcul de la matrice GCCA (SVD exacte) couche par couche...")
    cca_matrix = np.zeros((num_layers, num_layers))
    n_components = 64 
    
    for i in tqdm(range(num_layers), desc="WavLM Layers"):
        X = wavlm_layers[i]
        for j in range(num_layers):
            Y = wav2vec_layers[j]
            
            # Fast PyTorch CCA using QR and SVD
            # On déplace les matrices sur GPU si dispo pour que la SVD soit instantanée
            X_t = torch.tensor(X, device=device)
            Y_t = torch.tensor(Y, device=device)
            
            # Normalisation (Centrage + Réduction)
            # Mathématiquement la CCA est invariante au changement d'échelle, 
            # mais numériquement c'est très important pour la stabilité de la SVD.
            X_t = X_t - X_t.mean(dim=0)
            X_t = X_t / (X_t.std(dim=0) + 1e-8)
            
            Y_t = Y_t - Y_t.mean(dim=0)
            Y_t = Y_t / (Y_t.std(dim=0) + 1e-8)
            
            # QR decomposition
            Q_x, R_x = torch.linalg.qr(X_t)
            Q_y, R_y = torch.linalg.qr(Y_t)
            
            # SVD on the inner product of the orthogonal bases
            M = Q_x.T @ Q_y
            U, S, V = torch.linalg.svd(M)
            
            # S contains the canonical correlations!
            corrs = S[:n_components].cpu().numpy()
            cca_matrix[i, j] = np.mean(corrs)

    print("\n5. Génération de la Heatmap...")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cca_matrix, 
        annot=False, 
        cmap="viridis", 
        vmin=0.5,
        vmax=1.0,
        cbar_kws={'label': f'Mean CCA Score (Top {n_components})'}
    )
    plt.title(f"GCCA Similarity: WavLM vs Wav2Vec2 ({num_audios} audios LibriSpeech)")
    plt.xlabel("Wav2Vec2 Layers")
    plt.ylabel("WavLM Layers")
    
    output_file = output_dir / "gcca_wavlm_wav2vec2_matrix.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🎉 Matrice réelle générée et sauvegardée dans : {output_file}")

if __name__ == "__main__":
    main()
