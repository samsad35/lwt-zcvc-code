import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# On importe tes classes et la fonction statistique
from une import LibriSpeech
from une import WavLM, Wav2Vec2Encoder, HuBERTEncoder, WhisperEncoderWrapper
from une import test_gaussianity_projections


def extract_latents_for_model(model, audio_paths, desc_msg):
    """Extrait et concatène toutes les frames pour un modèle."""
    # Détection automatique du nombre de couches
    with torch.no_grad():
        first_hidden_states = model(str(audio_paths[0]))
        num_layers = len(first_hidden_states)
        layers_to_test = list(range(num_layers))
        
    latents_per_layer = {layer: [] for layer in layers_to_test}
    
    with torch.no_grad():
        for path in tqdm(audio_paths, desc=desc_msg):
            all_hidden_states = model(str(path))
            for layer in layers_to_test:
                latent_frames = all_hidden_states[layer].squeeze(0).cpu().numpy()
                latents_per_layer[layer].append(latent_frames)
                
    # Concaténation temporelle (Frame-level)
    flat_latents = {}
    for layer in layers_to_test:
        flat_latents[layer] = np.concatenate(latents_per_layer[layer], axis=0)
        
    return flat_latents, layers_to_test


def run_generic_comparative_experiment(
    dataset_root: str, 
    models_config: dict, 
    num_audios_to_test: int = 100, 
    device: str = "cuda"
):
    print(f"--- Chargement du dataset LibriSpeech depuis {dataset_root} ---")
    librispeech = LibriSpeech(root=Path(dataset_root), ext="flac")
    librispeech.generate_table()
    
    df_sampled = librispeech.table.sample(n=num_audios_to_test, random_state=42)
    audio_paths = df_sampled['path'].tolist()
    
    all_models_latents = {}

    for idx, (display_name, config) in enumerate(models_config.items(), 1):
        print(f"\n--- [{idx}/{len(models_config)}] Initialisation de {display_name} sur {device} ---")
        
        model_class = config["class"]
        model_hf_name = config["model_name"]
        
        model = model_class(model_name=model_hf_name, device=device)
        model.eval()
        
        flat_latents, model_layers = extract_latents_for_model(
            model, audio_paths, f"Extraction {display_name}"
        )
        all_models_latents[display_name] = {
            "latents": flat_latents,
            "layers": model_layers
        }
        
        del model
        torch.cuda.empty_cache()

    print("\n--- Analyse statistique des espaces (Frame-Level) ---")
    plot_data = [] 
    
    for display_name in models_config.keys():
        for layer in all_models_latents[display_name]["layers"]:
            print(f"👉 Évaluation {display_name} - COUCHE {layer}")
            
            matrix_to_test = all_models_latents[display_name]["latents"][layer]
            stats = test_gaussianity_projections(matrix_to_test, num_projections=100, sample_size=250)
            
            plot_data.append({
                "Couche": f"Couche {layer}",
                "Modèle": display_name,
                "Gaussianité (AD %)": stats["ad_pct"] * 100
            })

    print("\n--- Generating Comparison Plot ---")
    df_plot = pd.DataFrame(plot_data)
    
    sns.set_theme(style="ticks", rc={
        "font.family": "sans-serif",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333"
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    custom_palette = sns.cubehelix_palette(n_colors=len(models_config), start=.5, rot=-.5, light=.7, dark=.3)
    
    df_plot["Layer_Idx"] = df_plot["Couche"].str.extract('(\d+)').astype(int)
    
    sns.lineplot(
        data=df_plot, 
        x="Layer_Idx", 
        y="Gaussianité (AD %)", 
        hue="Modèle", 
        style="Modèle", 
        markers=True, 
        markersize=10, 
        linewidth=2.5,
        palette=custom_palette,
        ax=ax
    )
    
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_color('#333333')
    ax.spines['right'].set_color('#333333')
    
    ax.set_title("Evolution of Latent Space Gaussianity across Layers", fontsize=14, fontweight='semibold', pad=20)
    ax.set_xlabel("Transformer Layer Index", fontsize=11, labelpad=10)
    ax.set_ylabel("Gaussian Fraction (AD Test %)", fontsize=11, labelpad=10)
    
    all_layer_indices = sorted(df_plot["Layer_Idx"].unique())
    ax.set_xticks(all_layer_indices)
    ax.set_xticklabels([f"L{l}" for l in all_layer_indices])
    
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.5, which="major", axis="y")
    
    ax.legend(title="Models", loc="lower left", frameon=True, facecolor="white", edgecolor="none")
    
    plt.tight_layout()
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_png = output_dir / "une_comparison_multimodels.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"🎉 Plot successfully saved to: '{output_png}'!")
    plt.close()


if __name__ == '__main__':
    DATASET_PATH = "/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    MODELS_REGISTRY = {
        "WavLM (microsoft)": {
            "class": WavLM, 
            "model_name": "microsoft/wavlm-base"
        },
        "Wav2Vec2 (facebook)": {
            "class": Wav2Vec2Encoder, 
            "model_name": "facebook/wav2vec2-base"
        },
        "HuBERT (facebook)": {
            "class": HuBERTEncoder, 
            "model_name": "facebook/hubert-base-ls960"
        },
        "Whisper (openai)": {
            "class": WhisperEncoderWrapper, 
            "model_name": "openai/whisper-base"
        }
    }
    
    run_generic_comparative_experiment(
        dataset_root=DATASET_PATH,
        models_config=MODELS_REGISTRY,
        num_audios_to_test=100,
        device=DEVICE
    )