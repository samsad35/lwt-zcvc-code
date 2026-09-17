import torch
import numpy as np
from pathlib import Path

from une import LibriSpeech, WavLM, Wav2Vec2Encoder, HuBERTEncoder, WhisperEncoderWrapper
from benchmark_rank import compute_effective_rank
from benchmark_gaussianity import test_gaussianity_projections

def extract_latents(model, audio_paths):
    with torch.no_grad():
        first_hidden_states = model(str(audio_paths[0]))
        num_layers = len(first_hidden_states)
        layers_to_test = list(range(num_layers))
        
    latents_per_layer = {layer: [] for layer in layers_to_test}
    with torch.no_grad():
        for path in audio_paths:
            all_hidden_states = model(str(path))
            for layer in layers_to_test:
                latent_frames = all_hidden_states[layer].squeeze(0).cpu().numpy()
                latents_per_layer[layer].append(latent_frames)
                
    flat_latents = {}
    for layer in layers_to_test:
        flat_latents[layer] = np.concatenate(latents_per_layer[layer], axis=0)
    return flat_latents, layers_to_test

DATASET_PATH = "/scratch2/pictor/ssadok/dataset/audio/LibriSpeech/test-clean"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

librispeech = LibriSpeech(root=Path(DATASET_PATH), ext="flac")
librispeech.generate_table()
df_sampled = librispeech.table.sample(n=5, random_state=42)
audio_paths = df_sampled['path'].tolist()

models = {
    "WavLM": WavLM("microsoft/wavlm-base", DEVICE),
    "Wav2Vec2": Wav2Vec2Encoder("facebook/wav2vec2-base", DEVICE),
    "HuBERT": HuBERTEncoder("facebook/hubert-base-ls960", DEVICE),
    "Whisper": WhisperEncoderWrapper("openai/whisper-base", DEVICE)
}

print("=== RESULTATS (5 audios) ===")
for name, model in models.items():
    flat_latents, layers = extract_latents(model, audio_paths)
    ranks = []
    gauss = []
    for l in layers:
        mat = flat_latents[l]
        r = compute_effective_rank(mat)
        g = test_gaussianity_projections(mat, 50, 150)["ad_pct"] * 100
        ranks.append(f"{r:.0f}")
        gauss.append(f"{g:.0f}%")
    
    print(f"\n{name} Rank: " + " -> ".join(ranks))
    print(f"{name} Gauss: " + " -> ".join(gauss))
