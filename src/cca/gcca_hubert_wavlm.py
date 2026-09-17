import torch
import torchaudio
from transformers import HubertModel, WavLMModel
from sklearn.cross_decomposition import CCA
import numpy as np

def extract_features(model, waveform, sample_rate, target_sr=16000):
    # Resample if needed
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
    
    # Ensure waveform is 1D (mono)
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0)
    
    # Add batch dimension
    waveform = waveform.unsqueeze(0)
    
    with torch.no_grad():
        # Get hidden states
        outputs = model(waveform, output_hidden_states=True)
        # Use the last hidden state for CCA, shape: (batch_size, sequence_length, hidden_size)
        hidden_states = outputs.last_hidden_state
        
    return hidden_states.squeeze(0).numpy()

def main():
    print("Loading models...")
    # Load HuBERT and WavLM from huggingface hub
    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base")
    
    hubert_model.eval()
    wavlm_model.eval()
    
    print("Generating random audio for experiment (replace with real audio)...")
    # Generate 3 seconds of random noise at 16kHz
    sample_rate = 16000
    duration = 3.0
    waveform = torch.randn(int(sample_rate * duration))
    
    print("Extracting features...")
    hubert_features = extract_features(hubert_model, waveform, sample_rate)
    wavlm_features = extract_features(wavlm_model, waveform, sample_rate)
    
    # Check that lengths match
    min_len = min(hubert_features.shape[0], wavlm_features.shape[0])
    X = hubert_features[:min_len]
    Y = wavlm_features[:min_len]
    
    print(f"Features shape for CCA: {X.shape}")
    
    print("Computing CCA (n_components=10)...")
    # Initialize CCA. In GCCA with 2 views, it's equivalent to standard CCA.
    n_components = 10
    cca = CCA(n_components=n_components)
    
    # Fit the CCA model
    cca.fit(X, Y)
    
    # Transform the data
    X_c, Y_c = cca.transform(X, Y)
    
    # Calculate correlations (scores) for each component
    correlations = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]
    mean_correlation = np.mean(correlations)
    
    print("\n--- CCA/GCCA Results ---")
    print(f"Mean Correlation Score across {n_components} components: {mean_correlation:.4f}")
    for i, corr in enumerate(correlations):
        print(f"Component {i+1}: {corr:.4f}")

if __name__ == "__main__":
    main()
