import torch
import torchaudio
from pathlib import Path
import pandas as pd
from transformers import Wav2Vec2Model, WavLMModel, HubertModel
from scipy.stats import anderson
import numpy as np

class LibriSpeech:
    def __init__(self, root, ext="flac"):
        self.root = Path(root)
        self.ext = ext
        self.table = None
        
    def generate_table(self):
        paths = list(self.root.rglob(f"*.{self.ext}"))
        self.table = pd.DataFrame({'path': paths})

class BaseModelWrapper:
    def __init__(self, model_class, model_name, device):
        self.device = device
        self.model = model_class.from_pretrained(model_name).to(device)
        self.model.eval()
        
    def __call__(self, path, layer_idx=-1):
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
        # Ensure it's 2D (batch, time)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() > 2:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        waveform = waveform.to(self.device)
        with torch.no_grad():
            outputs = self.model(waveform, output_hidden_states=True)
            
        # Returns a tuple of hidden states, one for each layer
        return outputs.hidden_states
        
    def eval(self):
        self.model.eval()

class WavLM(BaseModelWrapper):
    def __init__(self, model_name, device):
        super().__init__(WavLMModel, model_name, device)

class Wav2Vec2Encoder(BaseModelWrapper):
    def __init__(self, model_name, device):
        super().__init__(Wav2Vec2Model, model_name, device)

class HuBERTEncoder(BaseModelWrapper):
    def __init__(self, model_name, device):
        super().__init__(HubertModel, model_name, device)

from transformers import WhisperModel, WhisperFeatureExtractor

class WhisperEncoderWrapper:
    def __init__(self, model_name, device):
        self.device = device
        self.model = WhisperModel.from_pretrained(model_name).get_encoder().to(device)
        self.model.eval()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        
    def __call__(self, path, layer_idx=-1):
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0) # Mono
            
        # Whisper attend des log-mel spectrogrammes, pas du raw audio
        inputs = self.feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_features, output_hidden_states=True)
            
        return outputs.hidden_states
        
    def eval(self):
        self.model.eval()

def test_gaussianity_projections(matrix, num_projections=100, sample_size=250):
    if len(matrix) == 0:
        return {"ad_pct": 0.0}
        
    # Standardize matrix
    matrix = (matrix - np.mean(matrix, axis=0)) / (np.std(matrix, axis=0) + 1e-8)
    
    hidden_size = matrix.shape[1]
    projections = np.random.randn(hidden_size, num_projections)
    projections = projections / np.linalg.norm(projections, axis=0, keepdims=True)
    
    projected = matrix @ projections
    
    if sample_size and len(matrix) > sample_size:
        indices = np.random.choice(len(matrix), sample_size, replace=False)
        projected = projected[indices]
    
    gaussian_count = 0
    for i in range(num_projections):
        proj_data = projected[:, i]
        result = anderson(proj_data, dist='norm')
        # index 2 corresponds to 5% significance level for normal dist in scipy
        if result.statistic < result.critical_values[2]:
            gaussian_count += 1
            
    return {"ad_pct": gaussian_count / num_projections}
