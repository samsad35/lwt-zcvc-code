"""
Model loading and audio feature extraction for Boosted LWT.

Uses:
  - WavLM-Large (Layer 6 representations, 1024-dim)
  - HiFi-GAN Vocoder (pretrained on WavLM-Large representations)
"""

from typing import Tuple, Union
from pathlib import Path
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F


def load_wavlm(device: Union[str, torch.device] = "cuda") -> torch.nn.Module:
    """
    Load pretrained WavLM-Large model from PyTorch Hub.
    Layer 6 is extracted for optimal linguistic representation.
    """
    device = torch.device(device)
    wavlm = torch.hub.load('bshall/knn-vc', 'wavlm_large', trust_repo=True)
    wavlm = wavlm.to(device).eval()
    return wavlm


def load_hifigan(device: Union[str, torch.device] = "cuda") -> torch.nn.Module:
    """
    Load pretrained HiFi-GAN vocoder matched to WavLM-Large representations.
    """
    device = torch.device(device)
    hifigan, _ = torch.hub.load('bshall/knn-vc', 'hifigan_wavlm', trust_repo=True, prematched=True)
    hifigan = hifigan.to(device).eval()
    return hifigan


def extract_features(
    audio_path_or_wav: Union[str, Path, torch.Tensor],
    wavlm: torch.nn.Module,
    device: Union[str, torch.device] = "cuda",
    target_sr: int = 16000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract 1024-dimensional WavLM Layer 6 representations from an audio file or waveform tensor.
    
    Returns:
        features: (T, 1024) tensor of representations on device
        wav_16k: (T_samples,) 16kHz audio tensor on CPU
    """
    device = torch.device(device)
    
    if isinstance(audio_path_or_wav, (str, Path)):
        wav, sr = torchaudio.load(str(audio_path_or_wav))
    elif isinstance(audio_path_or_wav, torch.Tensor):
        wav = audio_path_or_wav
        sr = target_sr
    else:
        raise TypeError(f"Expected str, Path or torch.Tensor, got {type(audio_path_or_wav)}")

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    elif wav.dim() == 2 and wav.size(0) > 1:
        # Convert to mono
        wav = torch.mean(wav, dim=0, keepdim=True)

    if sr != target_sr:
        wav = F.resample(wav, sr, target_sr)

    wav_device = wav.to(device)
    with torch.no_grad():
        feat, _ = wavlm.extract_features(wav_device, output_layer=6)

    return feat.squeeze(0), wav.squeeze(0).cpu()


def vocode(
    features: Union[torch.Tensor, np.ndarray],
    hifigan: torch.nn.Module,
    device: Union[str, torch.device] = "cuda",
    peak_normalize: bool = True,
) -> np.ndarray:
    """
    Synthesize 16kHz audio waveform from 1024-d WavLM features using HiFi-GAN.
    
    Returns:
        audio: (N_samples,) numpy array of audio samples in [-1.0, 1.0]
    """
    device = torch.device(device)
    with torch.inference_mode():
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32, device=device)
        else:
            features = features.to(device)

        if features.dim() == 2:
            features = features.unsqueeze(0)

        audio = hifigan(features).squeeze().cpu().numpy()

        if peak_normalize:
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val * 0.99

        return audio
