"""
High-Level User-Friendly Pipeline for Voice Conversion & Speech Disentanglement.
"""

from typing import List, Union, Optional
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

from .models import load_wavlm, load_hifigan, extract_features, vocode
from .background import load_universal_background, build_target_representation
from .transport import convert_lwt, convert_pure_content, convert_soft_local_wct, convert_classic_wct


class VoiceConverter:
    """
    High-level end-to-end Voice Conversion & Speech Disentanglement Pipeline.
    
    Example:
        >>> from boosted_lwt import VoiceConverter
        >>> converter = VoiceConverter(device='cuda')
        >>> # Zero-shot Voice Conversion with Boosted LWT
        >>> converted_audio = converter.convert(
        ...     source="source.wav",
        ...     target="target_reference.wav",
        ...     alpha=1.5,
        ...     output_path="converted.wav"
        ... )
        >>> # Extract Pure Content (Canonical Neutral Voice)
        >>> pure_audio = converter.extract_pure_content(
        ...     source="source.wav",
        ...     output_path="pure_neutral.wav"
        ... )
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        background_checkpoint: Optional[Union[str, Path]] = None,
        lazy_load: bool = False,
    ):
        if isinstance(device, str):
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
        self.device = torch.device(device)
        self.background_checkpoint = background_checkpoint

        self.wavlm = None
        self.hifigan = None
        self.ubm = None

        if not lazy_load:
            self._init_models()

    def _init_models(self):
        if self.wavlm is None:
            self.wavlm = load_wavlm(self.device)
        if self.hifigan is None:
            self.hifigan = load_hifigan(self.device)
        if self.ubm is None:
            self.ubm = load_universal_background(self.background_checkpoint, self.device)

    def convert(
        self,
        source: Union[str, Path, torch.Tensor],
        target: Union[str, Path, List[Union[str, Path]]],
        alpha: float = 1.5,
        beta: float = 20.0,
        output_path: Optional[Union[str, Path]] = None,
        method: str = "boosted_lwt",
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        Convert source speech to match target speaker voice.
        
        Args:
            source: Path to source audio or waveform tensor
            target: Path or list of paths to target reference audio files
            alpha: Covariance boost factor (alpha=1.5 recommended for Boosted LWT)
            beta: Softmax temperature for phonetic clustering
            output_path: Optional path to save converted 16kHz WAV file
            method: 'boosted_lwt', 'lwt', 'soft_local_wct', or 'classic_wct'
            sample_rate: Output sample rate (16000 Hz)
            
        Returns:
            audio_out: Synthesized audio numpy array in [-1.0, 1.0]
        """
        self._init_models()

        if isinstance(target, (str, Path)):
            target_list = [target]
        else:
            target_list = list(target)

        # 1. Build target representation
        target_model = build_target_representation(
            target_audio_paths_or_tensors=target_list,
            wavlm=self.wavlm,
            cents_shared_norm=self.ubm['cents_shared_norm'],
            cov_shared_k=self.ubm['cov_shared_k'],
            device=self.device,
            beta=beta,
            K_clusters=self.ubm['K_clusters'],
        )

        # 2. Extract source representations
        xs_t, _ = extract_features(source, self.wavlm, self.device)

        # 3. Apply conversion
        if method in ("boosted_lwt", "lwt"):
            eff_alpha = 1.0 if method == "lwt" else alpha
            x_hat = convert_lwt(
                xs_t=xs_t,
                target_model=target_model,
                cents_shared_norm=self.ubm['cents_shared_norm'],
                cov_shared_k=self.ubm['cov_shared_k'],
                alpha=eff_alpha,
                beta=beta,
                K_clusters=self.ubm['K_clusters'],
            )
        elif method == "soft_local_wct":
            x_hat = convert_soft_local_wct(
                xs_t=xs_t,
                target_model=target_model,
                cents_shared_norm=self.ubm['cents_shared_norm'],
                cov_shared_k=self.ubm['cov_shared_k'],
                beta=beta,
                K_clusters=self.ubm['K_clusters'],
            )
        elif method == "classic_wct":
            x_hat = convert_classic_wct(xs_t=xs_t, target_model=target_model)
        else:
            raise ValueError(f"Unknown conversion method: {method}. Choose from 'boosted_lwt', 'lwt', 'soft_local_wct', 'classic_wct'.")

        # 4. Vocode to waveform
        audio_out = vocode(x_hat, self.hifigan, self.device)

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_p), audio_out, sample_rate)

        return audio_out

    def extract_pure_content(
        self,
        source: Union[str, Path, torch.Tensor],
        output_path: Optional[Union[str, Path]] = None,
        beta: float = 20.0,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        Extract Pure Content (Analytical Speech Disentanglement).
        
        Projects speech onto the 40-speaker gender-balanced universal background.
        Produces a canonical neutral voice (~160 Hz) preserving 100% phonetic intelligibility.
        
        Args:
            source: Path to source audio file or waveform tensor
            output_path: Optional path to save neutral audio WAV file
            beta: Softmax temperature
            sample_rate: Output sample rate (16000 Hz)
            
        Returns:
            audio_out: Synthesized neutral audio numpy array
        """
        self._init_models()

        xs_t, _ = extract_features(source, self.wavlm, self.device)

        x_pure = convert_pure_content(
            xs_t=xs_t,
            cents_shared=self.ubm['cents_shared'],
            cents_shared_norm=self.ubm['cents_shared_norm'],
            cov_shared_k=self.ubm['cov_shared_k'],
            beta=beta,
            K_clusters=self.ubm['K_clusters'],
        )

        audio_out = vocode(x_pure, self.hifigan, self.device)

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_p), audio_out, sample_rate)

        return audio_out
