"""
Boosted Local Wasserstein Transport (Boosted LWT).

Zero-Shot Voice Conversion via Closed-Form Optimal Transport
and Analytical Speech Disentanglement in Self-Supervised Spaces.

ICASSP 2026 Submission.
"""

from .pipeline import VoiceConverter
from .transport import (
    compute_bures_wasserstein_map,
    convert_lwt,
    convert_pure_content,
    convert_soft_local_wct,
    convert_classic_wct,
    project_psd,
)
from .models import (
    load_wavlm,
    load_hifigan,
    extract_features,
    vocode,
)
from .background import (
    load_universal_background,
    build_target_representation,
)

__version__ = "0.1.0"

__all__ = [
    "VoiceConverter",
    "compute_bures_wasserstein_map",
    "convert_lwt",
    "convert_pure_content",
    "convert_soft_local_wct",
    "convert_classic_wct",
    "project_psd",
    "load_wavlm",
    "load_hifigan",
    "extract_features",
    "vocode",
    "load_universal_background",
    "build_target_representation",
]
