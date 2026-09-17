<div align="center">

# Local Transport Mixtures for Zero-Shot Voice Conversion in SSL Spaces

**Samir Sadok** &emsp;&emsp; **Xavier Alameda-Pineda**  
*Inria, Univ. Grenoble Alpes, CNRS, Grenoble INP, LJK, France*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/pytorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange.svg)]()

[**[🎧 Audio Demonstration Website]**](https://samsad35.github.io/ZC-VC-demo/) &bull; [**[📄 Paper]**](#citation) &bull; [**[🚀 Quickstart]**](#-quickstart)

</div>

---

## 📦 Installation

### From GitHub
```bash
git clone https://github.com/samsad35/lwt-zcvc-code.git
cd lwt-zcvc-code
pip install -e .
```

### Requirements
- Python $\ge$ 3.9
- PyTorch $\ge$ 1.12
- Torchaudio $\ge$ 0.12
- SoundFile, SciPy, NumPy, Scikit-learn

*(Models such as WavLM-Large and HiFi-GAN are automatically downloaded and cached via PyTorch Hub on first run).*

---

## 🚀 Quickstart

### 1. Python API

```python
from boosted_lwt import VoiceConverter

# Initialize pipeline (loads WavLM, HiFi-GAN, and balanced 40-speaker UBM)
converter = VoiceConverter(device="cuda")

# 1. Zero-shot Voice Conversion with Boosted LWT
converter.convert(
    source="path/to/source.wav",
    target=["path/to/target_ref1.wav", "path/to/target_ref2.wav"],
    alpha=1.5,
    output_path="output_converted.wav"
)

# 2. Extract Pure Content (Canonical Neutral Voice)
converter.extract_pure_content(
    source="path/to/source.wav",
    output_path="output_pure_neutral.wav"
)
```

### 2. Command Line Interface (CLI)

#### Zero-Shot Voice Conversion:
```bash
boosted-lwt convert \
  --source source_speech.wav \
  --target target_ref1.wav target_ref2.wav \
  --output converted.wav \
  --alpha 1.5
```

#### Analytical Pure Content Extraction:
```bash
boosted-lwt pure-content \
  --source source_speech.wav \
  --output neutral_voice.wav
```

---

## 📊 Benchmark Results ($N=200$ Test Conversions)

Evaluated across 10 unseen target speakers, 20 source utterances from 10 distinct speakers, and $T=20$ reference utterances from LibriSpeech `test-clean`:

| Method | Approach Type | W2V2 CER ↓ | W2V2 WER ↓ | Whisper CER ↓ | ECAPA Sim ↑ | MOS ↑ | Loudness Corr ↑ | RTF ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Boosted LWT ($\alpha=1.5$ - Ours)** | **Speaker-Scaled Wasserstein** | **0.71%** | **2.68%** | **1.68%** | **0.749 ★** | 4.37 | 0.917 | 0.0371 |
| **Local Wasserstein (LWT - Ours)** | Locally Adaptive Wasserstein | 0.58% | 2.38% | 1.64% | 0.720 | **4.38** | 0.922 | 0.0239 |
| **Soft Local WCT (Ours)** | Locally Adaptive Statistical | **0.54% ★** | **2.28% ★** | **1.62% ★** | 0.631 | 4.36 | **0.925 ★** | 0.0067 |
| **LinearVC** *(Interspeech 2025)* | Global Linear Projection | 0.67% | 2.82% | 1.70% | 0.707 | 4.37 | 0.913 | **0.00002 ★** |
| **kNN-VC ($k=4$)** *(Interspeech 2023)* | Local Instance Averaging | 1.11% | 4.00% | 1.81% | 0.743 | **4.38** | 0.920 | 0.0002 |
| **Classic WCT** | Global Gaussian Matching | 1.00% | 3.23% | 1.61% | 0.648 | 4.35 | 0.906 | 0.0009 |

---

## 📁 Repository Structure

```
boosted-lwt/
├── checkpoints/
│   └── shared_clusters_k10_balanced_40spk.pt  # 40-speaker 50/50 balanced UBM
├── examples/
│   ├── quickstart.py                          # Minimal conversion example
│   └── extract_pure_content.py                # Pure content disentanglement example
├── src/
│   └── boosted_lwt/
│       ├── __init__.py                        # Public API
│       ├── models.py                          # WavLM & HiFi-GAN loading
│       ├── transport.py                       # Closed-form optimal transport algorithms
│       ├── background.py                      # UBM loader & Target representation builder
│       ├── pipeline.py                        # High-level VoiceConverter class
│       └── cli.py                             # Command-line interface
├── pyproject.toml                             # PEP 621 package build config
├── LICENSE                                    # MIT License
└── README.md
```

---

## 📄 Citation

If you use this code or find our work useful in your research, please cite:

```bibtex
@article{sadok2026local,
  title     = {Local Transport Mixtures for Zero-Shot Voice Conversion in SSL Spaces},
  author    = {Sadok, Samir and Alameda-Pineda, Xavier},
  journal   = {Under Review},
  year      = {2026}
}
```

---

## 📜 License

This project is released under the [MIT License](LICENSE).
