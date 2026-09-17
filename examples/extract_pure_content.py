"""
Analytical Speech Disentanglement Example: Extract Pure Content.

Usage:
  python examples/extract_pure_content.py --source path/to/audio.wav --output neutral.wav
"""

import argparse
from pathlib import Path
from boosted_lwt import VoiceConverter


def main():
    parser = argparse.ArgumentParser(description="Extract Pure Content (Canonical Neutral Voice) using Boosted LWT")
    parser.add_argument("--source", "-s", type=str, default="output/audio_samples/boosted_lwt_demo/pair1_1284_to_1089/source.wav")
    parser.add_argument("--output", "-o", type=str, default="output/pure_neutral_extracted.wav")
    parser.add_argument("--device", "-d", type=str, default="cuda")
    args = parser.parse_args()

    print("=== Analytical Speech Disentanglement ===")
    print(f"Source file: {args.source}")
    print(f"Device: {args.device}")

    converter = VoiceConverter(device=args.device)
    
    print("Projecting speech onto 40-speaker gender-neutral universal background...")
    converter.extract_pure_content(
        source=args.source,
        output_path=args.output
    )

    print(f"[Success] Neutral canonical audio saved to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
