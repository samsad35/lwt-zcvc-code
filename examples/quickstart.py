"""
Minimal Quickstart Example for Boosted LWT Voice Conversion.

Usage:
  python examples/quickstart.py
"""

from pathlib import Path
from boosted_lwt import VoiceConverter

def main():
    # 1. Initialize converter (automatically loads WavLM-Large, HiFi-GAN, and balanced UBM)
    print("Initializing VoiceConverter...")
    converter = VoiceConverter(device="cuda")

    # Sample paths from demo
    source_wav = "output/audio_samples/boosted_lwt_demo/pair1_1284_to_1089/source.wav"
    target_wav = "output/audio_samples/boosted_lwt_demo/pair1_1284_to_1089/target_ref.wav"

    if not Path(source_wav).exists():
        print(f"Please provide sample audio files to run this example.")
        return

    # 2. Perform Voice Conversion with Boosted LWT (alpha=1.5)
    print("\n--- Running Boosted LWT Voice Conversion (alpha=1.5) ---")
    out_vc = "output/quickstart_converted_boosted.wav"
    converter.convert(
        source=source_wav,
        target=target_wav,
        alpha=1.5,
        output_path=out_vc
    )
    print(f"Converted audio saved to: {out_vc}")

    # 3. Extract Pure Content (Canonical Neutral Voice ~160 Hz)
    print("\n--- Extracting Analytical Pure Content ---")
    out_pure = "output/quickstart_pure_neutral.wav"
    converter.extract_pure_content(
        source=source_wav,
        output_path=out_pure
    )
    print(f"Pure neutral audio saved to: {out_pure}")

if __name__ == "__main__":
    main()
