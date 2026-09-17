"""
Command-line interface for Boosted LWT.

Commands:
  boosted-lwt convert --source <wav> --target <target_wavs...> --output <out_wav> [--alpha 1.5]
  boosted-lwt pure-content --source <wav> --output <out_wav>
"""

import argparse
import sys
from pathlib import Path
import torch

from .pipeline import VoiceConverter


def parse_args():
    parser = argparse.ArgumentParser(
        prog="boosted-lwt",
        description="Boosted Local Wasserstein Transport: Zero-Shot Voice Conversion & Speech Disentanglement"
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to execute")

    # 1. convert
    p_conv = subparsers.add_parser("convert", help="Zero-shot voice conversion to target speaker")
    p_conv.add_argument("--source", "-s", type=str, required=True, help="Path to source audio file (WAV, FLAC, etc.)")
    p_conv.add_argument("--target", "-t", type=str, nargs="+", required=True, help="Path(s) to target speaker reference audio(s)")
    p_conv.add_argument("--output", "-o", type=str, required=True, help="Output path for converted WAV audio")
    p_conv.add_argument("--alpha", "-a", type=float, default=1.5, help="Covariance boost factor (default: 1.5, proposed optimum)")
    p_conv.add_argument("--beta", type=float, default=20.0, help="Phonetic cluster temperature (default: 20.0)")
    p_conv.add_argument("--method", type=str, default="boosted_lwt", choices=["boosted_lwt", "lwt", "soft_local_wct", "classic_wct"], help="Conversion method")
    p_conv.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    p_conv.add_argument("--checkpoint", type=str, default=None, help="Custom path to universal background checkpoint")

    # 2. pure-content
    p_pure = subparsers.add_parser("pure-content", help="Extract pure speaker-invariant content (canonical neutral voice)")
    p_pure.add_argument("--source", "-s", type=str, required=True, help="Path to source audio file")
    p_pure.add_argument("--output", "-o", type=str, required=True, help="Output path for neutral canonical WAV audio")
    p_pure.add_argument("--beta", type=float, default=20.0, help="Phonetic cluster temperature (default: 20.0)")
    p_pure.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    p_pure.add_argument("--checkpoint", type=str, default=None, help="Custom path to universal background checkpoint")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command is None:
        print("Error: No command specified. Use 'boosted-lwt convert' or 'boosted-lwt pure-content'. Run with -h for help.")
        sys.exit(1)

    print(f"=== Boosted LWT ({args.command}) ===")
    print(f"Device: {args.device}")

    converter = VoiceConverter(
        device=args.device,
        background_checkpoint=args.checkpoint,
    )

    if args.command == "convert":
        print(f"Source: {args.source}")
        print(f"Target(s): {len(args.target)} utterance(s)")
        print(f"Alpha: {args.alpha} | Method: {args.method}")
        print(f"Converting...")
        converter.convert(
            source=args.source,
            target=args.target,
            alpha=args.alpha,
            beta=args.beta,
            method=args.method,
            output_path=args.output,
        )
        print(f"[Success] Converted audio saved to: {Path(args.output).resolve()}")

    elif args.command == "pure-content":
        print(f"Source: {args.source}")
        print(f"Extracting analytical pure content on neutral background...")
        converter.extract_pure_content(
            source=args.source,
            beta=args.beta,
            output_path=args.output,
        )
        print(f"[Success] Neutral canonical audio saved to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
