"""
Command Line Interface (CLI) for Face Masker.
Quickly anonymize videos in background or batch pipelines.
"""

import argparse
import sys
import time
from tqdm import tqdm
from engine import FaceMasker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Face Masker - Face anonymization for video."
    )
    parser.add_argument("input", help="Path to input video file (MP4, MKV, MOV, AVI...)")
    parser.add_argument("-o", "--output", default=None, help="Path to output video file (default: <name>_masked.mp4)")
    parser.add_argument("--conf", type=float, default=FaceMasker.DEFAULT_CONF_THRESHOLD, help="Detection confidence threshold")
    parser.add_argument("--no-tracking", action="store_true", help="Disable temporal box smoothing")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output is None:
        dot_idx = args.input.rfind(".")
        if dot_idx != -1:
            output_path = f"{args.input[:dot_idx]}_masked.mp4"
        else:
            output_path = f"{args.input}_masked.mp4"
    else:
        output_path = args.output

    print("=" * 60)
    print("FACE MASKER")
    print(f"Input file    : {args.input}")
    print(f"Output file   : {output_path}")
    print(f"Stabilization : {'Disabled' if args.no_tracking else 'Enabled'}")
    print("=" * 60)

    masker = FaceMasker()
    pbar = None
    start_time = time.time()

    def on_progress(current: int, total: int):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=total, desc="Processing video", unit="frame")
        pbar.update(current - pbar.n)

    try:
        masker.process_video(
            input_path=args.input,
            output_path=output_path,
            conf_threshold=args.conf,
            use_tracking=not args.no_tracking,
            progress_callback=on_progress
        )
        if pbar:
            pbar.close()
        elapsed = time.time() - start_time
        print(f"\n[SUCCESS] Anonymized video created with preserved audio: {output_path} ({elapsed:.1f}s)")
    except Exception as e:
        if pbar:
            pbar.close()
        print(f"\n[ERROR] Processing failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
