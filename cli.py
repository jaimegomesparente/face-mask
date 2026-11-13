"""
Command Line Interface (CLI) for Face Masker.
Quickly anonymize videos in background or batch pipelines.
"""

import argparse
import os
import sys
import time
from tqdm import tqdm
from engine import FaceMasker


def parse_time(value: str) -> float:
    """Seconds ("83.5") or timecode ("1:23.5", "0:01:23") -> seconds."""
    try:
        parts = [float(p) for p in value.split(":")]
    except ValueError:
        parts = []
    if not 1 <= len(parts) <= 3 or any(p < 0 for p in parts):
        raise argparse.ArgumentTypeError(f"invalid time '{value}' (use seconds, MM:SS or HH:MM:SS)")
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Face Masker - Face anonymization for video."
    )
    parser.add_argument("input", help="Path to input video file (MP4, MKV, MOV, AVI...)")
    parser.add_argument("-o", "--output", default=None, help="Path to output video file (default: <name>_masked.mp4)")
    parser.add_argument("--conf", type=float, default=FaceMasker.DEFAULT_CONF_THRESHOLD, help="Detection confidence threshold")
    parser.add_argument("--no-tracking", action="store_true", help="Disable temporal box smoothing")
    parser.add_argument(
        "--reid", action="store_true",
        help="Keep the same ID when a person leaves the frame and comes back (slower, in-memory face comparison)"
    )
    parser.add_argument(
        "--small-faces", action="store_true",
        help="Also detect on two halves of each frame: catches tiny faces (crowds, wide shots), about 3x slower"
    )
    parser.add_argument(
        "--start", type=parse_time, default=None, metavar="TIME",
        help="Only mask faces from this time on (seconds, MM:SS or HH:MM:SS; default: beginning)"
    )
    parser.add_argument(
        "--end", type=parse_time, default=None, metavar="TIME",
        help="Only mask faces up to this time (seconds, MM:SS or HH:MM:SS; default: end of video)"
    )
    args = parser.parse_args()
    if args.start is not None and args.end is not None and args.end <= args.start:
        parser.error("--end must be after --start")
    if args.reid and args.no_tracking:
        parser.error("--reid needs tracking: it cannot be combined with --no-tracking")
    return args


def main():
    args = parse_args()

    if args.output is None:
        # splitext only looks at the file name (rfind(".") also matched dots in folder names)
        output_path = f"{os.path.splitext(args.input)[0]}_masked.mp4"
    else:
        output_path = args.output

    print("=" * 60)
    print("FACE MASKER")
    print(f"Input file    : {args.input}")
    print(f"Output file   : {output_path}")
    print(f"Stabilization : {'Disabled' if args.no_tracking else 'Enabled'}")
    print(f"Persistent IDs: {'Enabled' if args.reid else 'Disabled'}")
    print(f"Small faces   : {'Enabled' if args.small_faces else 'Disabled'}")
    mask_range = None
    if args.start is not None or args.end is not None:
        mask_range = (args.start, args.end)
        start_txt = f"{args.start:g}s" if args.start is not None else "beginning"
        end_txt = f"{args.end:g}s" if args.end is not None else "end"
        print(f"Masked portion: {start_txt} -> {end_txt} (faces stay visible elsewhere)")
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
            reidentify=args.reid,
            small_faces=args.small_faces,
            mask_range=mask_range,
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
