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


def parse_ids(value: str) -> list:
    """ "3,7,12" -> [3, 7, 12] """
    try:
        ids = [int(p) for p in value.replace(" ", "").split(",") if p]
    except ValueError:
        ids = []
    if not ids or any(i < 1 for i in ids):
        raise argparse.ArgumentTypeError(f"invalid ID list '{value}' (use e.g. 3,7,12)")
    return ids


def parse_merges(value: str) -> list:
    """ "3+7,12+15+20" -> [[3, 7], [12, 15, 20]] """
    try:
        groups = [[int(i) for i in g.split("+")] for g in value.replace(" ", "").split(",") if g]
    except ValueError:
        groups = []
    if not groups or any(len(set(g)) < 2 or min(g) < 1 for g in groups):
        raise argparse.ArgumentTypeError(f"invalid merge list '{value}' (use e.g. 3+7,12+15+20)")
    return groups


def format_time(frame: int, fps: float) -> str:
    seconds = frame / fps if fps > 0 else 0.0
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


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
    parser.add_argument(
        "--list-ids", action="store_true",
        help="Print the table of IDs found in the video (when each one is on screen)"
    )
    parser.add_argument(
        "--exclude-ids", type=parse_ids, default=None, metavar="IDS",
        help="Leave these IDs unmasked, e.g. 3,7,12. IDs are the ones shown on the badges of a "
             "previous run with the same options (see --list-ids)"
    )
    parser.add_argument(
        "--merge-ids", type=parse_merges, default=None, metavar="GROUPS",
        help="IDs that are the same person, e.g. 3+7,12+15+20: each group is shown under its lowest ID"
    )
    args = parser.parse_args()
    if args.exclude_ids and args.no_tracking:
        parser.error("--exclude-ids needs tracking: it cannot be combined with --no-tracking")
    if args.merge_ids and args.no_tracking:
        parser.error("--merge-ids needs tracking: it cannot be combined with --no-tracking")
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
        use_tracking = not args.no_tracking
        analysis = masker.analyze_video(
            input_path=args.input,
            conf_threshold=args.conf,
            use_tracking=use_tracking,
            reidentify=args.reid,
            small_faces=args.small_faces,
            mask_range=mask_range,
            progress_callback=on_progress
        )
        merge_notes = []
        for group in args.merge_ids or []:
            kept, together = masker.merge_ids(analysis, group)
            merge_notes.append(f"IDs {'+'.join(map(str, sorted(set(group))))} merged into ID {kept}")
            if together:
                merge_notes.append(
                    f"[WARNING] IDs {'+'.join(map(str, sorted(set(group))))} are on screen together on "
                    f"{together} frame(s): they are probably different people"
                )
        known = {t.track_id for t in analysis.tracks}
        unknown = sorted(set(args.exclude_ids or ()) - known)
        total = analysis.total_frames
        masker.render_video(
            analysis,
            output_path,
            exclude_ids=args.exclude_ids,
            progress_callback=on_progress,
            progress_offset=total if use_tracking else 0,
            progress_total=total * 2 if use_tracking else total
        )
        if pbar:
            pbar.close()

        if use_tracking:
            excluded = set(args.exclude_ids or ()) & known
            if args.list_ids:
                print(f"\n{'ID':>5}  {'First seen':>10}  {'Last seen':>10}  {'Frames':>6}  Status")
                for t in analysis.tracks:
                    status = "visible" if t.track_id in excluded else "masked"
                    print(
                        f"{t.track_id:>5}  {format_time(t.first_frame, analysis.fps):>10}  "
                        f"{format_time(t.last_frame, analysis.fps):>10}  {t.n_frames:>6}  {status}"
                    )
            print(f"\nIDs found: {len(known)}, left unmasked: {len(excluded)}")
            for note in merge_notes:
                print(note)
            if unknown:
                print(f"[WARNING] Unknown IDs ignored: {', '.join(map(str, unknown))}", file=sys.stderr)
        elapsed = time.time() - start_time
        print(f"\n[SUCCESS] Anonymized video created with preserved audio: {output_path} ({elapsed:.1f}s)")
    except Exception as e:
        if pbar:
            pbar.close()
        print(f"\n[ERROR] Processing failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
