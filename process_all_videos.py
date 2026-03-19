"""
process_all_videos.py

Batch process all video + CSV pairs for row-aligned subject labeling.

Default folders are tailored to this repository:
- Videos: Footage/
- CSVs:   Emotion_Data_All_Videos/

Pairing convention handled:
- Video: "September 2025.mp4"
- CSV:   "001_September 2025.csv"  (numeric prefix optional)

Example
-------
python process_all_videos.py --ref_dir refs_subject --max_minutes 5
python process_all_videos.py --ref_dir refs_subject
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import asdict
from typing import Dict, List, Tuple

from build_identity_mask import FilterConfig, identity_filter
from extract_ref_faces import main as extract_refs_main


def normalize_name(name: str) -> str:
    """Normalize file stem for robust matching."""
    n = name.strip()
    # Remove leading numeric prefix like "001_"
    n = re.sub(r"^\d+_", "", n)
    # Collapse whitespace and lowercase
    n = re.sub(r"\s+", " ", n).strip().lower()
    return n


def collect_files(video_dir: str, csv_dir: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    video_map: Dict[str, str] = {}
    csv_map: Dict[str, str] = {}

    for fn in sorted(os.listdir(video_dir)):
        if fn.lower().endswith(".mp4"):
            stem = os.path.splitext(fn)[0]
            video_map[normalize_name(stem)] = os.path.join(video_dir, fn)

    for fn in sorted(os.listdir(csv_dir)):
        if fn.lower().endswith(".csv"):
            stem = os.path.splitext(fn)[0]
            csv_map[normalize_name(stem)] = os.path.join(csv_dir, fn)

    return video_map, csv_map


def build_pairs(video_dir: str, csv_dir: str) -> List[Tuple[str, str, str]]:
    """
    Return list of (key, video_path, csv_path).
    key is normalized shared stem.
    """
    video_map, csv_map = collect_files(video_dir, csv_dir)
    keys = sorted(set(video_map.keys()) & set(csv_map.keys()))
    return [(k, video_map[k], csv_map[k]) for k in keys]


def pretty_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m}m"


def safe_stem_from_key(key: str) -> str:
    s = key.replace(",", "")
    s = re.sub(r"\s+", "_", s)
    return s


def process_all(
    video_dir: str,
    csv_dir: str,
    ref_dir: str,
    out_frames_dir: str,
    out_minutes_dir: str,
    cfg: FilterConfig,
    bootstrap_refs_when_missing: bool = False,
    bootstrap_video: str | None = None,
    bootstrap_csv: str | None = None,
    bootstrap_n: int = 12,
    bootstrap_minutes: int = 4,
    bootstrap_forced_header_row: int | None = None,
) -> None:
    if not os.path.isdir(video_dir):
        raise RuntimeError(f"Video directory not found: {video_dir}")
    if not os.path.isdir(csv_dir):
        raise RuntimeError(f"CSV directory not found: {csv_dir}")

    os.makedirs(out_frames_dir, exist_ok=True)
    os.makedirs(out_minutes_dir, exist_ok=True)

    pairs = build_pairs(video_dir, csv_dir)
    if not pairs:
        raise RuntimeError(
            "No matching video/CSV pairs found. "
            "Expected names like 'September 2025.mp4' and '001_September 2025.csv'."
        )

    if not os.path.isdir(ref_dir):
        if not bootstrap_refs_when_missing:
            raise RuntimeError(
                f"Reference directory not found: {ref_dir}\n"
                f"Create it first with:\n"
                f"python extract_ref_faces.py --video \"Footage/September 2025.mp4\" "
                f"--csv \"Emotion_Data_All_Videos/001_September 2025.csv\" "
                f"--out_dir {ref_dir} --n 12 --minutes 4 --forced_header_row 24"
            )

        # Auto-bootstrap reference set when missing
        if bootstrap_video is None or bootstrap_csv is None:
            # default to first discovered pair
            _, default_video, default_csv = pairs[0]
            bootstrap_video = bootstrap_video or default_video
            bootstrap_csv = bootstrap_csv or default_csv

        print("Reference directory not found. Auto-bootstrapping references...")
        print(f"  Video: {bootstrap_video}")
        print(f"  CSV:   {bootstrap_csv}")
        print(f"  Out:   {ref_dir}")

        extract_refs_main(
            video=bootstrap_video,
            out_dir=ref_dir,
            n=bootstrap_n,
            minutes=bootstrap_minutes,
            csv_path=bootstrap_csv,
            every_sec=None,
            min_face_px=80,
            prune_sim=0.92,
            forced_header_row=bootstrap_forced_header_row if bootstrap_forced_header_row is not None else cfg.forced_header_row,
        )

        if not os.path.isdir(ref_dir):
            raise RuntimeError(f"Failed to create reference directory: {ref_dir}")

    print("=" * 80)
    print("BATCH SUBJECT FILTERING")
    print("=" * 80)
    print(f"Pairs found: {len(pairs)}")
    print(f"Config: {asdict(cfg)}")

    t0 = time.time()
    results = []

    for idx, (key, video_path, csv_path) in enumerate(pairs, start=1):
        name = safe_stem_from_key(key)
        out_csv = os.path.join(out_frames_dir, f"{name}_frames.csv")
        minute_csv = os.path.join(out_minutes_dir, f"{name}_minutes.csv")

        print("\n" + "-" * 80)
        print(f"[{idx}/{len(pairs)}] {key}")
        print(f"Video: {video_path}")
        print(f"CSV:   {csv_path}")

        v0 = time.time()
        ok = True
        err = None
        try:
            identity_filter(
                csv_path=csv_path,
                video_path=video_path,
                ref_dir=ref_dir,
                out_csv=out_csv,
                minute_csv=minute_csv,
                cfg=cfg,
            )
        except Exception as e:
            ok = False
            err = str(e)
            print(f"ERROR: {err}")

        v_elapsed = time.time() - v0
        results.append((key, ok, v_elapsed, err))
        print(f"Done in {pretty_duration(v_elapsed)}")

    total = time.time() - t0
    success = sum(1 for _, ok, _, _ in results if ok)
    fail = len(results) - success

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total duration: {pretty_duration(total)}")
    print(f"Successful: {success}/{len(results)}")
    if fail:
        print(f"Failed: {fail}")
    for key, ok, elapsed, err in results:
        status = "OK" if ok else "FAIL"
        print(f"{status:4s} | {pretty_duration(elapsed):>8s} | {key}")
        if err:
            print(f"      {err}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch subject filtering for all video/CSV pairs.")
    p.add_argument("--video_dir", default="Footage", help="Directory containing .mp4 files")
    p.add_argument("--csv_dir", default="Emotion_Data_All_Videos", help="Directory containing iMotions CSV files")
    p.add_argument("--ref_dir", required=True, help="Directory containing reference face images")
    p.add_argument("--out_frames_dir", default="out_frames", help="Output directory for frame-level CSV files")
    p.add_argument("--out_minutes_dir", default="out_minutes", help="Output directory for minute-level CSV summaries")

    p.add_argument("--sim_thresh_on", type=float, default=0.42)
    p.add_argument("--sim_thresh_off", type=float, default=0.38)
    p.add_argument("--min_run_seconds", type=float, default=0.5)
    p.add_argument("--max_minutes", type=int, default=None)
    p.add_argument("--seek_mode", choices=["timestamp", "frame"], default="timestamp")
    p.add_argument("--sample_step", type=int, default=30)
    p.add_argument("--conf_col", default=None)
    p.add_argument("--conf_thresh", type=float, default=0.0)
    p.add_argument(
        "--forced_header_row",
        type=int,
        default=None,
        help="Optional forced CSV header row index (0-based). For row-26-first-data use 24.",
    )

    p.add_argument(
        "--bootstrap_refs_when_missing",
        action="store_true",
        help="If ref_dir is missing, automatically create refs using one video/CSV pair.",
    )
    p.add_argument(
        "--bootstrap_video",
        default=None,
        help="Optional video path to use when auto-creating refs.",
    )
    p.add_argument(
        "--bootstrap_csv",
        default=None,
        help="Optional CSV path to use when auto-creating refs.",
    )
    p.add_argument(
        "--bootstrap_n",
        type=int,
        default=12,
        help="Number of reference images to create during auto-bootstrap.",
    )
    p.add_argument(
        "--bootstrap_minutes",
        type=int,
        default=4,
        help="Minutes window for auto-bootstrap reference extraction.",
    )
    p.add_argument(
        "--bootstrap_forced_header_row",
        type=int,
        default=None,
        help="Optional forced header row for bootstrap extraction.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = FilterConfig(
        sim_thresh_on=args.sim_thresh_on,
        sim_thresh_off=args.sim_thresh_off,
        min_run_seconds=args.min_run_seconds,
        max_minutes=args.max_minutes,
        seek_mode=args.seek_mode,
        sample_step=args.sample_step,
        conf_col=args.conf_col,
        conf_thresh=args.conf_thresh,
        forced_header_row=args.forced_header_row,
    )
    process_all(
        video_dir=args.video_dir,
        csv_dir=args.csv_dir,
        ref_dir=args.ref_dir,
        out_frames_dir=args.out_frames_dir,
        out_minutes_dir=args.out_minutes_dir,
        cfg=cfg,
        bootstrap_refs_when_missing=args.bootstrap_refs_when_missing,
        bootstrap_video=args.bootstrap_video,
        bootstrap_csv=args.bootstrap_csv,
        bootstrap_n=args.bootstrap_n,
        bootstrap_minutes=args.bootstrap_minutes,
        bootstrap_forced_header_row=args.bootstrap_forced_header_row,
    )
