"""
build_identity_mask.py

Frame-level subject identity filtering for iMotions emotion exports + video.

What this script does
---------------------
1) Loads an iMotions CSV (auto-detects header row; compatible with first data row on row 26).
2) Aligns each CSV row to a video timestamp/frame.
3) Detects face embedding with InsightFace and compares it to a reference centroid.
4) Writes row-aligned output with:
   - is_subject (bool)
   - subject_similarity (float)
   plus original metadata/emotion columns.
5) Writes minute-level QA summary (pct_subject, etc.).

Install
-------
pip install insightface onnxruntime opencv-python pandas numpy

Example
-------
python build_identity_mask.py ^
  --csv "Emotion_Data_All_Videos/001_September 2025.csv" ^
  --video "Footage/September 2025.mp4" ^
  --ref_dir "refs_subject" ^
  --out_csv "out_frames/Sep_2025_frames.csv" ^
  --minute_csv "out_minutes/Sep_2025_minutes.csv"
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import insightface


# -----------------------------------------------------------------------------
# CSV helpers
# -----------------------------------------------------------------------------

def load_imotions_csv(path: str, encoding: str = "latin1", forced_header_row: Optional[int] = None) -> pd.DataFrame:
    """
    Load iMotions CSV by detecting header row (line containing Row + Timestamp).

    Notes:
    - Your files commonly have first data row at line 26 (1-based), i.e. header row line 25.
    - If auto detection fails, this function falls back to header row index 24 (0-based).
    """
    if forced_header_row is not None:
        header_idx = forced_header_row
    else:
        header_idx = None
        with open(path, "rb") as f:
            for i in range(500):
                b = f.readline()
                if not b:
                    break
                line = b.decode(encoding, errors="ignore")
                if (not line.startswith("#")) and ("Timestamp" in line) and ("Row" in line):
                    header_idx = i
                    break

        # Fallback for known format: header line 25 (1-based) => index 24
        if header_idx is None:
            header_idx = 24

    df = pd.read_csv(path, encoding=encoding, header=header_idx, low_memory=False)
    required = ["Row", "Timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required column(s) {missing} in {path}")
    return df


def add_face_present_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Infer whether a face exists on a row using landmarks when available."""
    lm_cols = [c for c in df.columns if str(c).startswith("feature")]
    if lm_cols:
        df["face_present"] = (df[lm_cols].abs().sum(axis=1) > 0)
    else:
        # fallback heuristic
        if "Valence" in df.columns:
            df["face_present"] = df["Valence"].notna()
        else:
            df["face_present"] = True
    return df


# -----------------------------------------------------------------------------
# Face embedding helpers
# -----------------------------------------------------------------------------

def prepare_face_app(det_size: Tuple[int, int] = (320, 320)):
    """Prepare InsightFace with practical provider fallback for Windows."""
    providers_to_try = [
        ["DmlExecutionProvider", "CPUExecutionProvider"],
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]

    last_error = None
    for providers in providers_to_try:
        try:
            app = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
            # ctx_id 0 may use GPU provider if available; fallback to CPU below.
            try:
                app.prepare(ctx_id=0, det_size=det_size)
            except Exception:
                app.prepare(ctx_id=-1, det_size=det_size)
            print(f"Using providers: {providers}")
            return app
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Failed to initialize InsightFace providers: {last_error}")


def largest_face_embedding(app, bgr_image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """Return normalized embedding and bbox for largest detected face."""
    faces = app.get(bgr_image)
    if not faces:
        return None, None
    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    emb = np.asarray(face.embedding, dtype=np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    bbox = tuple(map(int, face.bbox))
    return emb, bbox


def build_reference_centroid(app, ref_dir: str) -> np.ndarray:
    """Build one centroid embedding from all valid reference face images."""
    if not os.path.isdir(ref_dir):
        raise RuntimeError(f"Reference directory does not exist: {ref_dir}")

    embs: List[np.ndarray] = []
    for fn in sorted(os.listdir(ref_dir)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue
        path = os.path.join(ref_dir, fn)
        img = cv2.imread(path)
        if img is None:
            continue
        emb, _ = largest_face_embedding(app, img)
        if emb is not None:
            embs.append(emb)

    if not embs:
        raise RuntimeError(f"No valid reference embeddings found in: {ref_dir}")

    centroid = np.stack(embs, axis=0).mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
    print(f"Built centroid from {len(embs)} reference image(s).")
    return centroid.astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# -----------------------------------------------------------------------------
# Identity filtering
# -----------------------------------------------------------------------------

@dataclass
class FilterConfig:
    sim_thresh_on: float = 0.42
    sim_thresh_off: float = 0.38
    min_run_seconds: float = 0.5
    max_minutes: Optional[int] = None
    seek_mode: str = "timestamp"  # timestamp | frame
    sample_step: int = 30          # process every Nth eligible row then forward-fill
    conf_col: Optional[str] = None
    conf_thresh: float = 0.0
    forced_header_row: Optional[int] = None


def identity_filter(
    csv_path: str,
    video_path: str,
    ref_dir: str,
    out_csv: str,
    minute_csv: str,
    cfg: FilterConfig,
) -> None:
    df = load_imotions_csv(csv_path, forced_header_row=cfg.forced_header_row)
    df = add_face_present_flag(df)

    if cfg.conf_col and cfg.conf_col in df.columns:
        df["high_conf"] = df[cfg.conf_col] >= cfg.conf_thresh
    else:
        df["high_conf"] = True

    if cfg.max_minutes is not None:
        max_ts = cfg.max_minutes * 60_000
        in_time_window = df["Timestamp"] < max_ts
    else:
        in_time_window = pd.Series([True] * len(df), index=df.index)

    consider = (df["face_present"] & df["high_conf"] & in_time_window).to_numpy()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    app = prepare_face_app()
    ref_centroid = build_reference_centroid(app, ref_dir)

    # outputs (row-aligned to full CSV)
    is_subject = np.zeros(len(df), dtype=bool)
    subject_similarity = np.full(len(df), np.nan, dtype=np.float32)

    rows_to_process = np.where(consider)[0]
    if len(rows_to_process) == 0:
        cap.release()
        print("No eligible rows found. Writing empty labels.")
    else:
        sample_step = max(1, int(cfg.sample_step))
        # Because we may process every Nth row, continuity must be computed in sampled units.
        min_run_samples = max(1, int(round((cfg.min_run_seconds * fps) / sample_step)))
        sampled_indices = rows_to_process[::sample_step]

        print(f"Eligible rows: {len(rows_to_process):,}")
        print(f"Sample step: {sample_step} -> analysing {len(sampled_indices):,} row(s)")

        active = False
        run_count = 0

        for idx_in_sampled, i in enumerate(sampled_indices):
            row = df.iloc[i]

            if cfg.seek_mode == "frame" and "SampleNumber" in df.columns:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["SampleNumber"]))
            else:
                cap.set(cv2.CAP_PROP_POS_MSEC, float(row["Timestamp"]))

            ok, frame = cap.read()
            sim = np.nan
            decision = False

            if ok and frame is not None:
                emb, _ = largest_face_embedding(app, frame)
                if emb is not None:
                    sim = cosine_sim(emb, ref_centroid)

                    # Hysteresis state transition to reduce flicker on cutaways
                    if active:
                        if sim >= cfg.sim_thresh_off:
                            decision = True
                        else:
                            active = False
                            run_count = 0
                            decision = False
                    else:
                        if sim >= cfg.sim_thresh_on:
                            run_count += 1
                            if run_count >= min_run_samples:
                                active = True
                                decision = True
                            else:
                                decision = False
                        else:
                            run_count = 0
                            decision = False
                else:
                    # no face in frame -> reset state
                    active = False
                    run_count = 0
                    decision = False
            else:
                active = False
                run_count = 0
                decision = False

            # forward-fill to keep row-level alignment for skipped rows
            next_sample_pos = idx_in_sampled + 1
            if next_sample_pos < len(sampled_indices):
                fill_end = sampled_indices[next_sample_pos]
            else:
                fill_end = min(i + sample_step, len(df))

            for j in range(i, fill_end):
                if consider[j]:
                    is_subject[j] = decision
                    subject_similarity[j] = sim

            if (idx_in_sampled + 1) % 200 == 0:
                pct = 100.0 * (idx_in_sampled + 1) / len(sampled_indices)
                print(f"Progress: {idx_in_sampled + 1:,}/{len(sampled_indices):,} ({pct:.1f}%)")

        cap.release()

    df["is_subject"] = is_subject
    df["subject_similarity"] = subject_similarity
    df["minute"] = (df["Timestamp"] // 60_000).astype(int)

    # Frame-level export keeps all original rows but selects practical columns first
    preferred = [
        "Row", "Timestamp", "SampleNumber", "minute",
        "Valence", "Joy", "Sadness", "Anger", "Fear", "Disgust", "Contempt", "Surprise",
        "Engagement", "Attention",
        "face_present", "high_conf", "subject_similarity", "is_subject",
    ]
    ordered_cols = [c for c in preferred if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    export_cols = ordered_cols + remaining_cols

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(minute_csv) or ".", exist_ok=True)
    df[export_cols].to_csv(out_csv, index=False)

    summary = (
        df.groupby("minute")
        .agg(
            rows=("Row", "count"),
            face_rows=("face_present", "sum"),
            subject_rows=("is_subject", "sum"),
            mean_similarity=("subject_similarity", "mean"),
        )
        .assign(
            pct_face=lambda d: d["face_rows"] / d["rows"],
            pct_subject=lambda d: d["subject_rows"] / d["rows"],
        )
        .reset_index()
    )
    summary.to_csv(minute_csv, index=False)

    print(f"Saved frame-level output: {out_csv}")
    print(f"Saved minute summary:     {minute_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Row-aligned identity mask for iMotions + video.")
    p.add_argument("--csv", required=True, help="iMotions CSV path")
    p.add_argument("--video", required=True, help="Video path")
    p.add_argument("--ref_dir", required=True, help="Directory with reference face images")
    p.add_argument("--out_csv", required=True, help="Frame-level output CSV path")
    p.add_argument("--minute_csv", required=True, help="Minute summary output CSV path")

    p.add_argument("--sim_thresh_on", type=float, default=0.42, help="Enter threshold for subject state")
    p.add_argument("--sim_thresh_off", type=float, default=0.38, help="Exit threshold for subject state")
    p.add_argument("--min_run_seconds", type=float, default=0.5, help="Min continuous run before enabling subject state")
    p.add_argument("--max_minutes", type=int, default=None, help="Optional limit to first N minutes")
    p.add_argument("--seek_mode", choices=["timestamp", "frame"], default="timestamp")
    p.add_argument("--sample_step", type=int, default=30, help="Analyse every Nth eligible row")
    p.add_argument("--conf_col", default=None, help="Optional confidence column name")
    p.add_argument("--conf_thresh", type=float, default=0.0, help="Confidence threshold")
    p.add_argument("--forced_header_row", type=int, default=None, help="Force CSV header row index (0-based)")
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
    identity_filter(
        csv_path=args.csv,
        video_path=args.video,
        ref_dir=args.ref_dir,
        out_csv=args.out_csv,
        minute_csv=args.minute_csv,
        cfg=cfg,
    )
