"""
extract_ref_faces.py

Create a small reference gallery (8-12 images) for the target subject.
These references are used by build_identity_mask.py to compute a subject centroid.

Typical usage (recommended: with CSV):
python extract_ref_faces.py ^
  --video "Footage/September 2025.mp4" ^
  --csv "Emotion_Data_All_Videos/001_September 2025.csv" ^
  --out_dir "refs_subject" ^
  --n 12 --minutes 4

Fallback usage (without CSV):
python extract_ref_faces.py ^
  --video "Footage/September 2025.mp4" ^
  --out_dir "refs_subject" ^
  --n 12 --minutes 4 --every_sec 6
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import List, Optional, Tuple

import cv2
import insightface
import numpy as np
import pandas as pd


def load_imotions_csv(path: str, encoding: str = "latin1", forced_header_row: Optional[int] = None) -> pd.DataFrame:
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
        if header_idx is None:
            header_idx = 24  # compatible with first data row on line 26 (1-based)

    df = pd.read_csv(path, encoding=encoding, header=header_idx, low_memory=False)
    if "Timestamp" not in df.columns:
        raise RuntimeError(f"Timestamp column not found in: {path}")
    return df


def infer_face_present(df: pd.DataFrame) -> pd.Series:
    lm_cols = [c for c in df.columns if str(c).startswith("feature")]
    if lm_cols:
        return (df[lm_cols].abs().sum(axis=1) > 0)
    if "Valence" in df.columns:
        return df["Valence"].notna()
    return pd.Series([True] * len(df), index=df.index)


def prepare_face_app(det_size: Tuple[int, int] = (640, 640)):
    providers_to_try = [
        ["DmlExecutionProvider", "CPUExecutionProvider"],
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]
    last_error = None
    for providers in providers_to_try:
        try:
            app = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
            try:
                app.prepare(ctx_id=0, det_size=det_size)
            except Exception:
                app.prepare(ctx_id=-1, det_size=det_size)
            print(f"Using providers: {providers}")
            return app
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Failed to initialize InsightFace: {last_error}")


def largest_face_crop_and_emb(app, bgr_image: np.ndarray, min_face_px: int = 80):
    faces = app.get(bgr_image)
    if not faces:
        return None, None, None

    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    x1, y1, x2, y2 = map(int, face.bbox)
    w, h = x2 - x1, y2 - y1
    if min(w, h) < min_face_px:
        return None, None, None

    pad = int(0.15 * max(w, h))
    x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
    x2p, y2p = min(bgr_image.shape[1], x2 + pad), min(bgr_image.shape[0], y2 + pad)
    crop = bgr_image[y1p:y2p, x1p:x2p].copy()

    emb = np.asarray(face.embedding, dtype=np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return crop, emb, (x1p, y1p, x2p, y2p)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def build_candidate_timestamps_ms(
    video_path: str,
    csv_path: Optional[str],
    minutes: int,
    n: int,
    every_sec: Optional[int],
    forced_header_row: Optional[int],
) -> List[int]:
    if csv_path:
        df = load_imotions_csv(csv_path, forced_header_row=forced_header_row)
        df["face_present"] = infer_face_present(df)
        df["minute"] = (df["Timestamp"] // 60_000).astype(int)
        sub = df[(df["minute"] < minutes) & (df["face_present"])].copy()
        if len(sub) == 0:
            raise RuntimeError(
                "No face-present rows found in chosen time window. "
                "Increase --minutes or use --every_sec without --csv."
            )
        # Oversample to allow pruning and missed detections
        target = max(n * 5, 40)
        idxs = np.linspace(0, len(sub) - 1, min(target, len(sub)), dtype=int)
        return [int(v) for v in sub.iloc[idxs]["Timestamp"].tolist()]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / fps if total_frames > 0 else minutes * 60
    cap.release()

    limit_sec = min(duration_sec, minutes * 60)
    interval = every_sec if every_sec is not None else max(2, int(round(limit_sec / (n * 5))))
    times_sec = np.arange(1, max(2, int(limit_sec)), max(1, interval))
    return [int(t * 1000) for t in times_sec.tolist()]


def main(
    video: str,
    out_dir: str,
    n: int,
    minutes: int,
    csv_path: Optional[str],
    every_sec: Optional[int],
    min_face_px: int,
    prune_sim: float,
    forced_header_row: Optional[int],
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    candidate_times_ms = build_candidate_timestamps_ms(
        video_path=video,
        csv_path=csv_path,
        minutes=minutes,
        n=n,
        every_sec=every_sec,
        forced_header_row=forced_header_row,
    )
    print(f"Candidate timestamps: {len(candidate_times_ms)}")

    app = prepare_face_app()
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    kept: List[Tuple[int, np.ndarray, np.ndarray]] = []
    kept_embs: List[np.ndarray] = []

    for t_ms in candidate_times_ms:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t_ms))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        crop, emb, _ = largest_face_crop_and_emb(app, frame, min_face_px=min_face_px)
        if crop is None or emb is None:
            continue

        if kept_embs:
            max_sim = max(cosine_sim(emb, e) for e in kept_embs)
            if max_sim >= prune_sim:
                continue

        kept.append((t_ms, crop, emb))
        kept_embs.append(emb)
        if len(kept) >= n:
            break

    cap.release()

    if not kept:
        raise RuntimeError(
            "No suitable faces extracted. Try increasing --minutes, lowering --min_face_px, "
            "or increasing diversity threshold (--prune_sim)."
        )

    for i, (t_ms, crop, _) in enumerate(kept):
        out_file = os.path.join(out_dir, f"ref_{i:02d}_{t_ms}ms.jpg")
        cv2.imwrite(out_file, crop)

    manifest = os.path.join(out_dir, "manifest.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "timestamp_ms", "filename"])
        for i, (t_ms, _, _) in enumerate(kept):
            w.writerow([i, t_ms, f"ref_{i:02d}_{t_ms}ms.jpg"])

    print(f"Saved {len(kept)} reference image(s) to: {out_dir}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract subject reference faces from a video.")
    ap.add_argument("--video", required=True, help="Path to source video")
    ap.add_argument("--out_dir", required=True, help="Directory to write references")
    ap.add_argument("--n", type=int, default=12, help="Number of references to keep")
    ap.add_argument("--minutes", type=int, default=4, help="Search window in first N minutes")
    ap.add_argument("--csv", dest="csv_path", default=None, help="Optional iMotions CSV path")
    ap.add_argument("--every_sec", type=int, default=None, help="Sampling interval when CSV is not provided")
    ap.add_argument("--min_face_px", type=int, default=80, help="Minimum face size in pixels")
    ap.add_argument("--prune_sim", type=float, default=0.92, help="Prune near-duplicates above this cosine sim")
    ap.add_argument(
        "--forced_header_row",
        type=int,
        default=None,
        help="Optional forced CSV header row index (0-based). For row-26-first-data use 24.",
    )
    args = ap.parse_args()

    main(
        video=args.video,
        out_dir=args.out_dir,
        n=args.n,
        minutes=args.minutes,
        csv_path=args.csv_path,
        every_sec=args.every_sec,
        min_face_px=args.min_face_px,
        prune_sim=args.prune_sim,
        forced_header_row=args.forced_header_row,
    )
