"""Turn an external ASL-alphabet photo dataset into landmark training rows.

Real, full-resolution photos of ASL letters are run through the same MediaPipe
hand tracker the live app uses, and each detected hand is stored as its
landmark feature vector (see landmark_features.py).

Default source: Hugging Face `Marxulia/asl_sign_languages_alphabets_v03`
(~10.9k photos, 26 classes, labels 0-25 = A-Z; its card states no license),
downloaded to data/external/asl_v03.parquet:

    curl.exe -L -o data/external/asl_v03.parquet ^
      https://huggingface.co/datasets/Marxulia/asl_sign_languages_alphabets_v03/resolve/main/data/train-00000-of-00001.parquet

Usage:
    python import_external_data.py              # every photo
    python import_external_data.py --stride 2   # every 2nd photo (faster)

Output: data/external_landmarks.csv (label + 63 feature columns), read
automatically by train_landmark_classifier.py.

Notes:
  * J and Z are skipped: they need motion, so a single static frame can't
    represent them.
  * Photos where MediaPipe finds no hand are skipped.
  * Rows keep the source's order (grouped by letter, consecutive frames), which
    the trainer relies on to build a leak-free held-out split.
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from hand_tracker import HandTracker
from landmark_features import NUM_FEATURES, landmark_features

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "data" / "external" / "asl_v03.parquet"
OUT_CSV = ROOT / "data" / "external_landmarks.csv"
SKIP_LABELS = {9, 25}  # J, Z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth photo per letter")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Dataset not found at {args.source}. See this file's docstring for the download command.")

    df = pd.read_parquet(args.source)
    df = df[~df["label"].isin(SKIP_LABELS)]
    df = df[df.groupby("label").cumcount() % args.stride == 0]
    print(f"{len(df)} photos to process from {args.source.name}")

    tracker = HandTracker(max_hands=1)
    kept = missed = 0
    t0 = time.time()
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + [f"f{i}" for i in range(NUM_FEATURES)])
        for n, (label, image) in enumerate(zip(df["label"], df["image"]), 1):
            frame = cv2.imdecode(np.frombuffer(image["bytes"], np.uint8), cv2.IMREAD_COLOR)
            box = tracker.process(frame) if frame is not None else None
            if box is None:
                missed += 1
            else:
                writer.writerow([int(label)] + landmark_features(box).tolist())
                kept += 1
            if n % 1000 == 0:
                print(f"  {n}/{len(df)}  kept {kept}  no-hand {missed}  ({time.time() - t0:.0f}s)", flush=True)
    tracker.close()

    print(f"Done: {kept} photos with a detected hand ({missed} skipped) -> {OUT_CSV}")


if __name__ == "__main__":
    main()
