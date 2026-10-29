"""Collect labeled hand-sign samples from your own webcam.

Why this script exists: the model is trained on photos of a few signers. Your
hand, camera and lighting are different, so recording your own signs (through
the same MediaPipe hand tracker and landmark features the live app uses,
see realtime_recognition.py) is the most reliable way to make it accurate for
YOU. train_landmark_classifier.py weights your samples x3.

Usage:
    python collect_data.py --all                  # guided session, every letter
    python collect_data.py --letter A             # just one letter
    python collect_data.py --all --num-samples 100

In --all mode you get a short countdown before each letter; get your hand into
position, then hold the sign steadily (slightly moving/rotating it a little
gives the model more variety). Press 'q' at any time to stop.

Samples are appended to data/custom_landmarks.csv (label + 63 landmark
features). train_landmark_classifier.py picks the file up automatically, so
after collecting just retrain. Delete the file to start over.
"""

import argparse
import csv
import string
import time
from pathlib import Path

import cv2

from hand_tracker import HandTracker
from landmark_features import NUM_FEATURES, landmark_features

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "data" / "custom_landmarks.csv"
WINDOW = "Collecting samples"
# J and Z require motion, so they can't be represented by a static classifier.
LETTERS = [c for c in string.ascii_uppercase if c not in ("J", "Z")]


def letter_to_label(letter: str) -> int:
    letter = letter.upper()
    idx = ord(letter) - ord("A")
    if letter in ("J", "Z"):
        raise ValueError("J and Z require motion and aren't supported by this static classifier.")
    return idx


def _show(frame, box, lines):
    display = frame.copy()
    if box is not None:
        cv2.rectangle(display, (box.x_min, box.y_min), (box.x_max, box.y_max), (0, 200, 0), 2)
    for i, line in enumerate(lines):
        cv2.putText(display, line, (20, 40 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imshow(WINDOW, display)
    return cv2.waitKey(1) & 0xFF == ord("q")


def countdown(cap, tracker, letter: str, seconds: int) -> bool:
    """Show a get-ready countdown. Returns True if the user pressed 'q'."""
    end = time.time() + seconds
    while time.time() < end:
        ok, frame = cap.read()
        if not ok:
            return True
        frame = cv2.flip(frame, 1)
        remaining = int(end - time.time()) + 1
        if _show(frame, tracker.process(frame), [f"Get ready: sign '{letter}'", f"Starting in {remaining}..."]):
            return True
    return False


def collect_letter(cap, tracker, writer, letter: str, num_samples: int) -> tuple[int, bool]:
    """Record samples for one letter. Returns (collected, quit_requested)."""
    label = letter_to_label(letter)
    collected = 0
    while collected < num_samples:
        ok, frame = cap.read()
        if not ok:
            return collected, True
        frame = cv2.flip(frame, 1)
        box = tracker.process(frame)
        if box is not None:
            writer.writerow([label] + landmark_features(box).tolist())
            collected += 1
        if _show(frame, box, [f"{letter}: {collected}/{num_samples}"]):
            return collected, True
        time.sleep(0.05)
    return collected, False


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--letter", help="Collect a single letter, e.g. A")
    group.add_argument("--all", action="store_true", help="Guided session over every letter (A-Y, no J)")
    parser.add_argument("--num-samples", type=int, default=150, help="Samples per letter")
    parser.add_argument("--countdown", type=int, default=3, help="Seconds to get ready before each letter (--all)")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    letters = LETTERS if args.all else [args.letter.upper()]
    for letter in letters:
        letter_to_label(letter)  # validate before opening the camera

    tracker = HandTracker(max_hands=1)
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera_index}.")

    is_new_file = not OUT_CSV.exists()
    OUT_CSV.parent.mkdir(exist_ok=True)
    f = open(OUT_CSV, "a", newline="")
    writer = csv.writer(f)
    if is_new_file:
        writer.writerow(["label"] + [f"f{i}" for i in range(NUM_FEATURES)])

    total = 0
    try:
        for letter in letters:
            if args.all and countdown(cap, tracker, letter, args.countdown):
                break
            print(f"Collecting {args.num_samples} samples for '{letter}'. Press 'q' to stop.")
            collected, quit_requested = collect_letter(cap, tracker, writer, letter, args.num_samples)
            f.flush()
            total += collected
            print(f"  {letter}: {collected} samples")
            if quit_requested:
                break
    finally:
        f.close()
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()

    print(f"Saved {total} samples to {OUT_CSV}")
    print("Now retrain:  python train_landmark_classifier.py")


if __name__ == "__main__":
    main()
