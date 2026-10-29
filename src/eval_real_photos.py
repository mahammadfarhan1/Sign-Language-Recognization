"""Cross-domain check: classify real photos from a different source/camera.

test_results/external_photos_clean/<LETTER>.jpg are real ASL photos that are
in NEITHER Sign MNIST nor the imported training data, so this is the honest
measure of how well the model transfers to new people/cameras/backgrounds
(a held-out split of the training source can't show that).

Usage:  python eval_real_photos.py
"""

from pathlib import Path

import cv2
import joblib

from hand_tracker import HandTracker
from labels import label_to_letter
from landmark_features import landmark_features
from realtime_recognition import MODEL_PATH

PHOTOS_DIR = Path(__file__).resolve().parent.parent / "test_results" / "external_photos_clean"


def main():
    clf = joblib.load(MODEL_PATH)["model"]
    tracker = HandTracker(max_hands=1)
    detected = correct = 0
    photos = sorted(PHOTOS_DIR.glob("*.jpg"))
    for path in photos:
        frame = cv2.imread(str(path))
        box = tracker.process(frame) if frame is not None else None
        if box is None:
            print(f"  {path.stem}: no hand detected")
            continue
        detected += 1
        pred = label_to_letter(clf.predict(landmark_features(box).reshape(1, -1))[0])
        correct += pred == path.stem.upper()
        print(f"  {path.stem}: predicted {pred} {'OK' if pred == path.stem.upper() else 'x'}")
    tracker.close()
    print(f"\nHand detected: {detected}/{len(photos)}")
    print(f"Correct letters (of detected): {correct}/{detected} = {100 * correct / max(detected, 1):.0f}%")


if __name__ == "__main__":
    main()
