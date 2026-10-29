"""Real-time sign language (ASL alphabet) recognition from a webcam feed.

Pipeline:
  1. Capture a frame from the webcam.
  2. Use MediaPipe to find the hand and its 21 landmarks.
  3. Turn the landmarks into a size/position-normalized feature vector
     (landmark_features.py).
  4. Classify it with the model from train_landmark_classifier.py
     (models/asl_landmarks.joblib).
  5. Smooth over recent frames, build words/sentences (sentence_builder.py)
     and overlay everything on the video feed.

Run with:  python realtime_recognition.py
Press 'q' to quit.
"""

import time
from collections import deque
from pathlib import Path

import cv2
import joblib

from hand_tracker import HandTracker
from labels import label_to_letter
from landmark_features import landmark_features
from sentence_builder import SentenceBuilder

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "asl_landmarks.joblib"


def main(camera_index: int = 0):
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Model not found at {MODEL_PATH}. Run train_landmark_classifier.py first."
        )
    bundle = joblib.load(MODEL_PATH)
    clf = bundle["model"]

    tracker = HandTracker(max_hands=1)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {camera_index}.")

    fps_window = deque(maxlen=30)
    pred_window = deque(maxlen=8)  # smooth predictions over recent frames
    sentence = SentenceBuilder()

    print(
        "Starting real-time recognition. Hold a sign ~1s to type it; hide your "
        "hand ~2s for a space.\n"
        "Keys: SPACE = space, BACKSPACE = delete, C = clear, Q = quit."
    )
    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            box = tracker.process(frame)

            label_text = "No hand detected"
            letter = None
            if box is not None:
                pred_label = clf.predict(landmark_features(box).reshape(1, -1))[0]
                pred_window.append(pred_label)
                # majority vote over recent frames for stability
                smoothed_label = max(set(pred_window), key=pred_window.count)
                letter = label_to_letter(smoothed_label)
                label_text = letter

                cv2.rectangle(
                    frame, (box.x_min, box.y_min), (box.x_max, box.y_max),
                    (0, 200, 0), 2,
                )
            else:
                pred_window.clear()  # don't let stale votes leak into the next sign

            progress = sentence.update(letter, time.time())

            fps_window.append(1.0 / max(time.time() - t0, 1e-6))
            fps = sum(fps_window) / len(fps_window)

            cv2.putText(
                frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 0, 255), 3,
            )
            # hold-to-type progress bar under the current letter
            cv2.rectangle(frame, (20, 65), (220, 77), (80, 80, 80), 1)
            cv2.rectangle(frame, (20, 65), (20 + int(200 * progress), 77), (0, 200, 0), -1)

            # sentence banner along the bottom (last 32 chars so it fits)
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h - 90), (w, h), (0, 0, 0), -1)
            cv2.putText(
                frame, sentence.text[-32:] + "_", (20, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2,
            )
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (20, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1,
            )

            cv2.imshow("Sign Language Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == 32:
                sentence.add_space()
            elif key == 8:
                sentence.backspace()
            elif key == ord("c"):
                sentence.clear()
    finally:
        if sentence.text.strip():
            print(f"Final text: {sentence.text.strip()}")
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    main()
