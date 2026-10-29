"""Thin wrapper around MediaPipe's HandLandmarker (Tasks API) for locating a
hand in a video frame.

mediapipe>=1.0 removed the old `mp.solutions.hands` API; hand detection now
goes through `mediapipe.tasks.python.vision.HandLandmarker`, which requires
the `hand_landmarker.task` model bundle (see models/hand_landmarker.task,
downloaded from the official MediaPipe model zoo).
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


@dataclass
class HandBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    landmarks: np.ndarray  # (21, 3) normalized x, y, z
    aspect: float = 1.0  # image width / height, needed to un-squash normalized x


class HandTracker:
    def __init__(self, max_hands: int = 1, min_detection_confidence: float = 0.5):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"HandLandmarker model not found at {MODEL_PATH}. Download it from "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task"
            )
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray) -> HandBox | None:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None

        hand_landmarks = result.hand_landmarks[0]
        coords = np.array(
            [(lm.x, lm.y, lm.z) for lm in hand_landmarks], dtype=np.float32
        )
        xs = coords[:, 0] * w
        ys = coords[:, 1] * h

        pad = 0.25  # extra margin around the hand so fingertips aren't clipped
        x_min = xs.min()
        x_max = xs.max()
        y_min = ys.min()
        y_max = ys.max()
        box_w = x_max - x_min
        box_h = y_max - y_min
        x_min -= box_w * pad
        x_max += box_w * pad
        y_min -= box_h * pad
        y_max += box_h * pad

        return HandBox(
            x_min=max(int(x_min), 0),
            y_min=max(int(y_min), 0),
            x_max=min(int(x_max), w),
            y_max=min(int(y_max), h),
            landmarks=coords,
            aspect=w / h,
        )

    def close(self):
        self._landmarker.close()
