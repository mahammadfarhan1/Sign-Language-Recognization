"""Feature extraction shared by training and real-time inference."""

import cv2
import numpy as np


def gradient_features(images: np.ndarray) -> np.ndarray:
    """Convert 28x28 grayscale images into gradient histograms."""
    features = []
    for image in images:
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=False)
        image_features = []
        for row in range(0, 28, 4):
            for col in range(0, 28, 4):
                cell_magnitude = magnitude[row : row + 4, col : col + 4].ravel()
                cell_angle = angle[row : row + 4, col : col + 4].ravel()
                bins = (cell_angle * 9 / (2 * np.pi)).astype(np.int32) % 9
                image_features.extend(
                    np.bincount(
                        bins, weights=cell_magnitude, minlength=9
                    )
                )
        features.append(image_features)
    return np.asarray(features, dtype=np.float32)