"""Hand-geometry features from MediaPipe's 21 hand landmarks.

Classifying the 28x28 grayscale crop throws away most of what distinguishes
letters (e.g. A/E/M/N/S/T are all fists that differ in thumb placement) and
memorizes camera/background appearance instead, so it collapses on any new
camera (test_results/real_photo_test.txt). The landmarks describe the hand's
actual shape and don't depend on skin tone, lighting or background.

Features: the 21 (x, y, z) points, translated so the wrist is the origin and
scaled by hand size. Orientation is deliberately kept (G/H/P/Q differ mainly
in which way the hand points).
"""

import numpy as np

NUM_FEATURES = 21 * 3


def landmark_features(box) -> np.ndarray:
    """Feature vector (63,) for a HandBox from hand_tracker.HandTracker."""
    pts = np.asarray(box.landmarks, dtype=np.float32)[:, :3].copy()
    # x and z are normalized by image width, y by height: undo the aspect squash
    pts[:, 0] *= box.aspect
    pts[:, 2] *= box.aspect
    pts -= pts[0]  # wrist -> origin
    scale = np.linalg.norm(pts[:, :2], axis=1).max()
    pts /= max(float(scale), 1e-6)
    return pts.ravel()


def augment_features(X: np.ndarray, y: np.ndarray, copies: int, seed: int = 0):
    """Return X, y plus `copies` jittered versions of every row.

    Each copy gets a random in-plane rotation (+-20 deg), independent x/y
    stretch (+-10%) and small landmark noise, mimicking different hand
    positions, camera angles and detector jitter. This is what keeps the model
    from memorizing the exact hand of the few people in the training photos.
    """
    rng = np.random.default_rng(seed)
    P = np.asarray(X, dtype=np.float32).reshape(-1, 21, 3)
    out_X, out_y = [P.reshape(len(P), NUM_FEATURES)], [y]
    for _ in range(copies):
        theta = np.deg2rad(rng.uniform(-20, 20, len(P)))
        c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
        Q = P.copy()
        Q[:, :, 0] = P[:, :, 0] * c - P[:, :, 1] * s
        Q[:, :, 1] = P[:, :, 0] * s + P[:, :, 1] * c
        Q[:, :, :2] *= rng.uniform(0.9, 1.1, (len(P), 1, 2))
        Q += rng.normal(0, 0.02, Q.shape)
        Q[:, 0, :] = 0  # wrist stays the origin
        Q /= np.maximum(np.linalg.norm(Q[:, :, :2], axis=2).max(axis=1), 1e-6)[:, None, None]
        out_X.append(Q.reshape(len(P), NUM_FEATURES))
        out_y.append(y)
    return np.vstack(out_X), np.concatenate(out_y)


def mirror_features(X: np.ndarray) -> np.ndarray:
    """Mirror image of each feature vector (left hand <-> right hand)."""
    X = np.asarray(X, dtype=np.float32).reshape(-1, 21, 3).copy()
    X[:, :, 0] *= -1
    return X.reshape(len(X), NUM_FEATURES)
