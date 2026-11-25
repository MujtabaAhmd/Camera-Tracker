"""
Shared feature code for the gesture project (numpy only, no camera libraries).

Used by:
  collect_gestures.py  (writes the CSV)
  train_gestures.py    (recomputes features from the raw landmark columns)
  gesture_runtime.py   (computes the same features live)

Keeping this in ONE place guarantees training and live use see identical features.
If you change normalize_landmarks, just retrain: training recomputes features from the
raw landmark columns in the CSV, so you do not need to re-record anything.
"""

import numpy as np

N_LANDMARKS = 21
N_FEATURES = (N_LANDMARKS - 1) * 3   # wrist row is always zero and is dropped -> 60

META_COLS = ["subject_id", "session_id", "label", "take", "variation",
             "frame_idx", "t_sec", "handedness", "handedness_score"]
RAW_COLS = [f"r{i}_{a}" for i in range(N_LANDMARKS) for a in "xyz"]
FEAT_COLS = [f"f{i}_{a}" for i in range(N_LANDMARKS) for a in "xyz"]


def csv_header():
    return META_COLS + RAW_COLS + FEAT_COLS


def normalize_landmarks(pts: np.ndarray, handedness: str):
    """pts: (21, 3) raw landmarks. Returns a flat (63,) array, or None if degenerate.

    - wrist moved to the origin (translation invariance)
    - left hands mirrored so they look like right hands
    - divided by the 2D wrist to middle-finger-base distance (scale invariance)
    """
    p = np.asarray(pts, dtype=np.float32).copy()
    p -= p[0]
    if handedness == "Left":
        p[:, 0] *= -1.0
    scale = float(np.linalg.norm(p[9, :2]))
    if scale < 1e-4:
        return None
    p /= scale
    return p.flatten()


def landmarks_to_features(pts: np.ndarray, handedness: str):
    """The 60-number feature vector fed to the classifiers, or None."""
    flat = normalize_landmarks(pts, handedness)
    if flat is None:
        return None
    return flat[3:]


class NumpyMLP:
    """Inference-only multilayer perceptron using plain numpy.

    Training happens in PyTorch (train_gestures.py), then the weights are copied here,
    so the live demo does not need PyTorch installed. Dropout is a training-only
    technique and is not used here.
    """

    def __init__(self, weights, biases, mean, std, classes):
        self.weights = [np.asarray(w, dtype=np.float32) for w in weights]
        self.biases = [np.asarray(b, dtype=np.float32) for b in biases]
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.classes_ = np.asarray(classes)

    def predict_proba(self, X):
        a = (np.atleast_2d(np.asarray(X, dtype=np.float32)) - self.mean) / self.std
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            a = a @ w + b
            if i < last:
                a = np.maximum(a, 0.0)
        a = a - a.max(axis=1, keepdims=True)
        e = np.exp(a)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
