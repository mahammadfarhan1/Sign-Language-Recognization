"""Train a hand-sign letter classifier on the Sign Language MNIST dataset.

The dataset (https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
contains 28x28 grayscale images of ASL hand signs for the letters A-Z,
excluding J and Z (which require motion). Each row in the CSV is
`label, pixel1, pixel2, ..., pixel784`.

This script trains a RandomForest classifier on the pixel data, evaluates it
on the held-out test split, and saves the trained model plus evaluation
artifacts (confusion matrix, classification report, sample predictions).
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.svm import SVC

from image_features import gradient_features
from labels import LABEL_TO_LETTER, label_to_letter

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "test_results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Extra training data, used automatically when the CSV exists:
#   external - real photos imported with import_external_data.py
#   webcam   - your own frames recorded with collect_data.py
# Both went through the live detect -> crop -> 28x28 pipeline, and real-camera
# images are what the app actually sees, so each counts for more than one
# Sign MNIST image during training. (name, csv, sample weight)
EXTRA_SOURCES = [
    ("external", DATA_DIR / "external_samples.csv", 2.0),
    ("webcam", DATA_DIR / "custom_samples.csv", 3.0),
]
HOLDOUT_FRACTION = 0.2
# Warn if the public test split falls below this.
MIN_ACCURACY = 0.90


def load_split(csv_path: Path):
    df = pd.read_csv(csv_path)
    y = df["label"].to_numpy()
    images = df.drop(columns=["label"]).to_numpy(dtype=np.uint8).reshape(-1, 28, 28)
    X = gradient_features(images)
    return X, y


def load_extra_samples(csv_path: Path):
    """Load an extra-data CSV and split it into train / held-out parts.

    Rows are recorded consecutively, so neighbouring frames are near-duplicates.
    A random split would leak them into the held-out set, so the last
    HOLDOUT_FRACTION of each letter's rows are held out instead.
    Returns (X_train, y_train, X_holdout, y_holdout), or None if there is no data.
    """
    if not csv_path.exists():
        return None
    X, y = load_split(csv_path)
    if len(y) == 0:
        return None
    train_idx, hold_idx = [], []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        cut = int(len(idx) * (1 - HOLDOUT_FRACTION))
        train_idx.extend(idx[:cut])
        hold_idx.extend(idx[cut:])
    return X[train_idx], y[train_idx], X[hold_idx], y[hold_idx]


def main():
    print("Loading training data...")
    X_train, y_train = load_split(DATA_DIR / "sign_mnist_train.csv")
    print(f"  {X_train.shape[0]} training samples, {X_train.shape[1]} pixel features")

    print("Loading test data...")
    X_test, y_test = load_split(DATA_DIR / "sign_mnist_test.csv")
    print(f"  {X_test.shape[0]} test samples")

    sample_weight = np.ones(len(y_train))
    holdouts = {}  # source name -> (X_holdout, y_holdout)
    extra_stats = {}
    for name, csv_path, weight in EXTRA_SOURCES:
        extra = load_extra_samples(csv_path)
        if extra is None:
            print(f"No {name} samples ({csv_path.name} not found).")
            continue
        X_e, y_e, X_e_hold, y_e_hold = extra
        print(
            f"{name} samples: {len(y_e)} for training, {len(y_e_hold)} held out "
            f"({len(set(y_e.tolist()))} letters) -> weight x{weight:g}"
        )
        X_train = np.vstack([X_train, X_e])
        y_train = np.concatenate([y_train, y_e])
        sample_weight = np.concatenate([sample_weight, np.full(len(y_e), weight)])
        holdouts[name] = (X_e_hold, y_e_hold)
        extra_stats[f"{name}_train_samples"] = int(len(y_e))
        extra_stats[f"{name}_holdout_samples"] = int(len(y_e_hold))

    classes = sorted(set(y_train.tolist()))
    print(f"Classes present: {len(classes)} -> {[label_to_letter(c) for c in classes]}")

    print("\nTraining RBF SVM on gradient features...")
    t0 = time.time()
    clf = SVC(
        C=10,
        kernel="rbf",
        gamma="scale",
        cache_size=4096,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    train_time = time.time() - t0
    print(f"Training finished in {train_time:.1f}s")

    print("\nEvaluating on held-out test set...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc * 100:.2f}%")
    if acc < MIN_ACCURACY:
        print(f"WARNING: accuracy is below the {MIN_ACCURACY * 100:.0f}% target.")

    for name, (X_hold, y_hold) in holdouts.items():
        hold_acc = accuracy_score(y_hold, clf.predict(X_hold))
        print(f"Held-out {name} accuracy: {hold_acc * 100:.2f}%  ({len(y_hold)} samples)")
        extra_stats[f"{name}_holdout_accuracy"] = round(float(hold_acc), 4)

    letters_present = [label_to_letter(c) for c in classes]
    report = classification_report(
        y_test, y_pred, labels=classes, target_names=letters_present, digits=3
    )
    print("\nClassification report:\n" + report)

    # Save model
    model_path = MODELS_DIR / "sign_mnist_rf.joblib"
    joblib.dump({"model": clf, "classes": classes}, model_path)
    print(f"\nSaved model to {model_path}")

    # Save metrics
    metrics = {
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "num_classes": len(classes),
        "train_time_seconds": round(train_time, 2),
        "test_accuracy": round(float(acc), 4),
    }
    metrics.update(extra_stats)
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(RESULTS_DIR / "classification_report.txt", "w") as f:
        f.write(f"Test accuracy: {acc * 100:.2f}%\n\n")
        f.write(report)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=letters_present,
        yticklabels=letters_present,
        ax=ax,
    )
    ax.set_xlabel("Predicted letter")
    ax.set_ylabel("True letter")
    ax.set_title(f"Sign Language Letter Recognition - Confusion Matrix (acc={acc*100:.1f}%)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    print(f"Saved confusion matrix to {RESULTS_DIR / 'confusion_matrix.png'}")

    # Sample predictions grid
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test), size=16, replace=False)
    test_images = (
        pd.read_csv(DATA_DIR / "sign_mnist_test.csv")
        .drop(columns=["label"])
        .to_numpy(dtype=np.float32)
        .reshape(-1, 28, 28)
        / 255.0
    )
    fig2, axes = plt.subplots(4, 4, figsize=(9, 9))
    for ax, idx in zip(axes.flat, sample_idx):
        img = test_images[idx]
        true_letter = label_to_letter(y_test[idx])
        pred_letter = label_to_letter(y_pred[idx])
        correct = true_letter == pred_letter
        ax.imshow(img, cmap="gray")
        ax.set_title(
            f"true:{true_letter} pred:{pred_letter}",
            color="green" if correct else "red",
            fontsize=10,
        )
        ax.axis("off")
    fig2.suptitle("Sample test predictions")
    fig2.tight_layout()
    fig2.savefig(RESULTS_DIR / "sample_predictions.png", dpi=150)
    print(f"Saved sample predictions to {RESULTS_DIR / 'sample_predictions.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
