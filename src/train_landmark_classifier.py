"""Train the ASL letter classifier on MediaPipe hand-landmark features.

Data (each optional, used when the CSV exists):
  data/external_landmarks.csv  real photos, from import_external_data.py
  data/custom_landmarks.csv    your own webcam, from collect_data.py (weighted x3,
                               since it matches the camera the app really sees)

The last 20% of each letter's rows are held out for evaluation. Rows are
consecutive frames, so a random split would leak near-duplicates into the
held-out set and inflate the score. Training data is augmented with mirrored
copies (left hand <-> right hand; the live view is also mirrored).

Two candidate models are compared on the held-out split; the better one is
refit on ALL the data and saved to models/asl_landmarks.joblib.

Usage:  python train_landmark_classifier.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from labels import label_to_letter
from landmark_features import augment_features, mirror_features

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "asl_landmarks.joblib"
RESULTS_DIR = ROOT / "test_results"

# (name, csv, sample weight)
SOURCES = [
    ("external", DATA_DIR / "external_landmarks.csv", 1.0),
    ("webcam", DATA_DIR / "custom_landmarks.csv", 3.0),
]
HOLDOUT_FRACTION = 0.2
AUGMENT_COPIES = 3
MIN_ACCURACY = 0.90


def expand(X, y):
    """Augment + mirror: every row becomes 2 * (1 + AUGMENT_COPIES) rows."""
    X, y = augment_features(X, y, AUGMENT_COPIES)
    return np.vstack([X, mirror_features(X)]), np.concatenate([y, y])


def load_source(csv_path: Path):
    """Return (X_train, y_train, X_hold, y_hold) with a per-letter block split."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    y = df["label"].to_numpy()
    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    train_idx, hold_idx = [], []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        cut = int(len(idx) * (1 - HOLDOUT_FRACTION))
        train_idx.extend(idx[:cut])
        hold_idx.extend(idx[cut:])
    return X[train_idx], y[train_idx], X[hold_idx], y[hold_idx]


def candidates():
    return {
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
        "ExtraTrees": make_pipeline(
            StandardScaler(), ExtraTreesClassifier(n_estimators=400, n_jobs=-1, random_state=0)
        ),
    }


def fit(model, X, y, w):
    model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    return model


def main():
    Xs, ys, ws = [], [], []          # training pool (augmented + mirrored)
    Xf, yf, wf = [], [], []          # same, but including the held-out rows (final refit)
    Xh_all, yh_all, holdouts = [], [], {}
    stats = {}
    for name, path, weight in SOURCES:
        data = load_source(path) if path.exists() else None
        if data is None:
            print(f"No {name} data ({path.name} not found).")
            continue
        X_tr, y_tr, X_ho, y_ho = data
        print(f"{name}: {len(y_tr)} train / {len(y_ho)} held out, {len(set(y_tr.tolist()))} letters, weight x{weight:g}")
        X_a, y_a = expand(X_tr, y_tr)
        Xs.append(X_a); ys.append(y_a); ws.append(np.full(len(y_a), weight))
        X_b, y_b = expand(np.vstack([X_tr, X_ho]), np.concatenate([y_tr, y_ho]))
        Xf.append(X_b); yf.append(y_b); wf.append(np.full(len(y_b), weight))
        Xh_all.append(X_ho); yh_all.append(y_ho)
        holdouts[name] = (X_ho, y_ho)
        stats[f"{name}_train_samples"] = int(len(y_tr))
        stats[f"{name}_holdout_samples"] = int(len(y_ho))
    if not Xs:
        raise SystemExit("No training data. Run import_external_data.py and/or collect_data.py first.")

    X_train, y_train, w_train = np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)
    X_hold, y_hold = np.vstack(Xh_all), np.concatenate(yh_all)
    classes = sorted(set(y_train.tolist()))
    print(f"Classes: {[label_to_letter(c) for c in classes]}\n")

    best_name, best_acc, best_model = None, -1.0, None
    for name, model in candidates().items():
        t0 = time.time()
        fit(model, X_train, y_train, w_train)
        acc = accuracy_score(y_hold, model.predict(X_hold))
        print(f"{name}: held-out accuracy {acc * 100:.2f}%  (trained in {time.time() - t0:.1f}s)")
        if acc > best_acc:
            best_name, best_acc = name, acc
    print(f"\nBest: {best_name}")

    # Per-source and per-letter detail for the winner (fit again on the split so numbers match the report)
    model = fit(candidates()[best_name], X_train, y_train, w_train)
    y_pred = model.predict(X_hold)
    for name, (X_ho, y_ho) in holdouts.items():
        a = accuracy_score(y_ho, model.predict(X_ho))
        print(f"  held-out {name}: {a * 100:.2f}%")
        stats[f"{name}_holdout_accuracy"] = round(float(a), 4)
    report = classification_report(
        y_hold, y_pred, labels=classes, target_names=[label_to_letter(c) for c in classes], digits=3
    )
    print("\n" + report)
    if best_acc < MIN_ACCURACY:
        print(f"WARNING: held-out accuracy is below the {MIN_ACCURACY * 100:.0f}% target.")

    # Final model: refit on everything, including the held-out rows
    final = fit(candidates()[best_name], np.vstack(Xf), np.concatenate(yf), np.concatenate(wf))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": final, "classes": classes, "name": best_name}, MODEL_PATH)
    print(f"Saved {best_name} to {MODEL_PATH}")

    RESULTS_DIR.mkdir(exist_ok=True)
    stats.update({"model": best_name, "holdout_accuracy": round(float(best_acc), 4), "num_classes": len(classes)})
    (RESULTS_DIR / "landmark_metrics.json").write_text(json.dumps(stats, indent=2))
    (RESULTS_DIR / "landmark_classification_report.txt").write_text(
        f"Held-out accuracy ({best_name}): {best_acc * 100:.2f}%\n\n{report}"
    )


if __name__ == "__main__":
    main()
