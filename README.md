# Real-Time Sign Language Recognition

Recognizes static ASL alphabet letters (A-Y, excluding J and Z, which
require motion) from a webcam feed in real time, and spells out words and
sentences.

## How it works

1. **Detection** — `src/hand_tracker.py` wraps MediaPipe's `HandLandmarker`
   (`models/hand_landmarker.task`) to find the hand and its 21 landmarks.
2. **Features** — `src/landmark_features.py` turns the landmarks into a
   63-number hand-geometry vector (wrist at the origin, scaled by hand size).
   Geometry doesn't depend on skin tone, lighting or background.
3. **Classification** — an RBF SVM (`src/train_landmark_classifier.py`,
   scikit-learn) predicts the letter. Training data is augmented (rotation,
   stretch, noise) and mirrored so either hand works.
4. **Words and sentences** — `src/sentence_builder.py`: hold a sign ~1 s to
   type it, hide your hand ~2 s for a space.
5. **Live app** — `src/realtime_recognition.py`: webcam -> landmarks ->
   classify -> smooth over recent frames -> overlay letter and sentence.

### Training data

| Source | What | Used how |
|---|---|---|
| Hugging Face [`Marxulia/asl_sign_languages_alphabets_v03`](https://huggingface.co/datasets/Marxulia/asl_sign_languages_alphabets_v03) | ~10k real photos of the letters (mostly one signer, studio background; **no license stated on its card**) | `import_external_data.py` extracts landmarks; 7,763 photos had a detected hand |
| Your own webcam | frames you record with `collect_data.py` | weighted x3, most accurate for you |

## Run everything

```powershell
cd C:\Users\moham\SignLanguageRecognition
.\run_all.ps1              # import photos (first time), train, test, start the app
.\run_all.ps1 -Collect     # ...but first record every letter with your webcam
```

(If PowerShell blocks it: `powershell -ExecutionPolicy Bypass -File .\run_all.ps1`.)

Or step by step:

```powershell
cd C:\Users\moham\SignLanguageRecognition\src
..\.venv\Scripts\python.exe import_external_data.py       # photos -> data/external_landmarks.csv (~3 min, once)
..\.venv\Scripts\python.exe collect_data.py --all         # optional: record your own signs
..\.venv\Scripts\python.exe train_landmark_classifier.py  # trains, prints held-out accuracy, saves the model
..\.venv\Scripts\python.exe eval_real_photos.py           # test on photos of other people
..\.venv\Scripts\python.exe realtime_recognition.py       # live app
```

Setup (already done here): `python -m venv .venv` then
`.venv\Scripts\pip install -r requirements.txt`. The photo dataset downloads to
`data/external/asl_v03.parquet` (`run_all.ps1` does it automatically; the
command is in `import_external_data.py`).

### Using the live app

- **Hold a sign steady for ~1 second** (watch the green bar) to type that letter.
- Double letter (the `LL` in HELLO): drop your hand out of view, or change sign, between the two.
- **Hide your hand for ~2 seconds** to add a space between words.
- Keys: `SPACE` space, `BACKSPACE` delete, `C` clear, `Q` quit (final text is printed).

## Results (this build) — please read

| Test | Result |
|---|---|
| Held-out photos from the training source (last 20% of every letter, no leakage) | **93.4%** |
| 22 photos of *other people / another camera* (`eval_real_photos.py`) | **9 of 20 detected hands correct (45%)**; hand not detected in 2 of 22 |

- The 93.4% clears a 90% bar, but it is measured on the same signer(s) as the
  training photos. On genuinely new people it is much lower, because the
  public photos show very few different hands.
- **The fix is your own data**: run `collect_data.py --all` and retrain. The
  model then learns *your* hand and camera. `train_landmark_classifier.py`
  reports the held-out accuracy on your own samples separately.
- Confusable letters in the report: P, U, V, H (`test_results/landmark_classification_report.txt`).

### Why not raw pixels (the original approach)

`src/train_classifier.py` (RBF SVM on 28x28 gradient features, Sign Language
MNIST) still runs and scores **95.7-96%** on the MNIST test split, but that
number is misleading: on real photos it scored **0 of 20**, and even after adding
real photos its held-out accuracy on them was only **29%**. A 28x28 grayscale image
can't separate letters such as A/E/M/N/S/T and memorizes the camera instead.
It's kept as a baseline only; the live app uses the landmark model.

## Limitations

- **J and Z** need motion and are not recognized.
- Whole-word signs (e.g. "thank you") are not recognized; words are spelled letter by letter.
- Accuracy on a new person/camera is well below the held-out figure until you record your own samples.

## Files

`test_results/`: `landmark_metrics.json`, `landmark_classification_report.txt`
(landmark model); `metrics.json`, `classification_report.txt`,
`confusion_matrix.png`, `sample_predictions.png` (pixel baseline);
`real_photo_test.txt` (original analysis of why pixels fail).
