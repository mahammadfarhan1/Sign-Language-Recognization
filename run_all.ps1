# Runs the whole project: (optionally) record your own signs -> train -> live app.
#
#   .\run_all.ps1            import the public photo dataset if needed, train, then
#                            start the live app (uses your samples if already recorded)
#   .\run_all.ps1 -Collect   first record every letter with your webcam, then train
#                            (learning from your signs) and start the live app
#   .\run_all.ps1 -Collect -Samples 100     fewer samples per letter (default 150)
#
# If PowerShell blocks the script:  powershell -ExecutionPolicy Bypass -File .\run_all.ps1
param(
    [switch]$Collect,
    [int]$Samples = 150
)

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$data = Join-Path $PSScriptRoot "data"
Set-Location (Join-Path $PSScriptRoot "src")

# 1. Real photos -> landmark training rows (downloads the dataset first if missing)
if (-not (Test-Path "$data\external_landmarks.csv")) {
    $parquet = "$data\external\asl_v03.parquet"
    if (-not (Test-Path $parquet)) {
        New-Item -ItemType Directory -Force "$data\external" | Out-Null
        curl.exe -L -o $parquet "https://huggingface.co/datasets/Marxulia/asl_sign_languages_alphabets_v03/resolve/main/data/train-00000-of-00001.parquet"
    }
    & $py import_external_data.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# 2. Your own webcam samples (optional, most accurate for you)
if ($Collect) {
    & $py collect_data.py --all --num-samples $Samples
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# 3. Train, check on real photos from other people, run the live app
& $py train_landmark_classifier.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py eval_real_photos.py

& $py realtime_recognition.py
