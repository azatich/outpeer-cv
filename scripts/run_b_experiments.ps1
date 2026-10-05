param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Device = "auto",
    [switch]$SkipTraining,
    [switch]$FinalTest
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Invoke-Python {
    & $Python @args
    if ($LASTEXITCODE -ne 0) { throw "Python failed with exit code $LASTEXITCODE" }
}

Invoke-Python -m src.evaluation.dataset
if (-not $SkipTraining) {
    Invoke-Python -m src.training.train --config configs/b/yolov8s_baseline.yaml --participant b --device $Device
}
Invoke-Python -m src.evaluation.evaluate --config configs/b/evaluation.yaml --device $Device
Invoke-Python -m src.evaluation.analyze
Invoke-Python -m src.evaluation.benchmark
if ($FinalTest) {
    # This step reads the frozen validation selection and cannot tune on test.
    Invoke-Python -m src.evaluation.final_test --device $Device
    Invoke-Python scripts/build_b_report.py
    Invoke-Python scripts/build_b_notebooks.py --execute
}
