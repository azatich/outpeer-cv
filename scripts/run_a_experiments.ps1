param(
    [int]$Epochs = 20,
    [int]$Batch = 4,
    [string]$Device = '0',
    [string]$Suffix = '20'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) { throw 'Create .venv and install requirements first.' }
if ($Suffix -notmatch '^[A-Za-z0-9_-]+$') { throw 'Suffix must contain only letters, digits, underscore or hyphen.' }
if ($Epochs -lt 1 -or $Batch -lt 1) { throw 'Epochs and Batch must be positive.' }
Push-Location -LiteralPath $projectRoot
try {
    $baselineName = 'a_n_baseline_' + $Suffix
    $augmentedName = 'a_n_aug_' + $Suffix
    & $projectPython -m src.training.train --config configs/a/yolov8n_baseline.yaml --epochs $Epochs --batch $Batch --device $Device --name $baselineName
    if ($LASTEXITCODE -ne 0) { throw 'Baseline training failed; see runs/a for status.' }
    & $projectPython -m src.training.train --config configs/a/yolov8n_aug.yaml --epochs $Epochs --batch $Batch --device $Device --name $augmentedName
    if ($LASTEXITCODE -ne 0) { throw 'Augmented training failed; see runs/a for status.' }
    & $projectPython -m src.training.summarize ('models/a/' + $baselineName) ('models/a/' + $augmentedName)
    if ($LASTEXITCODE -ne 0) { throw 'Comparison failed.' }
} finally {
    Pop-Location
}
