param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Device = "auto",
    [string]$Name = "taco_n_types_30",
    [string]$Reports = "reports/multiclass",
    [switch]$SkipData,
    [switch]$SkipTraining
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
foreach ($target in @("$Reports/validation", "$Reports/test", "$Reports/evaluation.yaml")) {
    if (Test-Path -LiteralPath $target) { throw "Output exists: $target. Use a new -Reports directory to preserve prior results." }
}
if (-not $SkipTraining -and (Test-Path -LiteralPath "models/b/$Name")) {
    throw "Model run exists: $Name. Use a new -Name."
}
function Invoke-Python {
    & $Python @args
    if ($LASTEXITCODE -ne 0) { throw "Python failed with exit code $LASTEXITCODE" }
}
if (-not $SkipData) {
    Invoke-Python -m src.data.taco
}
if (-not $SkipTraining) {
    Invoke-Python -m src.training.train --config configs/multiclass/yolov8n_taco.yaml --participant b --name $Name --device $Device
}
# Build a run-specific evaluation config: new report folders never overwrite the
# frozen validation/test of an earlier run. The prepared data split stays fixed.
Invoke-Python scripts/build_types_eval_config.py --name $Name --output "$Reports/evaluation.yaml"
Invoke-Python -m src.evaluation.evaluate --config "$Reports/evaluation.yaml" --output "$Reports/validation" --device $Device
Invoke-Python -m src.evaluation.final_test --selection "$Reports/validation/selection.json" --output "$Reports/test" --device $Device
Invoke-Python scripts/build_types_report.py --reports $Reports --name $Name
