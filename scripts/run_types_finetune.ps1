param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Device = "auto",
    [string]$Name = "taco_n_types_finetune_50",
    [string]$Reports = "reports/multiclass/finetune",
    [string]$Config = "configs/multiclass/yolov8n_taco_finetune.yaml",
    [string]$CompareWith = "",
    [switch]$SkipTraining
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
foreach ($target in @("$Reports/validation", "$Reports/test", "$Reports/evaluation.yaml", "models/b/${Name}_handoff.zip")) {
    if (Test-Path -LiteralPath $target) { throw "Output exists: $target. Use a new -Name and -Reports directory." }
}
function Invoke-Python {
    & $Python @args
    if ($LASTEXITCODE -ne 0) { throw "Python failed with exit code $LASTEXITCODE" }
}
if (-not $SkipTraining) {
    Invoke-Python -m src.training.train --config $Config --participant b --name $Name --device $Device
}
if ($CompareWith) {
    Invoke-Python scripts/build_types_eval_config.py --name $Name --baseline taco_n_types_30 --baseline $CompareWith --output "$Reports/evaluation.yaml"
} else {
    Invoke-Python scripts/build_types_eval_config.py --name $Name --baseline taco_n_types_30 --output "$Reports/evaluation.yaml"
}
Invoke-Python -m src.evaluation.evaluate --config "$Reports/evaluation.yaml" --output "$Reports/validation" --device $Device
# Only the validation winner is checked on test, at its already frozen threshold.
Invoke-Python -m src.evaluation.final_test --selection "$Reports/validation/selection.json" --output "$Reports/test" --device $Device
Invoke-Python scripts/verify_types_app.py --reports $Reports
Invoke-Python scripts/build_types_report.py --reports $Reports --archive-name $Name --recommend
