# run_training.ps1
# Run the ML training pipeline and save timestamped logs.

$ErrorActionPreference = "Stop"

# 1. Resolve the MLOps project root from this script's location.
$ProjectRoot = Split-Path -Parent $PSScriptRoot

# 2. Locate the virtual environment.
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment not found at: $VenvPython"
}

# 3. Locate the training script.
# Update this path if your train.py is somewhere else.
$TrainScript = Join-Path $ProjectRoot "mlops-day1\src\train.py"

if (-not (Test-Path $TrainScript)) {
    throw "Training script not found at: $TrainScript"
}

# 4. Create the log directory.
$LogDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Path $LogDir -Force |
    Out-Null

# 5. Generate a timestamped log filename.
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "training_$Timestamp.log"

Write-Host "Starting ML training..."
Write-Host "Project root: $ProjectRoot"
Write-Host "Python: $VenvPython"
Write-Host "Training script: $TrainScript"
Write-Host "Log file: $LogFile"

# 6. Run training from the project root and capture output.
Push-Location $ProjectRoot

try {
    & $VenvPython $TrainScript 2>&1 |
        Tee-Object -FilePath $LogFile

    $TrainingExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

# 7. Check the Python exit code.
if ($TrainingExitCode -ne 0) {
    throw "Training failed with exit code $TrainingExitCode. Inspect $LogFile"
}

Write-Host "Training completed successfully."
Write-Host "Log saved to: $LogFile"