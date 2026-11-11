# AuraSight one-command start (Windows). Builds the UI if needed, then serves
# everything from the local backend at http://127.0.0.1:8000
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (-not (Test-Path ".venv")) {
  py -3.11 -m venv .venv
  .\.venv\Scripts\python -m pip install -r requirements-dev.txt
}
if (-not (Test-Path "models\yolo11n.onnx")) { .\.venv\Scripts\python scripts\setup_models.py }
if (-not (Test-Path "frontend\dist\index.html")) {
  Push-Location frontend
  if (-not (Test-Path "node_modules")) { npm install --no-audit --no-fund }
  npm run build
  Pop-Location
}
Start-Process "http://127.0.0.1:8000"
.\.venv\Scripts\python -m backend.main
