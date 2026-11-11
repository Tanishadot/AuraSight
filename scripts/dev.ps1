# Development: backend on :8000 + Vite hot-reload UI on :5173 (proxying /api and /ws).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$backend = Start-Process -PassThru -NoNewWindow ".\.venv\Scripts\python.exe" "-m backend.main"
try {
  Set-Location frontend
  if (-not (Test-Path "node_modules")) { npm install --no-audit --no-fund }
  Start-Process "http://127.0.0.1:5173"
  npm run dev
} finally {
  Stop-Process -Id $backend.Id -ErrorAction SilentlyContinue
}
