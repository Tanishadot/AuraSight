#!/usr/bin/env bash
# Development on macOS/Linux: backend :8000 + Vite UI :5173
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] || { python3 -m venv .venv; .venv/bin/pip install -r requirements-dev.txt; }
[ -f models/yolo11n.onnx ] || .venv/bin/python scripts/setup_models.py
.venv/bin/python -m backend.main & BACK=$!
trap 'kill $BACK 2>/dev/null' EXIT
cd frontend && { [ -d node_modules ] || npm install --no-audit --no-fund; } && npm run dev
