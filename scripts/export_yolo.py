"""One-time export of YOLO11n to ONNX (dynamic input size).

Needs PyTorch + Ultralytics, which AuraSight itself does NOT need at runtime.
Use a throwaway environment:

    python -m venv .export-env
    .export-env\\Scripts\\pip install ultralytics onnx      (Windows)
    .export-env/bin/pip install ultralytics onnx            (macOS/Linux)
    <that python> scripts/export_yolo.py
"""
from pathlib import Path
import shutil

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
pt = ROOT / "models" / "yolo11n.pt"
model = YOLO(str(pt) if pt.exists() else "yolo11n.pt")
out = Path(model.export(format="onnx", imgsz=640, dynamic=True, simplify=False, opset=17))
dest = ROOT / "models" / "yolo11n.onnx"
if out.resolve() != dest.resolve():
    shutil.move(str(out), dest)
print("exported", dest)
