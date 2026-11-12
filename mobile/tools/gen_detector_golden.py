"""Reference detections from the *desktop* detector, for the mobile parity tests.

Runs backend/detection/yolo_onnx.py (YOLO11n, ONNX Runtime, the desktop's own
letterbox, decoding and NMS) on every fixture image at 320 px - the mobile
model's input size - and writes the results to
mobile/app/integration_test/golden/desktop_detections_320.json.

Run from the repository root with the desktop virtualenv:
    .venv/Scripts/python mobile/tools/gen_detector_golden.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.detection.yolo_onnx import COCO_NAMES, YoloDetector  # noqa: E402

SCENES = ROOT / "tests" / "fixtures" / "scenes"
OUT = ROOT / "mobile" / "app" / "integration_test" / "golden" / "desktop_detections_320.json"
SIZE, CONF = 320, 0.25


def main() -> None:
    det = YoloDetector(ROOT / "models" / "yolo11n.onnx")
    images = []
    for p in sorted(SCENES.glob("*.jpg")):
        img = cv2.imread(str(p))
        d = det.detect(img, SIZE, conf=CONF)
        images.append({
            "file": p.name,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "width": img.shape[1],
            "height": img.shape[0],
            "detections": [
                {"label": COCO_NAMES[int(c)], "cls": int(c), "score": round(float(s), 4),
                 "box": [round(float(v), 2) for v in b]}
                for b, s, c in zip(d.boxes, d.scores, d.class_ids)
            ],
        })
        print(f"{p.name}: " + ", ".join(f"{x['label']} {x['score']:.2f}" for x in images[-1]["detections"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"size": SIZE, "conf": CONF, "model": "yolo11n.onnx (desktop)", "images": images},
                              indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
