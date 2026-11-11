"""Measure real performance on this machine.

    python scripts/benchmark.py            # detector + tracker (+ VLM if llama-server is running)

Reports median latency per inference size, achievable detection FPS, process
memory, and VLM latency when available. Numbers go in the README's hardware
section; do not trust anyone's numbers but your own machine's.
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import psutil  # noqa: E402

from backend.config.settings import MODELS_DIR, Settings  # noqa: E402
from backend.detection.yolo_onnx import YoloDetector  # noqa: E402
from backend.tracking.bytetrack import ByteTracker  # noqa: E402

proc = psutil.Process(os.getpid())
img = cv2.imread(str(ROOT / "tests" / "fixtures" / "bus.jpg"))
img = cv2.resize(img, (640, 480))  # typical webcam frame
rss0 = proc.memory_info().rss
det = YoloDetector(MODELS_DIR / "yolo11n.onnx")
det.load()
print(f"CPU: {os.cpu_count()} logical cores | RAM total {psutil.virtual_memory().total / 2**30:.1f} GB")
print(f"\nDetector  (YOLO11n ONNX, CPU, {det._session.get_session_options().intra_op_num_threads} threads)")
for size in (256, 320, 416, 512, 640):
    for _ in range(3):
        det.detect(img, size)
    ts = []
    for _ in range(25):
        t = time.perf_counter()
        det.detect(img, size, conf=0.1)
        ts.append((time.perf_counter() - t) * 1000)
    med = statistics.median(ts)
    print(f"  {size:>3}px  median {med:6.1f} ms  p90 {sorted(ts)[22]:6.1f} ms  -> max ~{1000 / med:4.1f} FPS")

tr = ByteTracker()
r = det.detect(img, 416, conf=0.1)
ts = []
for _ in range(200):
    t = time.perf_counter()
    tr.update(r.boxes, r.scores, r.class_ids)
    ts.append((time.perf_counter() - t) * 1000)
print(f"\nTracker   ByteTrack median {statistics.median(ts):.2f} ms/frame ({len(r)} detections)")
print(f"Memory    backend process RSS {proc.memory_info().rss / 2**20:.0f} MB "
      f"(+{(proc.memory_info().rss - rss0) / 2**20:.0f} MB for detector)")

from backend.vision.analyzer import SceneAnalyzer  # noqa: E402
from backend.vision.backends import LlamaCppBackend  # noqa: E402

s = Settings(vlm_autostart=False)
b = LlamaCppBackend(s.vlm_url)
if b.health()[0]:
    an = SceneAnalyzer(lambda: s, None, None, None, backend=b)
    lat = []
    for i in range(3):
        res = an.describe(img)
        lat.append(res.latency_ms / 1000)
        print(f"\nVLM run {i + 1}: {lat[-1]:.1f} s  {res.text!r}")
    srv = [p for p in psutil.process_iter(["name", "memory_info"]) if "llama-server" in (p.info["name"] or "")]
    if srv:
        print(f"llama-server RSS {srv[0].info['memory_info'].rss / 2**20:.0f} MB")
    print(f"VLM median {statistics.median(lat):.1f} s per description")
else:
    print("\nVLM       skipped - llama-server not running on", s.vlm_url)
