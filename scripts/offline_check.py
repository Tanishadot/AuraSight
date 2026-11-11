"""Prove that AuraSight's inference path makes no network connections.

Before importing anything from AuraSight, this script replaces socket
connect/DNS so that any attempt to reach a non-loopback address raises.
It then runs the real pipeline pieces:

  1. YOLO11n detection on bundled photos
  2. ByteTrack tracking across frames
  3. A real webcam frame (if a camera is available)
  4. Offline TTS synthesis to a WAV file
  5. VLM scene description through the local llama-server (if running)

Loopback (127.0.0.1 / ::1 / localhost) stays allowed because the local
llama-server and the UI live there. Run:  python scripts/offline_check.py
"""
from __future__ import annotations

import ipaddress
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BLOCKED: list[str] = []
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_getaddrinfo = socket.getaddrinfo


def _is_local(host) -> bool:
    if host in ("localhost", "", None):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _guard_connect(self, address):
    host = address[0] if isinstance(address, tuple) else address
    if not _is_local(host):
        BLOCKED.append(str(address))
        raise OSError(f"[offline_check] blocked outbound connection to {address}")
    return _real_connect(self, address)


def _guard_connect_ex(self, address):
    host = address[0] if isinstance(address, tuple) else address
    if not _is_local(host):
        BLOCKED.append(str(address))
        return 111
    return _real_connect_ex(self, address)


def _guard_getaddrinfo(host, *a, **k):
    if not _is_local(host):
        BLOCKED.append(f"dns:{host}")
        raise socket.gaierror(f"[offline_check] blocked DNS lookup for {host}")
    return _real_getaddrinfo(host, *a, **k)


socket.socket.connect = _guard_connect
socket.socket.connect_ex = _guard_connect_ex
socket.getaddrinfo = _guard_getaddrinfo
for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY"):
    os.environ.pop(var, None)

# Sanity check: the guard really blocks the internet.
try:
    socket.create_connection(("1.1.1.1", 443), timeout=2)
    print("FAIL guard did not block outbound traffic")
    sys.exit(2)
except OSError:
    BLOCKED.clear()

import cv2  # noqa: E402

from backend.config.settings import MODELS_DIR, Settings  # noqa: E402
from backend.detection.yolo_onnx import YoloDetector  # noqa: E402
from backend.tracking.bytetrack import ByteTracker  # noqa: E402

results: list[tuple[str, str, str]] = []


def record(name, ok, detail=""):
    results.append((name, "PASS" if ok is True else ("SKIP" if ok is None else "FAIL"), detail))


det = YoloDetector(MODELS_DIR / "yolo11n.onnx")
img = cv2.imread(str(ROOT / "tests" / "fixtures" / "bus.jpg"))
t = time.perf_counter()
r = det.detect(img, 416, conf=0.4)
labels = sorted(det.names[c] for c in r.class_ids)
record("YOLO11n detection", "bus" in labels and labels.count("person") >= 3,
       f"{labels} in {(time.perf_counter() - t) * 1000:.0f} ms")

tr = ByteTracker()
ids = []
for dx in range(0, 40, 4):
    m = cv2.warpAffine(img, __import__("numpy").float32([[1, 0, -dx], [0, 1, 0]]), img.shape[1::-1])
    rr = det.detect(m, 416, conf=0.1)
    ids.append(sorted(o.track_id for o in tr.update(rr.boxes, rr.scores, rr.class_ids) if o.score > 0.6))
# Objects may drop out as the image pans, but no *new* IDs may appear for the same objects.
record("ByteTrack persistent IDs", set(ids[-1]) <= set(ids[1]) and len(ids[-1]) >= 3,
       f"ids frame2={ids[1]} last={ids[-1]}")

try:
    from backend.camera.service import CameraService

    cam = CameraService()
    if cam.start(0, 640, 480, timeout=6):
        f = cam.wait_for_frame(0, 3)
        rr = det.detect(f.image, 416, conf=0.4)
        record("Webcam frame + detection", True,
               f"{f.image.shape[1]}x{f.image.shape[0]}, {len(rr)} objects, brightness {f.image.mean():.0f}")
        cam.stop()
    else:
        record("Webcam frame + detection", None, cam.error or "no camera")
except Exception as e:  # pragma: no cover
    record("Webcam frame + detection", False, str(e))

try:
    from backend.speech.engines import create_engine

    out: dict = {}

    def tts():
        eng = create_engine()
        p = Path(tempfile.gettempdir()) / "aurasight_offline_check.wav"
        eng.synthesize_to_file("Offline speech check.", p, 1.0, 0.8, None)
        out["engine"], out["size"] = eng.name, p.stat().st_size

    th = threading.Thread(target=tts)
    th.start()
    th.join(30)
    record("Offline TTS synthesis", out.get("size", 0) > 1000, f"{out.get('engine')} wrote {out.get('size')} bytes")
except Exception as e:
    record("Offline TTS synthesis", False, str(e))

try:
    from backend.vision.analyzer import SceneAnalyzer
    from backend.vision.backends import LlamaCppBackend

    s = Settings(vlm_autostart=False)
    backend = LlamaCppBackend(s.vlm_url)
    if backend.health()[0]:
        an = SceneAnalyzer(lambda: s, None, None, None, backend=backend)
        res = an.describe(img)
        record("Local VLM scene description", len(res.text) > 10, f"{res.latency_ms / 1000:.1f} s: {res.text!r}")
    else:
        record("Local VLM scene description", None, "llama-server not running on " + s.vlm_url)
except Exception as e:
    record("Local VLM scene description", False, str(e))

print("\nAuraSight offline check (all non-loopback network access blocked)\n")
for name, status, detail in results:
    print(f"  [{status}] {name}: {detail}")
print(f"\n  Outbound connection attempts blocked: {len(BLOCKED)} {BLOCKED[:5]}")
failed = any(s == "FAIL" for _, s, _ in results) or BLOCKED
sys.exit(1 if failed else 0)
