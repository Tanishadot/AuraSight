"""End-to-end API tests: real FastAPI app, real pipeline threads, real YOLO
detector and tracker. The camera is a simulated capture replaying a real
photo, speech uses a recording engine, and the VLM points at an unreachable
port so the *unavailable* fallback path is exercised."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from backend.camera.service import CameraService
from backend.config.settings import MODELS_DIR, SettingsStore
from backend.detection.yolo_onnx import YoloDetector
from backend.events import EventBus
from backend.main import create_app
from backend.pipeline import Pipeline
from backend.speech.service import SpeechService
from backend.vision.analyzer import SceneAnalyzer
from backend.vision.backends import LlamaCppBackend
from tests.conftest import ClosedCapture, FakeCapture, RecordingEngine, wait_for


@pytest.fixture
def env(tmp_path, bus_image):
    if not (MODELS_DIR / "yolo11n.onnx").exists():
        pytest.skip("detector model missing")
    store = SettingsStore(tmp_path / "s.json")
    store.update({"vlm_autostart": False, "detection_fps": 10, "min_track_frames": 2})
    bus = EventBus()
    engine = RecordingEngine(0.01)
    speech = SpeechService(lambda: store.current, engine_factory=lambda: engine)
    analyzer = SceneAnalyzer(lambda: store.current, None, None, None,
                             backend=LlamaCppBackend("http://127.0.0.1:9"))
    state = {"fail": False}

    def open_fn(i, w, h):
        return ClosedCapture() if state["fail"] else FakeCapture(bus_image, fps=15)

    camera = CameraService(open_fn=open_fn)
    p = Pipeline(store, bus, camera=camera, detector=YoloDetector(MODELS_DIR / "yolo11n.onnx"),
                 speech=speech, analyzer=analyzer)
    app = create_app(p)
    with TestClient(app) as client:
        yield client, p, engine, state


def test_health_status_settings(env):
    client, p, _, _ = env
    assert client.get("/api/health").json() == {"ok": True}
    st = client.get("/api/status").json()
    assert set(st) >= {"camera", "detector", "vlm", "speech", "pipeline"}
    assert st["camera"]["state"] == "stopped"
    assert client.put("/api/settings", json={"inference_size": 320}).json()["inference_size"] == 320
    assert client.put("/api/settings", json={"inference_size": 333}).status_code == 422
    assert client.put("/api/settings", json={"bogus": 1}).status_code == 422
    hw = client.get("/api/hardware").json()
    assert hw["ram_total_gb"] > 0 and "recommendations" in hw


def test_live_detection_tracking_and_websocket(env):
    client, p, engine, _ = env
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and "settings" in hello
        r = client.post("/api/camera/start", json={"index": 0})
        assert r.status_code == 200 and r.json()["state"] == "running"
        dets, activities = [], []
        end = time.monotonic() + 30
        while time.monotonic() < end and len(dets) < 8:
            ev = ws.receive_json()
            if ev["type"] == "detections" and ev["objects"]:
                dets.append(ev)
            elif ev["type"] == "activity":
                activities.append(ev)
        assert len(dets) >= 8, "no detection events received"
        labels = {o["label"] for o in dets[-1]["objects"]}
        assert {"bus", "person"} <= labels
        # persistent track ids across frames
        bus_ids = [[o["track_id"] for o in d["objects"] if o["label"] == "bus"] for d in dets[2:]]
        assert len({tuple(x) for x in bus_ids}) == 1
        o = dets[-1]["objects"][0]
        assert {"direction", "distance", "phrase", "confidence", "box"} <= set(o)
        assert any(a["kind"] == "camera" for a in activities)

    assert wait_for(lambda: any(e["kind"] == "entered" for e in p.bus.history(100, "activity")), 5)
    assert wait_for(lambda: any(t.startswith("New:") for t in engine.spoken), 5)
    # stationary objects are not announced repeatedly
    time.sleep(1.5)
    assert sum(t.startswith("New:") for t in engine.spoken) <= 2

    objs = client.get("/api/objects").json()
    assert any(o["label"] == "bus" for o in objs)

    # MJPEG preview delivers JPEG frames
    with client.stream("GET", "/api/stream.mjpg") as resp:
        assert resp.headers["content-type"].startswith("multipart/x-mixed-replace")
        chunk = b""
        for part in resp.iter_bytes():
            chunk += part
            if b"\xff\xd9" in chunk:
                break
        assert b"\xff\xd8" in chunk and b"Content-Type: image/jpeg" in chunk
        p.camera.stop()  # ends the stream generator

    assert client.post("/api/camera/stop").json()["state"] == "stopped"


def test_describe_falls_back_when_vlm_unavailable(env):
    client, p, engine, _ = env
    assert client.post("/api/scene/describe").status_code == 409  # camera off
    client.post("/api/camera/start")
    assert wait_for(lambda: len(p.objects()) >= 3, 20)
    r = client.post("/api/scene/describe").json()
    assert r["source"] == "detector"
    assert r["description"].startswith("The scene model is unavailable. The object detector sees")
    assert "bus" in r["description"]
    assert wait_for(lambda: any("scene model is unavailable" in t for t in engine.spoken), 5)
    client.post("/api/camera/stop")


def test_ask_grounded_answer(env):
    client, p, engine, _ = env
    client.post("/api/camera/start")
    assert wait_for(lambda: any(o.label == "bus" for o in p.objects()), 20)
    r = client.post("/api/ask", json={"question": "Is there a bus?"}).json()
    assert r["grounded"] and r["answer"].startswith("Yes. I can see a bus")
    r = client.post("/api/ask", json={"question": "How many people are here?"}).json()
    assert r["answer"].startswith("I can see") and "people" in r["answer"]
    r = client.post("/api/ask", json={"question": "Where is my water bottle?"}).json()
    assert r["answer"].startswith("I don't see a bottle")
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    client.post("/api/camera/stop")


def test_camera_failure_is_reported(env):
    client, p, engine, state = env
    state["fail"] = True
    r = client.post("/api/camera/start")
    assert r.status_code == 503 and "Could not open camera" in r.json()["detail"]
    assert client.get("/api/status").json()["camera"]["state"] == "error"
    assert any(e["kind"] == "camera" and e["priority"] == "error" for e in p.bus.history(50, "activity"))


def test_pause_resume_and_speech_controls(env):
    client, p, engine, _ = env
    client.post("/api/camera/start")
    assert client.post("/api/camera/pause").json()["state"] == "paused"
    assert client.post("/api/camera/resume").json()["state"] == "running"
    assert client.post("/api/speech/say", json={"text": "hello there"}).json()["queued"]
    assert wait_for(lambda: "hello there" in engine.spoken, 3)
    assert client.post("/api/speech/stop").json()["stopped"]
    assert client.get("/api/speech/voices").json() == [{"id": "fake", "name": "Fake voice"}]
    client.post("/api/camera/stop")
