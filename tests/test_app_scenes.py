"""Whole-application tests on real photos replayed as a camera (ReplayCapture):
real YOLO + ByteTrack + verifier + scene monitor + announcer + speech queue."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.camera.service import CameraService, ReplayCapture
from backend.config.settings import MODELS_DIR, SettingsStore
from backend.detection.yolo_onnx import YoloDetector
from backend.events import EventBus
from backend.main import create_app
from backend.pipeline import Pipeline
from backend.speech.service import SpeechService
from backend.vision.analyzer import SceneAnalyzer
from backend.vision.backends import LlamaCppBackend
from tests.conftest import RecordingEngine, wait_for
from tests.test_vlm import StandIn

SCENES = Path(__file__).parent / "fixtures" / "scenes"


def build(tmp_path, image: str, vlm_url: str = "http://127.0.0.1:9"):
    if not (MODELS_DIR / "yolo11n.onnx").exists():
        pytest.skip("detector model missing")
    store = SettingsStore(tmp_path / "s.json")
    store.update({"vlm_autostart": False, "detection_fps": 8})
    engine = RecordingEngine(0.01)
    speech = SpeechService(lambda: store.current, engine_factory=lambda: engine)
    analyzer = SceneAnalyzer(lambda: store.current, None, None, None, backend=LlamaCppBackend(vlm_url))
    camera = CameraService(open_fn=lambda i, w, h: ReplayCapture(str(SCENES / image), fps=15))
    p = Pipeline(store, EventBus(), camera=camera, detector=YoloDetector(MODELS_DIR / "yolo11n.onnx"),
                 speech=speech, analyzer=analyzer)
    return p, engine


def test_window_misdetected_as_tv_is_never_announced(tmp_path):
    p, engine = build(tmp_path, "fp_window_as_tv.jpg")
    with TestClient(create_app(p)) as client:
        assert client.post("/api/camera/start").status_code == 200
        assert wait_for(lambda: p.frames_processed >= 20, 30)
        confirmed = client.get("/api/objects").json()
        assert all(o["label"] != "tv" for o in confirmed)
        activity = [e["message"] for e in p.bus.history(100, "activity")]
        assert not any("TV" in m for m in activity), activity
        assert not any("TV" in t for t in engine.spoken), engine.spoken
        a = client.post("/api/ask", json={"question": "Is there a TV?", "speak": False}).json()
        assert a["answer"].startswith("I don't see a TV or monitor")
        client.post("/api/camera/stop")


def test_real_tv_is_announced_and_answered(tmp_path):
    p, engine = build(tmp_path, "tv_living_room.jpg")
    with TestClient(create_app(p)) as client:
        client.post("/api/camera/start")
        assert wait_for(lambda: any(o.label == "tv" for o in p.objects()), 30)
        assert wait_for(lambda: any("TV or monitor" in t for t in engine.spoken), 10), engine.spoken
        a = client.post("/api/ask", json={"question": "Where is the TV?", "speak": False}).json()
        assert a["grounded"] and a["answer"].startswith("I can see a TV or monitor")
        client.post("/api/camera/stop")


def test_describe_streams_sentences_and_appends_detector_facts(tmp_path):
    vlm = StandIn(reply="The picture shows a woman sitting at a desk. A laptop is on the right side of the image. "
                        "Behind her there are flowers.")
    try:
        p, engine = build(tmp_path, "person_desk_laptop_cup.jpg", vlm.url)
        with TestClient(create_app(p)) as client:
            client.post("/api/camera/start")
            assert wait_for(lambda: {"person", "laptop"} <= {o.label for o in p.objects()}, 30)
            assert wait_for(lambda: p.analyzer.available, 10)
            engine.spoken.clear()
            d = client.post("/api/scene/describe").json()
            assert d["source"] == "vlm"
            assert d["description"].startswith(
                "I can see a woman sitting at a desk. A laptop is on your right. Behind her there are flowers. "
                "The object detector confirms ")
            assert "a person ahead" in d["description"] and "laptop" in d["description"]
            # the detector's objects were given to the model as context
            prompt = vlm.requests[-1]["body"]["messages"][0]["content"][1]["text"]
            assert prompt.startswith("An object detector found: ") and "laptop" in prompt
            # each sentence was spoken as it streamed in, then the grounded summary
            assert wait_for(lambda: len(engine.spoken) >= 4, 10), engine.spoken
            assert engine.spoken[0] == "I can see a woman sitting at a desk."
            assert engine.spoken[3].startswith("The object detector confirms")
            assert d["first_sentence_ms"] is not None and d["first_sentence_ms"] <= d["latency_ms"]
            client.post("/api/camera/stop")
    finally:
        vlm.close()
