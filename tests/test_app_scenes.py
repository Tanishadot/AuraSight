"""Whole-application tests on real photos replayed as a camera (ReplayCapture):
real YOLO + ByteTrack + verifier + scene monitor + announcer + speech queue."""
from __future__ import annotations

import time
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
                "I can see a woman sitting at a desk. A laptop is on your right. Behind her there are flowers.")
            # the person (woman) and laptop the model already named are not repeated;
            # only detector objects it did not mention are added
            tail = d["description"].split("flowers.", 1)[1]
            assert "laptop" not in tail and "person" not in tail
            if tail.strip():
                assert tail.strip().startswith("The object detector also found ")
            assert d["speech"] == "spoken"
            # the detector's objects were given to the model as context
            prompt = vlm.requests[-1]["body"]["messages"][0]["content"][1]["text"]
            assert prompt.startswith("An object detector found: ") and "laptop" in prompt
            # every sentence on screen is spoken, in order, word for word
            def said():  # obstacle alerts may cut in (and the sentence is then re-read)
                return [t for t in engine.spoken if not t.startswith("Caution")]
            assert wait_for(lambda: len(said()) >= len(d["sentences"]), 10), engine.spoken
            assert said()[: len(d["sentences"])] == d["sentences"]
            assert " ".join(d["sentences"]) == d["description"]
            assert d["first_sentence_ms"] is not None and d["first_sentence_ms"] <= d["latency_ms"]
            client.post("/api/camera/stop")
    finally:
        vlm.close()


def _describe_ready(tmp_path, reply: str, engine_delay: float = 0.01):
    vlm = StandIn(reply=reply)
    p, engine = build(tmp_path, "person_desk_laptop_cup.jpg", vlm.url)
    engine.duration = engine_delay
    # The desk photo triggers "Caution: person close" alerts at random moments;
    # their interplay with descriptions is covered in test_speech.py.
    p.store.update({"obstacle_alerts": False})
    return vlm, p, engine


LONG = ("A woman is sitting at a wooden desk in a bright room. A laptop is open on your right. "
        "A white cup stands next to the laptop.")


def test_replay_reads_the_same_description_again(tmp_path):
    vlm, p, engine = _describe_ready(tmp_path, LONG)
    try:
        with TestClient(create_app(p)) as client:
            assert client.post("/api/scene/replay").status_code == 404  # nothing yet
            client.post("/api/camera/start")
            assert wait_for(lambda: p.analyzer.available, 10)
            d = client.post("/api/scene/describe").json()
            assert wait_for(lambda: engine.spoken[-len(d["sentences"]):] == d["sentences"], 10)
            engine.spoken.clear()
            r = client.post("/api/scene/replay").json()
            assert r["speech"] == "spoken" and r["description"] == d["description"]
            assert wait_for(lambda: engine.spoken == d["sentences"], 10), engine.spoken
            client.post("/api/camera/stop")
    finally:
        vlm.close()


def test_stop_interrupts_description_immediately(tmp_path):
    vlm, p, engine = _describe_ready(tmp_path, LONG, engine_delay=3.0)
    try:
        with TestClient(create_app(p)) as client:
            client.post("/api/camera/start")
            assert wait_for(lambda: p.analyzer.available, 10)
            d = client.post("/api/scene/describe").json()
            assert wait_for(lambda: p.speech.current_kind == "description", 5)
            client.post("/api/speech/stop")
            assert wait_for(lambda: not p.speech.is_speaking, 1.0)  # well under one 3 s sentence
            assert engine.interrupts >= 1
            assert len(engine.spoken) < len(d["sentences"])  # the rest was dropped
            client.post("/api/camera/stop")
    finally:
        vlm.close()


def test_new_description_replaces_one_still_playing(tmp_path):
    vlm, p, engine = _describe_ready(tmp_path, LONG, engine_delay=1.0)
    try:
        with TestClient(create_app(p)) as client:
            client.post("/api/camera/start")
            assert wait_for(lambda: p.analyzer.available, 10)
            client.post("/api/scene/describe")
            assert wait_for(lambda: p.speech.current_kind == "description", 5)
            before = engine.interrupts
            d2 = client.post("/api/scene/describe").json()
            assert engine.interrupts > before  # the first one was cut off, not talked over
            # only the second description's sentences remain queued: never two at once
            assert wait_for(lambda: not p.speech.is_speaking and p.speech.current_kind is None, 15)
            assert engine.spoken[-len(d2["sentences"]):] == d2["sentences"], (engine.spoken, engine.interrupted)
            assert engine.max_concurrent == 1
            client.post("/api/camera/stop")
    finally:
        vlm.close()


def test_speak_out_loud_off_shows_but_does_not_speak(tmp_path):
    vlm, p, engine = _describe_ready(tmp_path, LONG)
    try:
        with TestClient(create_app(p)) as client:
            client.put("/api/settings", json={"speech_enabled": False})
            client.post("/api/camera/start")
            assert wait_for(lambda: p.analyzer.available, 10)
            engine.spoken.clear()
            d = client.post("/api/scene/describe").json()
            assert "woman" in d["description"] and len(d["sentences"]) >= 3
            assert d["speech"] == "muted"
            assert client.post("/api/scene/replay").json()["speech"] == "muted"
            time.sleep(0.5)
            assert not any(s in engine.spoken for s in d["sentences"]), engine.spoken
            client.post("/api/camera/stop")
    finally:
        vlm.close()


def test_tts_failure_is_reported_not_fatal(tmp_path):
    vlm, p, engine = _describe_ready(tmp_path, LONG)
    events: list[dict] = []
    try:
        with TestClient(create_app(p)) as client:
            original = p.speech._on_event
            p.speech._on_event = lambda t, d: (events.append(d), original(t, d))
            client.post("/api/camera/start")
            assert wait_for(lambda: p.analyzer.available, 10)
            engine.fail = True
            d = client.post("/api/scene/describe").json()
            assert d["description"]  # still shown
            assert wait_for(lambda: any(e.get("state") == "error" and e.get("kind") == "description"
                                        for e in events), 10), events
            assert wait_for(lambda: p.speech.state == "ready", 5)  # the worker survived
            engine.fail = False
            assert client.post("/api/scene/replay").json()["speech"] == "spoken"  # recovers
            client.post("/api/camera/stop")
    finally:
        vlm.close()
