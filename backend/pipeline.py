"""AuraSight core: wires camera -> detector -> tracker -> spatial -> events -> speech.

Threads:
  camera      grabs frames into a single-slot buffer (never queues)
  detection   pulls the newest frame at ``detection_fps`` (frame skipping),
              runs YOLO + ByteTrack + scene monitor, publishes events
  speech      one utterance at a time from a bounded priority queue
  vlm-*       short-lived, at most one scene analysis in flight
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from .assistant.announcer import Announcer, summarize_objects
from .assistant.qa import Assistant
from .assistant.scene_monitor import ObjectView, SceneMonitor
from .camera.service import CameraService, CameraState, list_cameras
from .config.settings import DATA_DIR, MODELS_DIR, TOOLS_DIR, Settings, SettingsStore
from .detection.verify import ObjectVerifier
from .detection.yolo_onnx import YoloDetector
from .events import EventBus
from .spatial.positioning import analyze, spoken_label
from .speech.service import Priority, SpeechService
from .tracking.bytetrack import ByteTracker
from .vision.analyzer import AnalyzerBusy, SceneAnalyzer
from .vision.backends import VLMUnavailable

log = logging.getLogger("aurasight.pipeline")


class CameraNotRunning(RuntimeError):
    pass


class Pipeline:
    def __init__(
        self,
        store: SettingsStore,
        bus: EventBus,
        camera: CameraService | None = None,
        detector: YoloDetector | None = None,
        speech: SpeechService | None = None,
        analyzer: SceneAnalyzer | None = None,
    ):
        self.store = store
        self.bus = bus
        s = store.current
        if camera is None:
            import os

            from .camera.service import replay_open_fn
            src = os.environ.get("AURASIGHT_CAMERA_SOURCE")
            camera = CameraService(open_fn=replay_open_fn(src)) if src else CameraService()
        self.camera = camera
        self.camera._on_state = self._on_camera_state
        self.detector = detector or YoloDetector(MODELS_DIR / s.detector_model, s.detector_threads)
        self.speech = speech or SpeechService(lambda: self.store.current, on_event=self._on_speech_event)
        if speech is not None:
            speech._on_event = self._on_speech_event
        self.analyzer = analyzer or SceneAnalyzer(lambda: self.store.current, MODELS_DIR, TOOLS_DIR,
                                                  DATA_DIR, on_status=self._on_vlm_status)
        self.analyzer._on_status = self._on_vlm_status
        self.tracker = ByteTracker(track_thresh=s.confidence_threshold,
                                   max_lost_frames=self._lost_frames(s))
        self.verifier = ObjectVerifier(lambda: self.store.current, self._crop_detect, self.detector.names)
        self.monitor = SceneMonitor(lambda: self.store.current)
        self.announcer = Announcer(lambda: self.store.current, self.speech)
        self.assistant = Assistant(self.analyzer, self._frame_image, self.objects)

        self._objects: list[ObjectView] = []
        self._obj_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.det_fps = 0.0
        self.frames_processed = 0
        self.frames_skipped = 0
        self._last_auto_describe = 0.0
        self._last_status_pub = 0.0
        store.on_change(self._on_settings)

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> None:
        self._stop.clear()
        self.speech.start()
        s = self.store.current
        if s.vlm_backend != "none":
            self.analyzer.start_background()
        self._thread = threading.Thread(target=self._loop, name="detection", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        self.camera.stop()
        if self._thread:
            self._thread.join(3)
        self.speech.shutdown()
        self.analyzer.shutdown()

    # -- status ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        return {
            "camera": self.camera.status(),
            "detector": self.detector.status(),
            "vlm": self.analyzer.status(),
            "speech": self.speech.status(),
            "pipeline": {
                "detection_fps": round(self.det_fps, 1),
                "frames_processed": self.frames_processed,
                "frames_skipped": self.frames_skipped,
                "objects": len(self._objects),
            },
        }

    def publish_status(self) -> None:
        self.bus.publish("status", {"status": self.status()})

    # -- camera control ------------------------------------------------------------
    def start_camera(self, index: int | None = None) -> bool:
        s = self.store.current
        if index is not None and index != s.camera_index:
            self.store.update({"camera_index": index})
            s = self.store.current
        self._reset_scene()
        ok = self.camera.start(s.camera_index, s.camera_width, s.camera_height)
        if ok and self.detector.state == "not_loaded":
            threading.Thread(target=self._load_detector, daemon=True).start()
        return ok

    def _load_detector(self) -> None:
        self.detector.load()
        self.publish_status()
        if self.detector.state == "error":
            self.bus.activity("error", self.detector.error or "Detector failed", "error")

    def pause_camera(self) -> None:
        self.camera.pause()

    def resume_camera(self) -> None:
        self.camera.resume()

    def stop_camera(self) -> None:
        self.camera.stop()
        self._reset_scene()
        self.bus.publish("detections", {"objects": [], "frame_id": 0, "width": 0, "height": 0})

    def cameras(self) -> list[dict]:
        return list_cameras(skip_index=self.camera.index if self.camera.is_active else None)

    def _reset_scene(self) -> None:
        self.tracker.reset()
        self.verifier.reset()
        self.monitor.reset()
        self.announcer.reset()
        with self._obj_lock:
            self._objects = []

    def _on_camera_state(self, state: CameraState, error: str | None) -> None:
        msgs = {
            CameraState.RUNNING: ("Camera on. Video is processed only on this device.", "info"),
            CameraState.PAUSED: ("Camera paused.", "info"),
            CameraState.STOPPED: ("Camera off.", "info"),
            CameraState.ERROR: (f"Camera problem: {error}", "error"),
        }
        if state in msgs:
            msg, prio = msgs[state]
            self.bus.activity("camera", msg, prio, state=state.value)
            if state in (CameraState.RUNNING, CameraState.STOPPED, CameraState.ERROR):
                spoken = {CameraState.RUNNING: "Camera on.", CameraState.STOPPED: "Camera off.",
                          CameraState.ERROR: "Camera problem. " + (error or "")}[state]
                self.speech.say(spoken, Priority.USER, kind="system", max_age=5)
        self.publish_status()

    # -- frames / objects -----------------------------------------------------------
    def _frame_image(self) -> np.ndarray | None:
        if self.camera.state not in (CameraState.RUNNING, CameraState.PAUSED):
            return None
        f = self.camera.latest()
        return None if f is None else f.image

    def objects(self) -> list[ObjectView]:
        """Confirmed objects only - the only ones used for speech, alerts and answers."""
        with self._obj_lock:
            return [o for o in self._objects if o.confirmed]

    def all_objects(self) -> list[ObjectView]:
        """Every tracked object including unverified ones (for the UI)."""
        with self._obj_lock:
            return list(self._objects)

    def _crop_detect(self, crop: np.ndarray, size: int):
        r = self.detector.detect(crop, size, conf=0.05)
        return r.boxes, r.scores, r.class_ids

    @staticmethod
    def _lost_frames(s: Settings) -> int:
        return max(5, int(round(2.0 * s.detection_fps)))  # ~2 s of occlusion tolerance

    def _on_settings(self, s: Settings, changed: set[str]) -> None:
        self.tracker.track_thresh = s.confidence_threshold
        self.tracker.det_thresh = s.confidence_threshold + 0.05
        self.tracker.max_lost_frames = self._lost_frames(s)
        if changed & {"vlm_backend", "vlm_url", "vlm_model", "vlm_mmproj", "vlm_autostart", "vlm_threads"}:
            self.analyzer.reconfigure()
        if "speech_enabled" in changed and not s.speech_enabled:
            self.speech.stop_all()
        self.bus.publish("settings", {"settings": s.model_dump()})

    # -- detection loop --------------------------------------------------------------
    def _loop(self) -> None:
        last_id = 0
        last_det = 0.0
        fps_t, fps_n = time.monotonic(), 0
        while not self._stop.is_set():
            now = time.monotonic()
            if now - self._last_status_pub > 2.0:
                self._last_status_pub = now
                self.publish_status()
            if self.camera.state != CameraState.RUNNING:
                time.sleep(0.1)
                continue
            frame = self.camera.wait_for_frame(last_id, timeout=0.5)
            if frame is None:
                continue
            s = self.store.current
            interval = 1.0 / s.detection_fps
            if self.analyzer.busy:
                # The VLM is CPU-bound; detection competing for the same cores made a
                # description take ~2.5x longer. Drop to 1 fps until it finishes.
                interval = max(interval, 1.0)
            # 20 % tolerance so camera-frame jitter doesn't halve the effective rate
            if time.monotonic() - last_det < interval * 0.8:
                self.frames_skipped += frame.id - last_id
                last_id = frame.id
                time.sleep(min(0.02, interval / 4))
                continue
            self.frames_skipped += max(0, frame.id - last_id - 1)
            last_id = frame.id
            last_det = time.monotonic()
            if self.detector.state != "ready":
                if self.detector.state in ("not_loaded",):
                    self._load_detector()
                if self.detector.state != "ready":
                    time.sleep(1.0)
                    continue
            try:
                self._process(frame.image, frame.id, s)
            except Exception as e:  # keep the loop alive no matter what
                log.exception("detection step failed")
                self.bus.activity("error", f"Detection error: {e}", "error")
                time.sleep(0.5)
                continue
            fps_n += 1
            if time.monotonic() - fps_t >= 1.0:
                self.det_fps = fps_n / (time.monotonic() - fps_t)
                fps_t, fps_n = time.monotonic(), 0

    def _process(self, img: np.ndarray, frame_id: int, s: Settings) -> list:
        h, w = img.shape[:2]
        dets = self.detector.detect(img, s.inference_size, conf=0.1)
        tracks = self.tracker.update(dets.boxes, dets.scores, dets.class_ids)
        states = self.verifier.update(tracks, img)
        now_wall = time.time()
        objects: list[ObjectView] = []
        for t in tracks:
            st = states[t.track_id]
            label = self.detector.names[t.cls]
            box = t.box.astype(float)
            box[[0, 2]] = box[[0, 2]].clip(0, w)
            box[[1, 3]] = box[[1, 3]].clip(0, h)
            sp = analyze(box, w, h, label)
            first = self.monitor.first_seen(t.track_id)
            age = 0.0 if first is None else time.monotonic() - first
            objects.append(ObjectView(t.track_id, label, t.score,
                                      [box[0] / w, box[1] / h, box[2] / w, box[3] / h], sp, t.hits,
                                      now_wall - age, status=st.status, status_reason=st.reason))
        # Only confirmed objects can create events, announcements and alerts.
        events = self.monitor.update([o for o in objects if o.confirmed], img)
        with self._obj_lock:
            self._objects = objects
        self.frames_processed += 1
        self.bus.publish("detections", {
            "frame_id": frame_id, "width": w, "height": h,
            "inference_ms": round(dets.inference_ms, 1),
            "fps": round(self.det_fps, 1),
            "objects": [o.to_dict() for o in objects if o.status != "rejected"],
        })
        for e in events:
            self.bus.activity(e.kind, e.message, e.priority, label=e.label, track_id=e.track_id)
        self.announcer.handle(events)
        self._maybe_auto_describe(s, any(e.kind == "scene_change" for e in events))
        return events

    def _maybe_auto_describe(self, s: Settings, scene_changed: bool) -> None:
        if not s.auto_describe or not self.analyzer.available or self.analyzer.busy:
            return
        now = time.monotonic()
        due = now - self._last_auto_describe >= s.auto_describe_interval_s
        if due or (scene_changed and now - self._last_auto_describe >= 10):
            self._last_auto_describe = now
            threading.Thread(target=self._safe_describe, kwargs={"speak": True, "auto": True},
                             name="vlm-auto", daemon=True).start()

    def _safe_describe(self, speak: bool, auto: bool) -> None:
        try:
            self.describe_scene(speak=speak, auto=auto)
        except Exception:
            pass

    # -- user-facing actions ---------------------------------------------------------
    def describe_scene(self, speak: bool = True, auto: bool = False) -> dict[str, Any]:
        img = self._frame_image()
        if img is None:
            raise CameraNotRunning("The camera is not running.")
        img = img.copy()  # the exact frame analysed, immune to the capture thread
        objects = self.objects()  # confirmed only
        det_summary = summarize_objects(objects)
        prio = Priority.NORMAL if auto else Priority.USER
        labels = sorted({spoken_label(o.label) for o in objects})
        spoken: list[str] = []
        self.bus.publish("scene_progress", {"stage": "analysing", "auto": auto})

        def on_sentence(sentence: str) -> None:
            # Speak each sentence the moment it is complete instead of waiting for
            # the whole answer: first words arrive seconds earlier.
            spoken.append(sentence)
            self.bus.publish("scene_progress", {"stage": "streaming", "text": " ".join(spoken), "auto": auto})
            if speak:
                self.speech.say(sentence, prio, kind="description")

        first_ms = None
        try:
            res = self.analyzer.describe(img, hint_labels=labels, on_sentence=on_sentence)
            narrative, source, latency = res.text, "vlm", res.latency_ms
            first_ms = res.first_sentence_ms
            if not narrative:
                narrative = "I'm not sure what I'm looking at."
                if speak:
                    self.speech.say(narrative, prio, kind="description")
            grounded = f"The object detector confirms {det_summary}." if det_summary else ""
            text = f"{narrative} {grounded}".strip()
            if speak and grounded:
                self.speech.say(grounded, prio, kind="description")
        except AnalyzerBusy:
            self.bus.publish("scene_progress", {"stage": "idle", "auto": auto})
            raise
        except VLMUnavailable as e:
            source, latency = "detector", 0.0
            text = (f"The scene model is unavailable. The object detector sees {det_summary}."
                    if det_summary else
                    "The scene model is unavailable, and the object detector doesn't see any "
                    "objects it recognises.")
            self.bus.activity("vlm", f"Scene model unavailable: {e}", "warning")
            if speak:
                self.speech.say(text, prio, kind="description")
        payload = {"description": text, "source": source, "latency_ms": round(latency),
                   "first_sentence_ms": None if first_ms is None else round(first_ms),
                   "auto": auto, "objects": det_summary}
        self.bus.publish("scene", payload)
        self.bus.publish("scene_progress", {"stage": "idle", "auto": auto})
        self.bus.activity("description", text, "info", source=source)
        return payload

    def ask(self, question: str, speak: bool = True) -> dict[str, Any]:
        self.bus.activity("question", question, "info")
        ans = self.assistant.answer(question)
        d = ans.to_dict()
        self.bus.publish("answer", d)
        self.bus.activity("answer", ans.answer, "info", sources=ans.sources)
        if speak:
            self.speech.say(ans.answer, Priority.USER, kind="answer")
        return d

    def summary(self, speak: bool = True) -> str:
        if self._frame_image() is None:
            text = "The camera is off."
        else:
            det = summarize_objects(self.objects(), 6)
            text = f"I can see {det}." if det else "I don't see any objects I recognise right now."
        if speak:
            self.speech.say(text, Priority.USER, kind="summary")
        self.bus.activity("summary", text, "info")
        return text

    # -- callbacks ----------------------------------------------------------------------
    def _on_speech_event(self, type_: str, payload: dict) -> None:
        self.bus.publish("speech", payload)

    def _on_vlm_status(self, status: dict) -> None:
        self.bus.publish("vlm_status", {"vlm": status})
