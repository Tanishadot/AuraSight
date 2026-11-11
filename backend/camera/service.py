"""Webcam capture running in its own thread.

The capture thread only ever keeps the *latest* frame (a single-slot buffer),
so slow consumers skip frames instead of building up a backlog. Frames never
leave this process except as the low-rate MJPEG preview served to the local UI.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

log = logging.getLogger("aurasight.camera")


class Capture(Protocol):
    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def release(self) -> None: ...


class CameraState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class Frame:
    id: int
    image: np.ndarray  # BGR, HxWx3
    ts: float


def opencv_open(index: int, width: int, height: int) -> Capture:
    import cv2

    # DirectShow opens much faster than MSMF on Windows and supports set().
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class ReplayCapture:
    """Plays still images (a file or a folder, cycling) or a video file as if it
    were a webcam. For testing and demos: set AURASIGHT_CAMERA_SOURCE."""

    def __init__(self, source: str, fps: float = 15.0, seconds_per_image: float = 8.0):
        import cv2

        p = Path(source)
        self.fps, self.spi = fps, seconds_per_image
        self._video = None
        self._images: list[np.ndarray] = []
        if p.is_dir():
            files = sorted(f for f in p.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
            self._images = [img for img in (cv2.imread(str(f)) for f in files) if img is not None]
        elif p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            img = cv2.imread(str(p))
            self._images = [img] if img is not None else []
        else:
            self._video = cv2.VideoCapture(str(p))
        self._t0 = time.monotonic()
        self._opened = bool(self._images) or (self._video is not None and self._video.isOpened())

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        time.sleep(1.0 / self.fps)
        if self._video is not None:
            ok, img = self._video.read()
            if not ok:  # loop
                self._video.set(1, 0)  # CAP_PROP_POS_FRAMES
                ok, img = self._video.read()
            return ok, img
        i = int((time.monotonic() - self._t0) / self.spi) % len(self._images)
        return True, self._images[i].copy()

    def release(self) -> None:
        self._opened = False
        if self._video is not None:
            self._video.release()


def replay_open_fn(source: str):
    secs = float(os.environ.get("AURASIGHT_SOURCE_SECONDS", "8"))
    return lambda index, w, h: ReplayCapture(source, seconds_per_image=secs)


class CameraService:
    MAX_CONSECUTIVE_FAILURES = 25

    def __init__(
        self,
        open_fn: Callable[[int, int, int], Capture] = opencv_open,
        on_state: Callable[[CameraState, str | None], None] | None = None,
    ):
        self._open_fn = open_fn
        self._on_state = on_state
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._next_id = 1
        self._state = CameraState.STOPPED
        self._error: str | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self.index: int | None = None
        self.resolution: tuple[int, int] | None = None
        self.measured_fps: float = 0.0

    # -- state ----------------------------------------------------------------
    @property
    def state(self) -> CameraState:
        return self._state

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def is_active(self) -> bool:
        """True while the physical camera is open (running or paused)."""
        return self._state in (CameraState.STARTING, CameraState.RUNNING, CameraState.PAUSED)

    def _set_state(self, state: CameraState, error: str | None = None) -> None:
        self._state = state
        self._error = error
        with self._cond:
            self._cond.notify_all()
        if self._on_state:
            try:
                self._on_state(state, error)
            except Exception:
                log.exception("camera state listener failed")

    def status(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "error": self._error,
            "index": self.index,
            "resolution": self.resolution,
            "fps": round(self.measured_fps, 1),
            "active": self.is_active,
        }

    # -- control ----------------------------------------------------------------
    def start(self, index: int, width: int = 640, height: int = 480, timeout: float = 8.0) -> bool:
        """Open the camera; blocks until the first frame arrives or failure."""
        self.stop()
        self._stop.clear()
        self._paused.clear()
        self.index = index
        with self._cond:
            self._frame = None
        self._set_state(CameraState.STARTING)
        self._thread = threading.Thread(
            target=self._run, args=(index, width, height), name="camera", daemon=True
        )
        self._thread.start()
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._state == CameraState.STARTING and time.monotonic() < deadline:
                self._cond.wait(0.1)
        if self._state == CameraState.STARTING:
            self._stop.set()
            self._set_state(CameraState.ERROR, f"Camera {index} did not deliver a frame within {timeout:.0f}s")
        return self._state == CameraState.RUNNING

    def pause(self) -> None:
        if self._state == CameraState.RUNNING:
            self._paused.set()
            self._set_state(CameraState.PAUSED)

    def resume(self) -> None:
        if self._state == CameraState.PAUSED:
            self._paused.clear()
            self._set_state(CameraState.RUNNING)

    def stop(self) -> None:
        t = self._thread
        if t and t.is_alive():
            self._stop.set()
            t.join(timeout=3.0)
        self._thread = None
        with self._cond:
            self._frame = None
        self.measured_fps = 0.0
        if self._state != CameraState.STOPPED:
            self._set_state(CameraState.STOPPED)

    # -- frames ------------------------------------------------------------------
    def latest(self) -> Frame | None:
        with self._cond:
            return self._frame

    def wait_for_frame(self, after_id: int, timeout: float = 1.0) -> Frame | None:
        """Return the newest frame with id > after_id, or None on timeout."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                f = self._frame
                if f is not None and f.id > after_id and self._state == CameraState.RUNNING:
                    return f
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)

    # -- capture thread ----------------------------------------------------------
    def _run(self, index: int, width: int, height: int) -> None:
        cap: Capture | None = None
        try:
            try:
                cap = self._open_fn(index, width, height)
            except Exception as e:  # driver-level failures
                self._set_state(CameraState.ERROR, f"Could not open camera {index}: {e}")
                return
            if cap is None or not cap.isOpened():
                self._set_state(
                    CameraState.ERROR,
                    f"Could not open camera {index}. It may be in use by another app, "
                    "disconnected, or blocked in the OS privacy settings.",
                )
                return
            failures = 0
            t_window, n_window = time.monotonic(), 0
            while not self._stop.is_set():
                if self._paused.is_set():
                    time.sleep(0.05)
                    continue
                ok, img = cap.read()
                if not ok or img is None:
                    failures += 1
                    if failures >= self.MAX_CONSECUTIVE_FAILURES:
                        self._set_state(CameraState.ERROR, "The camera stopped delivering frames.")
                        return
                    time.sleep(0.02)
                    continue
                failures = 0
                if self._paused.is_set() or self._stop.is_set():
                    continue  # read raced with pause/stop: discard
                now = time.monotonic()
                with self._cond:
                    self._frame = Frame(self._next_id, img, time.time())
                    self._next_id += 1
                    self._cond.notify_all()
                if self._state == CameraState.STARTING:
                    self.resolution = (int(img.shape[1]), int(img.shape[0]))
                    self._set_state(CameraState.RUNNING)
                n_window += 1
                if now - t_window >= 1.0:
                    self.measured_fps = n_window / (now - t_window)
                    t_window, n_window = now, 0
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass


def list_cameras(max_probe: int = 4, skip_index: int | None = None) -> list[dict[str, Any]]:
    """Enumerate cameras. Uses DirectShow names on Windows when available,
    otherwise probes indices with OpenCV (skipping the one we already hold)."""
    src = os.environ.get("AURASIGHT_CAMERA_SOURCE")
    if src:
        return [{"index": 0, "name": f"Replay: {Path(src).name}"}]
    if sys.platform == "win32":
        try:
            from pygrabber.dshow_graph import FilterGraph  # type: ignore

            names = FilterGraph().get_input_devices()
            return [{"index": i, "name": n} for i, n in enumerate(names)]
        except Exception:
            pass
    import cv2

    found = []
    for i in range(max_probe):
        if i == skip_index:
            found.append({"index": i, "name": f"Camera {i} (in use)"})
            continue
        cap = opencv_open(i, 320, 240)
        try:
            if cap.isOpened():
                found.append({"index": i, "name": f"Camera {i}"})
        finally:
            cap.release()
    return found
