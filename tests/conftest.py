from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.config.settings import MODELS_DIR, SettingsStore

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def bus_image() -> np.ndarray:
    img = cv2.imread(str(FIXTURES / "bus.jpg"))
    assert img is not None
    return img


@pytest.fixture(scope="session")
def zidane_image() -> np.ndarray:
    img = cv2.imread(str(FIXTURES / "zidane.jpg"))
    assert img is not None
    return img


@pytest.fixture(scope="session")
def detector():
    from backend.detection.yolo_onnx import YoloDetector

    path = MODELS_DIR / "yolo11n.onnx"
    if not path.exists():
        pytest.skip("yolo11n.onnx not downloaded (run scripts/setup_models.py)")
    d = YoloDetector(path)
    assert d.load()
    return d


@pytest.fixture
def store(tmp_path) -> SettingsStore:
    return SettingsStore(tmp_path / "settings.json")


class FakeCapture:
    """Simulated webcam: replays an image, panning it slightly each frame."""

    def __init__(self, image: np.ndarray, fps: float = 30.0, fail_after: int | None = None):
        self.image = image
        self.fps = fps
        self.n = 0
        self.fail_after = fail_after
        self.released = False

    def isOpened(self) -> bool:
        return not self.released

    def read(self):
        time.sleep(1.0 / self.fps)
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            return False, None
        dx = (self.n % 20) - 10  # small horizontal jitter
        m = np.float32([[1, 0, dx], [0, 1, 0]])
        h, w = self.image.shape[:2]
        return True, cv2.warpAffine(self.image, m, (w, h), borderMode=cv2.BORDER_REPLICATE)

    def release(self) -> None:
        self.released = True


class ClosedCapture:
    def isOpened(self):
        return False

    def read(self):
        return False, None

    def release(self):
        pass


class RecordingEngine:
    """Fake TTS engine that records what would have been spoken."""

    name = "recording"

    def __init__(self, duration: float = 0.05):
        self.spoken: list[str] = []
        self.interrupted: list[str] = []
        self.duration = duration
        self._stop = threading.Event()

    def speak(self, text, rate, volume, voice_id) -> bool:
        self._stop.clear()
        if self._stop.wait(self.duration):
            self.interrupted.append(text)
            return False
        self.spoken.append(text)
        return True

    def interrupt(self):
        self._stop.set()

    def synthesize_to_file(self, text, path, rate, volume, voice_id):
        Path(path).write_bytes(b"RIFF")

    def voices(self):
        return [{"id": "fake", "name": "Fake voice"}]

    def close(self):
        pass


def wait_for(cond, timeout=10.0, interval=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(interval)
    return False
