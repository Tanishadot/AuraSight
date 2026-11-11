import os
import time

import numpy as np
import pytest

from backend.camera.service import CameraService, CameraState, opencv_open
from tests.conftest import ClosedCapture, FakeCapture, wait_for


def test_start_acquire_pause_resume_stop(bus_image):
    states = []
    cam = CameraService(open_fn=lambda i, w, h: FakeCapture(bus_image), on_state=lambda s, e: states.append(s))
    assert cam.start(0)
    assert cam.state == CameraState.RUNNING and cam.is_active
    f1 = cam.wait_for_frame(0, 2)
    f2 = cam.wait_for_frame(f1.id, 2)
    assert f2.id > f1.id and f2.image.shape == bus_image.shape
    cam.pause()
    assert cam.state == CameraState.PAUSED
    n = cam.latest().id
    time.sleep(0.2)
    assert cam.latest().id == n  # no new frames while paused
    assert cam.wait_for_frame(n, 0.2) is None
    cam.resume()
    assert cam.wait_for_frame(n, 2) is not None
    cam.stop()
    assert cam.state == CameraState.STOPPED and cam.latest() is None and not cam.is_active
    assert states[:2] == [CameraState.STARTING, CameraState.RUNNING]
    assert CameraState.PAUSED in states and states[-1] == CameraState.STOPPED


def test_open_failure_reports_error():
    cam = CameraService(open_fn=lambda i, w, h: ClosedCapture())
    assert cam.start(3, timeout=2) is False
    assert cam.state == CameraState.ERROR
    assert "Could not open camera 3" in cam.error


def test_open_exception_reports_error():
    def boom(i, w, h):
        raise OSError("driver crashed")

    cam = CameraService(open_fn=boom)
    assert cam.start(0, timeout=2) is False
    assert "driver crashed" in cam.error


def test_camera_that_stops_delivering_frames(bus_image):
    cap = FakeCapture(bus_image, fps=200, fail_after=5)
    cam = CameraService(open_fn=lambda i, w, h: cap)
    assert cam.start(0)
    assert wait_for(lambda: cam.state == CameraState.ERROR, 5)
    assert "stopped delivering" in cam.error
    assert wait_for(lambda: cap.released, 2)


def test_single_slot_buffer_never_queues(bus_image):
    cam = CameraService(open_fn=lambda i, w, h: FakeCapture(bus_image, fps=200))
    cam.start(0)
    time.sleep(0.5)
    f = cam.latest()
    assert f.id > 20  # many frames produced, only one retained
    cam.stop()


@pytest.mark.skipif(os.environ.get("AURASIGHT_SKIP_HARDWARE") == "1", reason="hardware tests disabled")
def test_real_webcam_if_present():
    """Uses the machine's real webcam (index 0). Skips if none can be opened."""
    probe = opencv_open(0, 640, 480)
    has = probe.isOpened()
    probe.release()
    if not has:
        pytest.skip("no webcam available")
    cam = CameraService()
    try:
        assert cam.start(0, 640, 480), cam.error
        f = cam.wait_for_frame(0, 3)
        assert f is not None and f.image.ndim == 3 and f.image.dtype == np.uint8
        assert cam.status()["resolution"][0] >= 320
    finally:
        cam.stop()
