"""Real YOLO11n ONNX inference on real images (no mocks)."""
from pathlib import Path

import numpy as np
import pytest

from backend.detection.yolo_onnx import YoloDetector, letterbox


def labels(det, res, min_score=0.5):
    return sorted(det.names[c] for c, s in zip(res.class_ids, res.scores) if s >= min_score)


@pytest.mark.parametrize("size", [320, 416, 640])
def test_detects_bus_and_people(detector, bus_image, size):
    res = detector.detect(bus_image, imgsz=size, conf=0.25)
    found = labels(detector, res)
    assert "bus" in found
    assert found.count("person") >= 3
    h, w = bus_image.shape[:2]
    assert (res.boxes[:, [0, 2]] >= 0).all() and (res.boxes[:, [0, 2]] <= w).all()
    assert (res.boxes[:, [1, 3]] >= 0).all() and (res.boxes[:, [1, 3]] <= h).all()
    assert res.inference_ms > 0


def test_bus_box_is_plausible(detector, bus_image):
    res = detector.detect(bus_image, imgsz=416, conf=0.5)
    bus = [b for b, c in zip(res.boxes, res.class_ids) if detector.names[c] == "bus"][0]
    h, w = bus_image.shape[:2]
    # the bus spans most of the image width in the upper/middle part
    assert (bus[2] - bus[0]) / w > 0.7
    assert bus[1] / h < 0.4


def test_zidane_people_and_tie(detector, zidane_image):
    res = detector.detect(zidane_image, imgsz=640, conf=0.25)
    found = labels(detector, res, 0.4)
    assert found.count("person") >= 2
    assert "tie" in found


def test_blank_image_has_no_confident_detections(detector):
    res = detector.detect(np.full((480, 640, 3), 127, np.uint8), imgsz=320, conf=0.4)
    assert len(res) == 0


def test_letterbox_pads_to_multiple_of_32():
    img = np.zeros((480, 640, 3), np.uint8)
    out, r, (px, py) = letterbox(img, 416)
    assert out.shape[0] % 32 == 0 and out.shape[1] % 32 == 0
    assert out.shape[1] == 416 and r == pytest.approx(0.65)


def test_missing_model_reports_error(tmp_path):
    d = YoloDetector(tmp_path / "nope.onnx")
    assert d.load() is False
    assert d.state == "error" and "not found" in d.error
    with pytest.raises(RuntimeError):
        d.detect(np.zeros((64, 64, 3), np.uint8))
