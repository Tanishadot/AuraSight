"""Object confirmation: temporal stability + class-aware thresholds + zoomed-crop
re-detection. Uses the real YOLO11n model on real photos where YOLO mistakes
architecture for appliances (see tests/fixtures/scenes/ATTRIBUTION.md)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.config.settings import Settings
from backend.detection.verify import CONFUSABLE, ObjectVerifier
from backend.detection.yolo_onnx import COCO_NAMES
from backend.tracking.bytetrack import ByteTracker, TrackOutput

SCENES = Path(__file__).parent / "fixtures" / "scenes"
TV = COCO_NAMES.index("tv")
CHAIR = COCO_NAMES.index("chair")


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def run_video(detector, image_name: str, frames: int = 15, settings: Settings | None = None):
    """Replay a photo as a short video with slight camera shake through
    detector -> ByteTrack -> verifier; return {label: [(status, mean)]}."""
    s = settings or Settings()
    img = cv2.imread(str(SCENES / image_name))
    h, w = img.shape[:2]
    clock = Clock()
    tracker = ByteTracker(track_thresh=s.confidence_threshold, max_lost_frames=10)

    def crop_detect(crop, size):
        r = detector.detect(crop, size, conf=0.05)
        return r.boxes, r.scores, r.class_ids

    v = ObjectVerifier(lambda: s, crop_detect, COCO_NAMES, clock=clock)
    final = {}
    for i in range(frames):
        clock.t += 0.2  # 5 detections per second
        m = np.float32([[1, 0, 3 * np.sin(i)], [0, 1, 2 * np.cos(i)]])
        frame = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        r = detector.detect(frame, 416, conf=0.1)
        tracks = tracker.update(r.boxes, r.scores, r.class_ids)
        states = v.update(tracks, frame)
        for t in tracks:
            st = states[t.track_id]
            final[t.track_id] = (COCO_NAMES[t.cls], st.status, float(np.mean(st.scores)))
    out: dict[str, list] = {}
    for label, status, mean in final.values():
        out.setdefault(label, []).append((status, round(mean, 2)))
    return out


# --- real false positives must never be confirmed ---------------------------------
@pytest.mark.parametrize("image,label", [
    ("fp_window_as_tv.jpg", "tv"),  # bright window in a dark room, YOLO tv 0.59
    ("fp_glass_transom_as_tv.jpg", "tv"),  # stained-glass panel above a door, tv 0.52
    ("fp_door_as_refrigerator.jpg", "refrigerator"),  # plank door, refrigerator 0.63
])
def test_architecture_false_positives_are_not_confirmed(detector, image, label):
    result = run_video(detector, image)
    assert all(status != "confirmed" for status, _ in result.get(label, [])), result


# --- genuine TVs/monitors must still be confirmed --------------------------------
@pytest.mark.parametrize("image", ["tv_living_room.jpg", "tv_living_room_1998.jpg", "multi_modern_living_room.jpg"])
def test_real_televisions_are_confirmed(detector, image):
    result = run_video(detector, image)
    assert any(status == "confirmed" for status, _ in result["tv"]), result


def test_ordinary_objects_confirm_without_extra_checks(detector):
    result = run_video(detector, "person_desk_laptop_cup.jpg")
    for label in ("person", "laptop", "chair"):
        assert any(status == "confirmed" for status, _ in result[label]), (label, result)


# --- unit-level rules ------------------------------------------------------------
def make(settings=None, crop_score=0.0):
    s = settings or Settings(min_track_frames=3)
    clock = Clock()
    calls = []

    def crop_detect(crop, size):
        calls.append(size)
        h, w = crop.shape[:2]
        if crop_score <= 0:
            return np.zeros((0, 4), np.float32), np.zeros(0), np.zeros(0, int)
        # pretend the object is re-detected filling the padded crop's centre
        return np.array([[0.2 * w, 0.2 * h, 0.8 * w, 0.8 * h]], np.float32), np.array([crop_score]), np.array([TV])

    return ObjectVerifier(lambda: s, crop_detect, COCO_NAMES, clock=clock), clock, calls


def track(tid, cls, score, box=(100, 100, 300, 260)):
    return TrackOutput(tid, np.array(box, np.float32), score, cls, 1, 1)


FRAME = np.zeros((480, 640, 3), np.uint8)


def feed(v, clock, tracks_per_frame, dt=0.2):
    states = {}
    for tracks in tracks_per_frame:
        clock.t += dt
        states = v.update(tracks, FRAME)
    return states


def test_needs_min_frames_and_mean_confidence():
    v, clock, _ = make()
    st = feed(v, clock, [[track(1, CHAIR, 0.9)]] * 2)
    assert st[1].status == "tentative"
    st = feed(v, clock, [[track(1, CHAIR, 0.9)]])
    assert st[1].status == "confirmed"
    v, clock, _ = make()
    st = feed(v, clock, [[track(2, CHAIR, 0.40)]] * 10)  # below the 0.45 base threshold
    assert st[2].status == "tentative" and "mean confidence" in st[2].reason


def test_flickering_track_is_not_confirmed():
    v, clock, _ = make()
    seq = [[track(1, CHAIR, 0.9)], [], [], [], [track(1, CHAIR, 0.9)], [], [], [], [track(1, CHAIR, 0.9)]]
    st = feed(v, clock, seq)
    assert st[1].status == "tentative" and st[1].reason == "flickering"


def test_confusable_class_requires_crop_verification():
    v, clock, calls = make(crop_score=0.0)
    st = feed(v, clock, [[track(1, TV, 0.58)]] * 6)
    assert st[1].status == "rejected" and calls  # crop check ran and failed
    v, clock, calls = make(crop_score=0.8)
    st = feed(v, clock, [[track(1, TV, 0.58)]] * 6)
    assert st[1].status == "confirmed" and "verified" in st[1].reason


def test_strong_confusable_detection_skips_crop_check():
    v, clock, calls = make(crop_score=0.0)
    st = feed(v, clock, [[track(1, TV, 0.85)]] * 4)
    assert st[1].status == "confirmed" and calls == []


def test_confirmed_confusable_object_is_demoted_when_confidence_drops():
    v, clock, _ = make(crop_score=0.8)
    st = feed(v, clock, [[track(1, TV, 0.58)]] * 6)
    assert st[1].status == "confirmed"
    st = feed(v, clock, [[track(1, TV, 0.30)]] * 10)
    assert st[1].status == "tentative"  # demoted, so it is no longer announced or used in answers


def test_confusable_set_covers_observed_failures():
    assert {"tv", "refrigerator"} <= CONFUSABLE and "person" not in CONFUSABLE
