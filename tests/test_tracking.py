import cv2
import numpy as np

from backend.tracking.bytetrack import ByteTracker, iou_matrix


def step(tr, boxes, scores=None, classes=None):
    boxes = np.array(boxes, np.float32).reshape(-1, 4)
    scores = np.array(scores if scores is not None else [0.9] * len(boxes), np.float32)
    classes = np.array(classes if classes is not None else [0] * len(boxes))
    return {o.track_id: o for o in tr.update(boxes, scores, classes)}


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 1 / 3, 0.0], atol=1e-5)


def test_ids_persist_for_moving_objects():
    tr = ByteTracker()
    ids_seen = set()
    for t in range(40):
        out = step(tr, [[10 + 3 * t, 50, 60 + 3 * t, 150], [400 - 2 * t, 60, 460 - 2 * t, 200]])
        ids_seen |= set(out)
    assert ids_seen == {1, 2}


def test_new_track_requires_confirmation_after_first_frame():
    tr = ByteTracker()
    assert step(tr, [[0, 0, 50, 50]]) .keys() == {1}  # first frame -> immediately active
    out = step(tr, [[0, 0, 50, 50], [200, 200, 260, 260]])
    assert set(out) == {1}  # the new object is not reported yet
    out = step(tr, [[0, 0, 50, 50], [200, 200, 260, 260]])
    assert set(out) == {1, 2}


def test_id_survives_short_occlusion():
    tr = ByteTracker(max_lost_frames=10)
    for t in range(5):
        step(tr, [[100 + 2 * t, 100, 160 + 2 * t, 220]])
    for _ in range(5):  # hidden
        assert step(tr, []) == {}
    out = step(tr, [[120, 100, 180, 220]])
    assert set(out) == {1}


def test_id_is_dropped_after_long_absence():
    tr = ByteTracker(max_lost_frames=3)
    for _ in range(3):
        step(tr, [[100, 100, 160, 220]])
    for _ in range(6):
        step(tr, [])
    step(tr, [[100, 100, 160, 220]])
    out = step(tr, [[100, 100, 160, 220]])
    assert set(out) == {2}


def test_low_confidence_detection_keeps_track_alive():
    """ByteTrack's second association stage: a blurred (low score) detection
    still continues the existing track instead of losing it."""
    tr = ByteTracker(track_thresh=0.5)
    for _ in range(3):
        step(tr, [[100, 100, 160, 220]], [0.9])
    out = step(tr, [[102, 100, 162, 220]], [0.2])
    assert set(out) == {1}


def test_no_matching_across_classes():
    tr = ByteTracker()
    step(tr, [[100, 100, 200, 200]], classes=[0])
    step(tr, [[100, 100, 200, 200]], classes=[0])
    out = step(tr, [[100, 100, 200, 200]], classes=[56])  # same place, different class
    assert 1 not in out


def test_tracks_on_real_detections(detector, bus_image):
    """Real detector + tracker on a slowly panning image: IDs stay stable."""
    tr = ByteTracker(track_thresh=0.45)
    h, w = bus_image.shape[:2]
    first_ids = None
    for t in range(12):
        m = np.float32([[1, 0, -4 * t], [0, 1, 0]])
        img = cv2.warpAffine(bus_image, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        r = detector.detect(img, 416, conf=0.1)
        out = tr.update(r.boxes, r.scores, r.class_ids)
        ids = {o.track_id for o in out if o.score > 0.6}
        if first_ids is None:
            first_ids = ids
    assert first_ids and first_ids <= {o.track_id for o in out}
