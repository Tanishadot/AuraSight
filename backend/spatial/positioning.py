"""Approximate, *relative* spatial descriptions from bounding-box geometry.

AuraSight has no depth sensor and no camera calibration, so it never states
metres. "Near / middle distance / far" is a heuristic based on how much of the
frame an object fills, normalised by a rough per-class size prior (a bottle
that fills half the frame is much closer than a person filling half the frame).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Typical apparent-size prior per class (1.0 = large, like a standing person).
_LARGE = {"person", "couch", "bed", "dining table", "refrigerator", "car", "bus", "truck",
          "train", "motorcycle", "bicycle", "horse", "cow", "elephant", "bear", "giraffe",
          "zebra", "bench", "airplane", "boat"}
_MEDIUM = {"chair": 0.55, "toilet": 0.5, "tv": 0.5, "dog": 0.45, "suitcase": 0.5,
           "potted plant": 0.45, "oven": 0.55, "sink": 0.45, "microwave": 0.4,
           "backpack": 0.4, "laptop": 0.4, "umbrella": 0.5, "fire hydrant": 0.45,
           "stop sign": 0.4, "sheep": 0.6, "cat": 0.35, "skateboard": 0.35,
           "surfboard": 0.6, "teddy bear": 0.35, "handbag": 0.35, "keyboard": 0.3,
           "clock": 0.3, "vase": 0.3, "traffic light": 0.35, "parking meter": 0.45}
_SMALL_DEFAULT = 0.22  # bottle, cup, phone, remote, mouse, book, cutlery, fruit, ...

# Classes that can physically block a walking path at floor level.
OBSTACLE_CLASSES = {
    "person", "chair", "couch", "bed", "dining table", "bench", "bicycle", "motorcycle", "car",
    "bus", "truck", "dog", "cat", "suitcase", "potted plant", "toilet", "refrigerator",
    "fire hydrant", "backpack", "skateboard", "stop sign", "parking meter", "horse", "cow",
}


def size_prior(label: str) -> float:
    if label in _LARGE:
        return 1.0
    return _MEDIUM.get(label, _SMALL_DEFAULT)


@dataclass(frozen=True)
class SpatialInfo:
    horizontal: str  # "left" | "centre" | "right"
    direction: str  # "on your left" | "slightly left" | "ahead" | "slightly right" | "on your right"
    distance: str  # "near" | "middle" | "far"
    in_path: bool  # inside the central walking corridor
    cx: float  # normalised centre x
    area: float  # normalised box area

    @property
    def distance_phrase(self) -> str:
        return {"near": "near", "middle": "middle distance", "far": "far away"}[self.distance]

    def phrase(self) -> str:
        return f"{self.direction}, {self.distance_phrase}"


def horizontal_zone(cx: float) -> tuple[str, str]:
    if cx < 0.2:
        return "left", "on your left"
    if cx < 0.4:
        return "left", "slightly left"
    if cx <= 0.6:
        return "centre", "ahead"
    if cx <= 0.8:
        return "right", "slightly right"
    return "right", "on your right"


def distance_bucket(box: np.ndarray, frame_w: int, frame_h: int, label: str) -> str:
    x1, y1, x2, y2 = [float(v) for v in box]
    hf = (y2 - y1) / frame_h
    wf = (x2 - x1) / frame_w
    s = max(hf, wf)
    norm = s / size_prior(label)
    touches_bottom = y2 >= 0.97 * frame_h
    # Pinhole reasoning for a ~45 deg vertical FOV webcam: a 1.7 m person fills
    # the frame height at ~2 m and 40 % of it at ~5 m. Priors scale this per class.
    if norm >= 0.85 or (touches_bottom and s >= 0.45):
        return "near"
    if norm >= 0.4:
        return "middle"
    return "far"


def analyze(box: np.ndarray, frame_w: int, frame_h: int, label: str) -> SpatialInfo:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = ((x1 + x2) / 2) / frame_w
    horizontal, direction = horizontal_zone(cx)
    # Walking corridor: the central 40 % of the image width.
    overlap = max(0.0, min(x2 / frame_w, 0.7) - max(x1 / frame_w, 0.3))
    in_path = overlap >= 0.2 or 0.3 <= cx <= 0.7
    area = ((x2 - x1) * (y2 - y1)) / (frame_w * frame_h)
    return SpatialInfo(horizontal, direction, distance_bucket(box, frame_w, frame_h, label),
                       in_path, cx, area)


def plural(label: str, n: int) -> str:
    if n == 1:
        return label
    irregular = {"person": "people", "mouse": "mice", "knife": "knives", "bus": "buses",
                 "couch": "couches", "wine glass": "wine glasses", "skis": "skis",
                 "sheep": "sheep", "toothbrush": "toothbrushes", "sandwich": "sandwiches",
                 "dining table": "tables", "tv": "TVs", "cell phone": "phones",
                 "hair drier": "hair driers", "glass": "glasses"}
    return irregular.get(label, label + "s")


def spoken_label(label: str) -> str:
    """Friendlier names for speech."""
    return {"dining table": "table", "cell phone": "phone", "tv": "TV or monitor",
            "potted plant": "plant", "couch": "sofa"}.get(label, label)


def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"
