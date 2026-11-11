"""Turns per-frame tracks into meaningful, de-duplicated scene events.

Events: ``entered``, ``left``, ``approaching``, ``obstacle``, ``scene_change``.

Anti-chatter measures:
  * An object must be tracked for ``min_track_frames`` before it "enters".
  * If a track is lost and a new ID of the same class re-appears nearby within
    a few seconds, it is treated as the *same* object (no second "entered").
  * Obstacle warnings fire on the *transition* into the near-centre zone, with
    a per-track cooldown, so a stationary chair is not announced repeatedly.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from ..spatial.positioning import OBSTACLE_CLASSES, SpatialInfo, spoken_label


@dataclass
class ObjectView:
    track_id: int
    label: str
    confidence: float
    box: list[float]  # normalised xyxy 0..1
    spatial: SpatialInfo
    hits: int
    first_seen: float
    approaching: bool = False
    status: str = "confirmed"  # confirmed | tentative | rejected (see detection/verify.py)
    status_reason: str = ""

    @property
    def confirmed(self) -> bool:
        return self.status == "confirmed"

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "spoken_label": spoken_label(self.label),
            "confidence": round(self.confidence, 3),
            "box": [round(v, 4) for v in self.box],
            "horizontal": self.spatial.horizontal,
            "direction": self.spatial.direction,
            "distance": self.spatial.distance,
            "in_path": self.spatial.in_path,
            "phrase": self.spatial.phrase(),
            "age_s": round(time.time() - self.first_seen, 1),
            "approaching": self.approaching,
            "status": self.status,
            "status_reason": self.status_reason,
        }


@dataclass
class SceneEvent:
    kind: str  # entered | left | approaching | obstacle | scene_change
    message: str
    priority: str  # info | warning
    label: str | None = None
    track_id: int | None = None
    spatial: SpatialInfo | None = None


@dataclass
class _Memory:
    track_id: int
    label: str
    first_seen: float
    last_seen: float
    hits: int = 0
    entered: bool = False
    last_alert: float = 0.0
    near_path: bool = False
    cx: float = 0.5
    spatial: SpatialInfo | None = None
    areas: deque = field(default_factory=lambda: deque(maxlen=40))


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


class SceneMonitor:
    REID_WINDOW_S = 4.0
    REID_MAX_DX = 0.2
    APPROACH_RATIO = 1.5
    APPROACH_WINDOW_S = 1.2
    ALERT_COOLDOWN_S = 10.0
    CHANGE_THRESHOLD = 28.0
    CHANGE_COOLDOWN_S = 10.0

    def __init__(self, settings_getter: Callable[[], object], clock: Callable[[], float] = time.monotonic):
        self._settings = settings_getter
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        self._mem: dict[int, _Memory] = {}
        self._recently_left: list[_Memory] = []
        self._baseline: np.ndarray | None = None
        self._change_streak = 0
        self._last_change = -1e9

    def first_seen(self, track_id: int) -> float | None:
        m = self._mem.get(track_id)
        return None if m is None else m.first_seen

    def update(self, objects: list[ObjectView], frame_bgr: np.ndarray | None = None) -> list[SceneEvent]:
        s = self._settings()
        now = self._clock()
        events: list[SceneEvent] = []
        seen_ids = set()

        for o in objects:
            seen_ids.add(o.track_id)
            m = self._mem.get(o.track_id)
            if m is None:
                m = _Memory(o.track_id, o.label, now, now)
                self._inherit_from_recent(m, o)
                self._mem[o.track_id] = m
            m.last_seen = now
            m.hits += 1
            m.cx = o.spatial.cx
            m.spatial = o.spatial
            m.areas.append((now, o.spatial.area))
            name = spoken_label(o.label)

            # approaching: apparent size grew quickly while in the walking path
            ratio = self._growth(m, now)
            o.approaching = ratio >= self.APPROACH_RATIO
            near_path = o.spatial.in_path and o.spatial.distance == "near"
            obstacle_like = o.label in OBSTACLE_CLASSES

            if m.entered and o.approaching and o.spatial.in_path and obstacle_like \
                    and o.spatial.distance != "far" and now - m.last_alert >= self.ALERT_COOLDOWN_S * 0.8:
                m.last_alert = now
                events.append(SceneEvent("approaching", f"{_cap(name)} approaching, {o.spatial.direction}.",
                                         "warning", o.label, o.track_id, o.spatial))
            elif m.entered and near_path and not m.near_path and obstacle_like \
                    and now - m.last_alert >= self.ALERT_COOLDOWN_S:
                m.last_alert = now
                events.append(SceneEvent("obstacle", f"Caution: {name} close, {o.spatial.direction}.",
                                         "warning", o.label, o.track_id, o.spatial))

            if not m.entered and m.hits >= s.min_track_frames:
                m.entered = True
                if near_path and obstacle_like:
                    m.last_alert = now
                    events.append(SceneEvent("obstacle", f"Caution: {name} close, {o.spatial.direction}.",
                                             "warning", o.label, o.track_id, o.spatial))
                else:
                    events.append(SceneEvent("entered", f"{_cap(name)} {o.spatial.phrase()}.", "info",
                                             o.label, o.track_id, o.spatial))
            m.near_path = near_path

        # departures
        for tid in list(self._mem):
            m = self._mem[tid]
            if tid in seen_ids:
                continue
            if now - m.last_seen >= s.leave_after_s:
                del self._mem[tid]
                if m.entered:
                    self._recently_left.append(m)
                    events.append(SceneEvent("left", f"{_cap(spoken_label(m.label))} no longer visible.",
                                             "info", m.label, tid, m.spatial))
        self._recently_left = [m for m in self._recently_left if now - m.last_seen < self.REID_WINDOW_S]

        if frame_bgr is not None:
            ev = self._scene_change(frame_bgr, now)
            if ev:
                events.append(ev)
        return events

    # -- helpers ------------------------------------------------------------------
    def _inherit_from_recent(self, m: _Memory, o: ObjectView) -> None:
        """Same class re-appearing close to where a lost track was -> same object."""
        best = None
        for old in self._recently_left + [x for x in self._mem.values() if x.label == o.label]:
            if old.label != o.label or old.track_id == m.track_id:
                continue
            if old.last_seen >= m.first_seen:  # still visible this frame -> a different object
                continue
            if abs(old.cx - o.spatial.cx) <= self.REID_MAX_DX and old.entered:
                best = old
                break
        if best is not None:
            m.entered = True
            m.first_seen = best.first_seen
            m.last_alert = best.last_alert
            m.near_path = best.near_path
            if best in self._recently_left:
                self._recently_left.remove(best)

    def _growth(self, m: _Memory, now: float) -> float:
        if len(m.areas) < 3:
            return 1.0
        cur = m.areas[-1][1]
        past = [a for (t, a) in m.areas if now - t >= self.APPROACH_WINDOW_S]
        if not past:
            return 1.0
        ref = past[-1]
        return cur / ref if ref > 1e-4 else 1.0

    def _scene_change(self, frame_bgr: np.ndarray, now: float) -> SceneEvent | None:
        small = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (32, 24),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        if self._baseline is None:
            self._baseline = small
            return None
        diff = float(np.abs(small - self._baseline).mean())
        # slow EMA absorbs gradual lighting drift
        self._baseline = 0.97 * self._baseline + 0.03 * small
        if diff > self.CHANGE_THRESHOLD:
            self._change_streak += 1
        else:
            self._change_streak = 0
        if self._change_streak >= 2 and now - self._last_change >= self.CHANGE_COOLDOWN_S:
            self._last_change = now
            self._change_streak = 0
            self._baseline = small
            return SceneEvent("scene_change", "The view has changed significantly.", "info")
        return None
