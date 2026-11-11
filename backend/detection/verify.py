"""Turns raw tracks into *confirmed* objects.

Why this exists
---------------
YOLO11n confuses some architecture with appliance/screen classes: a bright
window or glass panel in a darker room looks like a lit screen ("tv"), a large
flat door looks like a "refrigerator". On real photos these false positives
score 0.5-0.65, which overlaps genuine-but-difficult detections, so a single
global threshold cannot fix it. AuraSight therefore confirms an object only
when it passes three independent checks:

1. **Temporal stability** - tracked for ``min_track_frames`` and seen in most
   frames since it appeared (flicker never gets confirmed).
2. **Class-aware confidence** - the *mean* recent score must reach the base
   threshold; for classes known to be confused with architecture
   (``CONFUSABLE``) a higher threshold applies.
3. **Secondary verification** for confusable classes that are not
   overwhelmingly confident: the detector is re-run on a zoomed crop around
   the box. Measured on the fixture images, genuine TVs/monitors re-detect at
   0.39-0.93 while windows/doors mistaken for TV/refrigerator re-detect at 0.00.

Unconfirmed tracks are still reported (status ``"tentative"``) so the UI can
show them as unverified, but they are never announced, never trigger alerts,
and are not used to answer questions. Tracks that fail verification are
``"rejected"`` and re-checked later only if they become much more confident.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..tracking.bytetrack import TrackOutput, iou_matrix

# Classes that YOLO11n is known to hallucinate from windows, doors, cabinets,
# picture frames and plain walls. Measured on tests/fixtures/scenes.
CONFUSABLE = {"tv", "refrigerator", "oven", "microwave", "toaster", "sink", "toilet"}


@dataclass
class _State:
    first_frame: int
    scores: deque = field(default_factory=lambda: deque(maxlen=10))
    seen_frames: int = 0
    status: str = "tentative"  # tentative | confirmed | rejected
    verify_attempts: int = 0
    next_verify_at: float = 0.0
    verify_scores: list = field(default_factory=list)
    reason: str = "collecting evidence"

    @property
    def verify_score(self) -> float | None:
        return float(np.mean(self.verify_scores)) if self.verify_scores else None


class ObjectVerifier:
    # Two padded crops, averaged. Measured over jittered frames (see README):
    # real TVs/monitors 0.45-0.90, window->tv 0.27-0.40, door->refrigerator ~0.1.
    CROP_CONFIGS = ((0.3, 320), (0.6, 320))
    VERIFY_ACCEPT = 0.45
    VERIFY_FAST_ACCEPT = 0.60  # one pass is enough
    VERIFY_FAST_REJECT = 0.20
    SECOND_PASS_S = 0.4  # second look on a different frame
    VERIFY_RETRY_S = 3.0  # after a negative decision, look again later
    MAX_VERIFY_ROUNDS = 3
    MIN_PRESENCE = 0.6  # fraction of frames since first seen
    DEMOTE_MARGIN = 0.05  # hysteresis for confirmed confusable objects

    def __init__(self, settings_getter: Callable[[], object],
                 crop_detect: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray, np.ndarray]] | None,
                 names: list[str], clock: Callable[[], float] = time.monotonic):
        self._settings = settings_getter
        self._crop_detect = crop_detect
        self._names = names
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        self._states: dict[int, _State] = {}
        self._frame = 0
        self.verifications_run = 0

    def update(self, tracks: list[TrackOutput], frame_bgr: np.ndarray | None) -> dict[int, _State]:
        """Update evidence for this frame; returns state per visible track id."""
        s = self._settings()
        self._frame += 1
        now = self._clock()
        visible = {}
        for t in tracks:
            st = self._states.get(t.track_id)
            if st is None:
                st = self._states[t.track_id] = _State(first_frame=self._frame)
            st.scores.append(float(t.score))
            st.seen_frames += 1
            self._evaluate(t, st, s, frame_bgr, now)
            visible[t.track_id] = st
        # forget tracks gone for a while (tracker IDs are never reused)
        if self._frame % 50 == 0:
            live = {t.track_id for t in tracks}
            for tid in [k for k in self._states if k not in live]:
                if self._frame - self._states[tid].first_frame > 300:
                    del self._states[tid]
        return visible

    # ------------------------------------------------------------------
    def _evaluate(self, t: TrackOutput, st: _State, s, frame, now: float) -> None:
        label = self._names[t.cls]
        span = self._frame - st.first_frame + 1
        mean = float(np.mean(st.scores))
        base = s.confidence_threshold
        confusable = label in CONFUSABLE
        need = max(base, s.confusable_threshold) if confusable else base

        if st.status == "confirmed":
            # Confusable classes must keep earning it: a glass panel that briefly
            # looked like a TV is demoted when its confidence sinks again.
            if confusable and len(st.scores) >= 5 and mean < need - self.DEMOTE_MARGIN:
                st.status, st.reason = "tentative", f"confidence fell to {mean:.2f}"
                st.verify_scores, st.verify_attempts = [], 0
            return

        if st.seen_frames < s.min_track_frames:
            st.reason = f"seen {st.seen_frames}/{s.min_track_frames} frames"
            return
        if st.seen_frames / span < self.MIN_PRESENCE:
            st.reason = "flickering"
            return
        if mean < need:
            st.reason = f"mean confidence {mean:.2f} < {need:.2f}"
            return
        if not confusable or mean >= s.strong_confidence:
            st.status, st.reason = "confirmed", "stable and confident"
            return
        # Ambiguous confusable class -> zoomed-crop re-detection (secondary check).
        if frame is None or self._crop_detect is None or now < st.next_verify_at:
            return
        if st.verify_attempts >= 2 * self.MAX_VERIFY_ROUNDS:
            return  # gave up; stays rejected unless it becomes strongly confident
        st.verify_attempts += 1
        st.verify_scores.append(self.crop_verify(frame, t.box, t.cls))
        self.verifications_run += 1
        v, n = st.verify_score, len(st.verify_scores)
        if v >= self.VERIFY_FAST_ACCEPT or (n >= 2 and v >= self.VERIFY_ACCEPT):
            st.status, st.reason = "confirmed", f"verified on zoomed crops ({v:.2f})"
        elif v < self.VERIFY_FAST_REJECT or n >= 2:
            st.status, st.reason = "rejected", f"not confirmed on zoomed crops ({v:.2f})"
            st.verify_scores = []  # a later round starts fresh (camera may have moved)
            st.next_verify_at = now + self.VERIFY_RETRY_S
        else:
            st.reason = f"checking ({v:.2f})"
            st.next_verify_at = now + self.SECOND_PASS_S

    def crop_verify(self, frame: np.ndarray, box: np.ndarray, cls: int) -> float:
        """Re-detect the object inside padded crops; mean of the best same-class
        score overlapping the original box, over CROP_CONFIGS."""
        return float(np.mean([self._crop_score(frame, box, cls, pad, size)
                              for pad, size in self.CROP_CONFIGS]))

    def _crop_score(self, frame: np.ndarray, box: np.ndarray, cls: int, pad: float, size: int) -> float:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box]
        bw, bh = x2 - x1, y2 - y1
        cx1 = int(max(0, x1 - pad * bw)); cy1 = int(max(0, y1 - pad * bh))
        cx2 = int(min(w, x2 + pad * bw)); cy2 = int(min(h, y2 + pad * bh))
        if cx2 - cx1 < 16 or cy2 - cy1 < 16:
            return 0.0
        boxes, scores, classes = self._crop_detect(frame[cy1:cy2, cx1:cx2], size)
        if len(scores) == 0:
            return 0.0
        target = np.array([[x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]], np.float32)
        ious = iou_matrix(target, np.asarray(boxes, np.float32))[0]
        ok = (np.asarray(classes) == cls) & (ious > 0.4)
        return float(np.max(np.asarray(scores)[ok])) if ok.any() else 0.0
