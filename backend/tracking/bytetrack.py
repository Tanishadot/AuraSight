"""ByteTrack multi-object tracker (Zhang et al., 2022), NumPy implementation.

Key ideas preserved from the paper / reference implementation:
  * Kalman filter on (cx, cy, aspect, height) with constant velocity.
  * Two-stage association: high-confidence detections first, then the
    remaining tracks are matched against *low*-confidence detections, which
    recovers occluded / blurred objects instead of dropping their IDs.
  * New tracks start unconfirmed and must be matched again before they are
    reported (unless they appear in the very first frame).
  * Lost tracks are kept for ``max_lost_frames`` so IDs survive short gaps.

AuraSight addition: detections are only associated with tracks of the same
class, which prevents ID swaps between e.g. a person and a chair.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------------- #
# Kalman filter (xyah), as used by SORT/DeepSORT/ByteTrack
# --------------------------------------------------------------------------- #
class KalmanFilterXYAH:
    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray):
        mean = np.r_[measurement, np.zeros(4)]
        h = measurement[3]
        std = [
            2 * self._std_weight_position * h, 2 * self._std_weight_position * h, 1e-2,
            2 * self._std_weight_position * h, 10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h, 1e-5, 10 * self._std_weight_velocity * h,
        ]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [
            self._std_weight_position * h, self._std_weight_position * h, 1e-2,
            self._std_weight_position * h, self._std_weight_velocity * h,
            self._std_weight_velocity * h, 1e-5, self._std_weight_velocity * h,
        ]
        motion_cov = np.diag(np.square(std))
        mean = self._motion_mat @ mean
        cov = self._motion_mat @ cov @ self._motion_mat.T + motion_cov
        return mean, cov

    def update(self, mean, cov, measurement):
        h = mean[3]
        std = [self._std_weight_position * h, self._std_weight_position * h, 1e-1,
               self._std_weight_position * h]
        innovation_cov = np.diag(np.square(std))
        proj_mean = self._update_mat @ mean
        proj_cov = self._update_mat @ cov @ self._update_mat.T + innovation_cov
        kalman_gain = np.linalg.solve(proj_cov, (cov @ self._update_mat.T).T).T
        innovation = measurement - proj_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_cov = cov - kalman_gain @ proj_cov @ kalman_gain.T
        return new_mean, new_cov


def xyxy_to_xyah(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h], dtype=np.float64)


def xyah_to_xyxy(m: np.ndarray) -> np.ndarray:
    cx, cy, a, h = m[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def linear_assignment(cost: np.ndarray, thresh: float):
    """Hungarian matching; pairs costing more than ``thresh`` are rejected."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    rows, cols = linear_sum_assignment(cost)
    matches, ur, uc = [], set(range(cost.shape[0])), set(range(cost.shape[1]))
    for r, c in zip(rows, cols):
        if cost[r, c] <= thresh:
            matches.append((r, c))
            ur.discard(r)
            uc.discard(c)
    return matches, sorted(ur), sorted(uc)


# --------------------------------------------------------------------------- #
# Tracks
# --------------------------------------------------------------------------- #
class TrackState(IntEnum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


@dataclass
class STrack:
    box: np.ndarray  # xyxy at creation
    score: float
    cls: int
    track_id: int = 0
    state: TrackState = TrackState.NEW
    is_activated: bool = False
    frame_id: int = 0
    start_frame: int = 0
    hits: int = 0
    mean: np.ndarray | None = None
    cov: np.ndarray | None = None
    last_box: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        self.last_box = np.asarray(self.box, dtype=np.float32)

    @property
    def xyxy(self) -> np.ndarray:
        if self.mean is None:
            return self.last_box
        return xyah_to_xyxy(self.mean)

    def predict(self, kf: KalmanFilterXYAH) -> None:
        mean = self.mean.copy()
        if self.state != TrackState.TRACKED:
            mean[7] = 0  # freeze height velocity while lost
        self.mean, self.cov = kf.predict(mean, self.cov)

    def activate(self, kf: KalmanFilterXYAH, frame_id: int, track_id: int) -> None:
        self.track_id = track_id
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(self.last_box))
        self.state = TrackState.TRACKED
        self.is_activated = frame_id == 1
        self.frame_id = self.start_frame = frame_id
        self.hits = 1

    def update(self, kf: KalmanFilterXYAH, det: "STrack", frame_id: int) -> None:
        self.mean, self.cov = kf.update(self.mean, self.cov, xyxy_to_xyah(det.last_box))
        self.last_box = det.last_box
        self.score = det.score
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.hits += 1

    def re_activate(self, kf: KalmanFilterXYAH, det: "STrack", frame_id: int) -> None:
        self.update(kf, det, frame_id)


@dataclass
class TrackOutput:
    track_id: int
    box: np.ndarray  # xyxy float32
    score: float
    cls: int
    hits: int
    age_frames: int


class ByteTracker:
    def __init__(self, track_thresh: float = 0.45, match_thresh: float = 0.8,
                 low_thresh: float = 0.1, max_lost_frames: int = 15):
        self.track_thresh = track_thresh
        self.det_thresh = track_thresh + 0.05  # threshold to start a new track
        self.match_thresh = match_thresh
        self.low_thresh = low_thresh
        self.max_lost_frames = max_lost_frames
        self.kf = KalmanFilterXYAH()
        self.reset()

    def reset(self) -> None:
        self.frame_id = 0
        self._next_id = 0
        self.tracked: list[STrack] = []
        self.lost: list[STrack] = []

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    @staticmethod
    def _cost(tracks: list[STrack], dets: list[STrack], fuse_score: bool) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)), dtype=np.float32)
        tb = np.stack([t.xyxy for t in tracks])
        db = np.stack([d.last_box for d in dets])
        iou = iou_matrix(tb, db)
        if fuse_score:
            iou = iou * np.array([d.score for d in dets])[None, :]
        cost = 1.0 - iou
        tc = np.array([t.cls for t in tracks])[:, None]
        dc = np.array([d.cls for d in dets])[None, :]
        cost[tc != dc] = 1.0 + 1e-3  # never match across classes
        return cost

    def update(self, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> list[TrackOutput]:
        self.frame_id += 1
        fid = self.frame_id
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        classes = np.asarray(classes).reshape(-1)

        hi = scores >= self.track_thresh
        lo = (scores >= self.low_thresh) & ~hi
        dets_hi = [STrack(b, float(s), int(c)) for b, s, c in zip(boxes[hi], scores[hi], classes[hi])]
        dets_lo = [STrack(b, float(s), int(c)) for b, s, c in zip(boxes[lo], scores[lo], classes[lo])]

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        confirmed = [t for t in self.tracked if t.is_activated]
        pool = confirmed + self.lost
        for t in pool:
            t.predict(self.kf)

        activated, refound, lost_now, removed = [], [], [], []

        # 1) high-confidence detections vs all confirmed + lost tracks
        matches, u_track, u_det = linear_assignment(self._cost(pool, dets_hi, True), self.match_thresh)
        for ti, di in matches:
            t, d = pool[ti], dets_hi[di]
            if t.state == TrackState.TRACKED:
                t.update(self.kf, d, fid)
                activated.append(t)
            else:
                t.re_activate(self.kf, d, fid)
                refound.append(t)

        # 2) remaining *tracked* tracks vs low-confidence detections
        r_tracked = [pool[i] for i in u_track if pool[i].state == TrackState.TRACKED]
        matches, u_track2, _ = linear_assignment(self._cost(r_tracked, dets_lo, False), 0.5)
        for ti, di in matches:
            r_tracked[ti].update(self.kf, dets_lo[di], fid)
            activated.append(r_tracked[ti])
        for ti in u_track2:
            t = r_tracked[ti]
            t.state = TrackState.LOST
            lost_now.append(t)

        # 3) unconfirmed tracks vs leftover high detections
        left_hi = [dets_hi[i] for i in u_det]
        matches, u_unconf, u_det = linear_assignment(self._cost(unconfirmed, left_hi, True), 0.7)
        for ti, di in matches:
            unconfirmed[ti].update(self.kf, left_hi[di], fid)
            activated.append(unconfirmed[ti])
        for ti in u_unconf:
            unconfirmed[ti].state = TrackState.REMOVED
            removed.append(unconfirmed[ti])

        # 4) start new tracks
        for di in u_det:
            d = left_hi[di]
            if d.score >= self.det_thresh:
                d.activate(self.kf, fid, self._new_id())
                activated.append(d)

        # 5) age out lost tracks
        still_lost = []
        for t in self.lost + lost_now:
            if t.state == TrackState.TRACKED:
                continue  # was re-found above
            if fid - t.frame_id > self.max_lost_frames:
                t.state = TrackState.REMOVED
                removed.append(t)
            else:
                still_lost.append(t)

        seen: set[int] = set()
        tracked = []
        for t in activated + refound:
            if id(t) not in seen and t.state == TrackState.TRACKED:
                seen.add(id(t))
                tracked.append(t)
        self.tracked = tracked
        self.lost = [t for t in still_lost if id(t) not in seen]

        return [
            TrackOutput(t.track_id, t.xyxy.copy(), t.score, t.cls, t.hits, fid - t.start_frame + 1)
            for t in self.tracked if t.is_activated
        ]
