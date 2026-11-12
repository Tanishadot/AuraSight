/// ByteTrack multi-object tracker (Zhang et al., 2022). Port of
/// backend/tracking/bytetrack.py; golden tests check the two agree.
///
///  * Kalman filter on (cx, cy, aspect, height) with constant velocity.
///  * Two-stage association: high-confidence detections first, then the
///    remaining tracks against *low*-confidence detections, which recovers
///    occluded or blurred objects instead of dropping their IDs.
///  * New tracks start unconfirmed and must be matched again before they are
///    reported (unless they appear in the very first frame).
///  * Lost tracks are kept for [ByteTracker.maxLostFrames] so IDs survive gaps.
///  * AuraSight addition: detections only match tracks of the same class.
library;

import '../geometry.dart';
import 'assignment.dart';
import 'kalman.dart';

/// One detector output in pixel coordinates.
class Detection {
  final Box box;
  final double score;
  final int cls;
  const Detection(this.box, this.score, this.cls);
}

enum TrackState { newTrack, tracked, lost, removed }

class _STrack {
  Box lastBox;
  double score;
  final int cls;
  int trackId = 0;
  TrackState state = TrackState.newTrack;
  bool isActivated = false;
  int frameId = 0;
  int startFrame = 0;
  int hits = 0;
  List<double>? mean;
  List<List<double>>? cov;

  _STrack(Box box, this.score, this.cls) : lastBox = box.toF32();

  Box get xyxy {
    final m = mean;
    if (m == null) return lastBox;
    final w = m[2] * m[3];
    return Box(f32(m[0] - w / 2), f32(m[1] - m[3] / 2), f32(m[0] + w / 2), f32(m[1] + m[3] / 2));
  }

  void predict(KalmanFilterXYAH kf) {
    final m = List<double>.of(mean!);
    if (state != TrackState.tracked) m[7] = 0; // freeze height velocity while lost
    (mean, cov) = kf.predict(m, cov!);
  }

  void activate(KalmanFilterXYAH kf, int fid, int id) {
    trackId = id;
    (mean, cov) = kf.initiate(xyxyToXyah(lastBox.toList()));
    state = TrackState.tracked;
    isActivated = fid == 1;
    frameId = startFrame = fid;
    hits = 1;
  }

  void update(KalmanFilterXYAH kf, _STrack det, int fid) {
    (mean, cov) = kf.update(mean!, cov!, xyxyToXyah(det.lastBox.toList()));
    lastBox = det.lastBox;
    score = det.score;
    state = TrackState.tracked;
    isActivated = true;
    frameId = fid;
    hits += 1;
  }
}

/// A confirmed track, reported once per frame.
class TrackOutput {
  final int trackId;
  final Box box;
  final double score;
  final int cls;
  final int hits;
  final int ageFrames;
  const TrackOutput(this.trackId, this.box, this.score, this.cls, this.hits, this.ageFrames);
}

class ByteTracker {
  double trackThresh;
  double detThresh;
  final double matchThresh;
  final double lowThresh;
  int maxLostFrames;
  final _kf = KalmanFilterXYAH();

  int frameId = 0;
  int _nextId = 0;
  List<_STrack> _tracked = [];
  List<_STrack> _lost = [];

  ByteTracker({this.trackThresh = 0.45, this.matchThresh = 0.8, this.lowThresh = 0.1, this.maxLostFrames = 15})
      : detThresh = trackThresh + 0.05;

  void reset() {
    frameId = 0;
    _nextId = 0;
    _tracked = [];
    _lost = [];
  }

  static List<List<double>> _cost(List<_STrack> tracks, List<_STrack> dets, bool fuseScore) {
    return [
      for (final t in tracks)
        [
          for (final d in dets)
            if (t.cls != d.cls)
              1.0 + 1e-3 // never match across classes
            else
              1.0 - f32(iou(t.xyxy, d.lastBox)) * (fuseScore ? d.score : 1.0)
        ]
    ];
  }

  List<TrackOutput> update(List<Detection> detections) {
    frameId += 1;
    final fid = frameId;
    final dets = [for (final d in detections) _STrack(d.box, f32(d.score), d.cls)];
    final detsHi = [for (final d in dets) if (d.score >= trackThresh) d];
    final detsLo = [for (final d in dets) if (d.score >= lowThresh && d.score < trackThresh) d];

    final unconfirmed = [for (final t in _tracked) if (!t.isActivated) t];
    final confirmed = [for (final t in _tracked) if (t.isActivated) t];
    final pool = [...confirmed, ..._lost];
    for (final t in pool) {
      t.predict(_kf);
    }

    final activated = <_STrack>[], refound = <_STrack>[], lostNow = <_STrack>[];

    // 1) high-confidence detections vs all confirmed + lost tracks
    var r = linearAssignment(_cost(pool, detsHi, true), pool.length, detsHi.length, matchThresh);
    for (final (ti, di) in r.matches) {
      final t = pool[ti];
      final wasTracked = t.state == TrackState.tracked;
      t.update(_kf, detsHi[di], fid);
      (wasTracked ? activated : refound).add(t);
    }
    final uDetHi = r.unmatchedCols;

    // 2) remaining *tracked* tracks vs low-confidence detections
    final rTracked = [for (final i in r.unmatchedRows) if (pool[i].state == TrackState.tracked) pool[i]];
    r = linearAssignment(_cost(rTracked, detsLo, false), rTracked.length, detsLo.length, 0.5);
    for (final (ti, di) in r.matches) {
      rTracked[ti].update(_kf, detsLo[di], fid);
      activated.add(rTracked[ti]);
    }
    for (final ti in r.unmatchedRows) {
      rTracked[ti].state = TrackState.lost;
      lostNow.add(rTracked[ti]);
    }

    // 3) unconfirmed tracks vs leftover high detections
    final leftHi = [for (final i in uDetHi) detsHi[i]];
    r = linearAssignment(_cost(unconfirmed, leftHi, true), unconfirmed.length, leftHi.length, 0.7);
    for (final (ti, di) in r.matches) {
      unconfirmed[ti].update(_kf, leftHi[di], fid);
      activated.add(unconfirmed[ti]);
    }
    for (final ti in r.unmatchedRows) {
      unconfirmed[ti].state = TrackState.removed;
    }

    // 4) start new tracks
    for (final di in r.unmatchedCols) {
      final d = leftHi[di];
      if (d.score >= detThresh) {
        d.activate(_kf, fid, ++_nextId);
        activated.add(d);
      }
    }

    // 5) age out lost tracks
    final stillLost = <_STrack>[];
    for (final t in [..._lost, ...lostNow]) {
      if (t.state == TrackState.tracked) continue; // re-found above
      if (fid - t.frameId > maxLostFrames) {
        t.state = TrackState.removed;
      } else {
        stillLost.add(t);
      }
    }

    final seen = <_STrack>{};
    final tracked = <_STrack>[];
    for (final t in [...activated, ...refound]) {
      if (t.state == TrackState.tracked && seen.add(t)) tracked.add(t);
    }
    _tracked = tracked;
    _lost = [for (final t in stillLost) if (!seen.contains(t)) t];

    return [
      for (final t in _tracked)
        if (t.isActivated) TrackOutput(t.trackId, t.xyxy, t.score, t.cls, t.hits, fid - t.startFrame + 1)
    ];
  }
}
