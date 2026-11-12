/// Turns per-frame objects into meaningful, de-duplicated scene events.
/// Port of backend/assistant/scene_monitor.py.
///
/// Events: entered, left, approaching, obstacle, sceneChange.
///  * An object must be tracked for `minTrackFrames` before it "enters".
///  * A lost track whose class re-appears nearby within a few seconds is the
///    *same* object (no second "entered").
///  * Obstacle warnings fire on the *transition* into the near-centre zone,
///    with a per-track cooldown, so a stationary chair isn't repeated.
library;

import 'dart:collection';
import 'dart:typed_data';

import 'settings.dart';
import 'spatial.dart';
import 'verify.dart';

class ObjectView {
  final int trackId;
  final String label;
  final double confidence;
  final List<double> box; // normalised xyxy 0..1
  final SpatialInfo spatial;
  final int hits;
  final double firstSeen;
  bool approaching;
  final ObjectStatus status;
  final String statusReason;

  ObjectView(this.trackId, this.label, this.confidence, this.box, this.spatial, this.hits, this.firstSeen,
      {this.approaching = false, this.status = ObjectStatus.confirmed, this.statusReason = ''});

  bool get confirmed => status == ObjectStatus.confirmed;
}

enum EventKind { entered, left, approaching, obstacle, sceneChange }

enum EventPriority { info, warning }

class SceneEvent {
  final EventKind kind;
  final String message;
  final EventPriority priority;
  final String? label;
  final int? trackId;
  final SpatialInfo? spatial;
  const SceneEvent(this.kind, this.message, this.priority, {this.label, this.trackId, this.spatial});

  /// The desktop's event name, used in golden tests and the activity log.
  String get kindName => kind == EventKind.sceneChange ? 'scene_change' : kind.name;
}

class _Memory {
  final int trackId;
  final String label;
  double firstSeen;
  double lastSeen;
  int hits = 0;
  bool entered = false;
  double lastAlert = 0;
  bool nearPath = false;
  double cx = 0.5;
  SpatialInfo? spatial;
  final Queue<(double, double)> areas = Queue();
  _Memory(this.trackId, this.label, this.firstSeen, this.lastSeen);
}

String _cap(String s) => s.isEmpty ? s : s[0].toUpperCase() + s.substring(1);

class SceneMonitor {
  static const reidWindowS = 4.0;
  static const reidMaxDx = 0.2;
  static const approachRatio = 1.5;
  static const approachWindowS = 1.2;
  static const alertCooldownS = 10.0;
  static const changeThreshold = 28.0;
  static const changeCooldownS = 10.0;
  static const thumbW = 32, thumbH = 24;

  final AuraSettings Function() _settings;
  final double Function() _clock;

  Map<int, _Memory> _mem = {};
  List<_Memory> _recentlyLeft = [];
  Float32List? _baseline;
  int _changeStreak = 0;
  double _lastChange = -1e9;

  SceneMonitor(this._settings, this._clock);

  void reset() {
    _mem = {};
    _recentlyLeft = [];
    _baseline = null;
    _changeStreak = 0;
    _lastChange = -1e9;
  }

  double? firstSeen(int trackId) => _mem[trackId]?.firstSeen;

  /// [thumbnail]: optional 32x24 greyscale frame (see [thumbnailFromLuma]) for
  /// scene-change detection.
  List<SceneEvent> update(List<ObjectView> objects, [Float32List? thumbnail]) {
    final s = _settings();
    final now = _clock();
    final events = <SceneEvent>[];
    final seenIds = <int>{};

    for (final o in objects) {
      seenIds.add(o.trackId);
      var m = _mem[o.trackId];
      if (m == null) {
        m = _Memory(o.trackId, o.label, now, now);
        _inheritFromRecent(m, o);
        _mem[o.trackId] = m;
      }
      m.lastSeen = now;
      m.hits += 1;
      m.cx = o.spatial.cx;
      m.spatial = o.spatial;
      m.areas.addLast((now, o.spatial.area));
      if (m.areas.length > 40) m.areas.removeFirst();
      final name = spokenLabel(o.label);

      // approaching: apparent size grew quickly while in the walking path
      o.approaching = _growth(m, now) >= approachRatio;
      final nearPath = o.spatial.inPath && o.spatial.distance == Distance.near;
      final obstacleLike = obstacleClasses.contains(o.label);

      if (m.entered &&
          o.approaching &&
          o.spatial.inPath &&
          obstacleLike &&
          o.spatial.distance != Distance.far &&
          now - m.lastAlert >= alertCooldownS * 0.8) {
        m.lastAlert = now;
        events.add(SceneEvent(
            EventKind.approaching, '${_cap(name)} approaching, ${o.spatial.direction}.', EventPriority.warning,
            label: o.label, trackId: o.trackId, spatial: o.spatial));
      } else if (m.entered && nearPath && !m.nearPath && obstacleLike && now - m.lastAlert >= alertCooldownS) {
        m.lastAlert = now;
        events.add(SceneEvent(
            EventKind.obstacle, 'Caution: $name close, ${o.spatial.direction}.', EventPriority.warning,
            label: o.label, trackId: o.trackId, spatial: o.spatial));
      }

      if (!m.entered && m.hits >= s.minTrackFrames) {
        m.entered = true;
        if (nearPath && obstacleLike) {
          m.lastAlert = now;
          events.add(SceneEvent(
              EventKind.obstacle, 'Caution: $name close, ${o.spatial.direction}.', EventPriority.warning,
              label: o.label, trackId: o.trackId, spatial: o.spatial));
        } else {
          events.add(SceneEvent(EventKind.entered, '${_cap(name)} ${o.spatial.phrase()}.', EventPriority.info,
              label: o.label, trackId: o.trackId, spatial: o.spatial));
        }
      }
      m.nearPath = nearPath;
    }

    // departures
    for (final tid in _mem.keys.toList()) {
      if (seenIds.contains(tid)) continue;
      final m = _mem[tid]!;
      if (now - m.lastSeen >= s.leaveAfterS) {
        _mem.remove(tid);
        if (m.entered) {
          _recentlyLeft.add(m);
          events.add(SceneEvent(EventKind.left, '${_cap(spokenLabel(m.label))} no longer visible.', EventPriority.info,
              label: m.label, trackId: tid, spatial: m.spatial));
        }
      }
    }
    _recentlyLeft = [
      for (final m in _recentlyLeft)
        if (now - m.lastSeen < reidWindowS) m
    ];

    if (thumbnail != null) {
      final ev = _sceneChange(thumbnail, now);
      if (ev != null) events.add(ev);
    }
    return events;
  }

  void _inheritFromRecent(_Memory m, ObjectView o) {
    _Memory? best;
    for (final old in [..._recentlyLeft, ..._mem.values.where((x) => x.label == o.label)]) {
      if (old.label != o.label || old.trackId == m.trackId) continue;
      if (old.lastSeen >= m.firstSeen) continue; // still visible this frame -> a different object
      if ((old.cx - o.spatial.cx).abs() <= reidMaxDx && old.entered) {
        best = old;
        break;
      }
    }
    if (best != null) {
      m.entered = true;
      m.firstSeen = best.firstSeen;
      m.lastAlert = best.lastAlert;
      m.nearPath = best.nearPath;
      _recentlyLeft.remove(best);
    }
  }

  double _growth(_Memory m, double now) {
    if (m.areas.length < 3) return 1.0;
    final cur = m.areas.last.$2;
    double? ref;
    for (final (t, a) in m.areas) {
      if (now - t >= approachWindowS) ref = a;
    }
    if (ref == null) return 1.0;
    return ref > 1e-4 ? cur / ref : 1.0;
  }

  SceneEvent? _sceneChange(Float32List small, double now) {
    final base = _baseline;
    if (base == null) {
      _baseline = Float32List.fromList(small);
      return null;
    }
    var sum = 0.0;
    for (var i = 0; i < small.length; i++) {
      sum += (small[i] - base[i]).abs();
    }
    final diff = sum / small.length;
    // slow EMA absorbs gradual lighting drift (float32, as on the desktop)
    for (var i = 0; i < small.length; i++) {
      base[i] = 0.97 * base[i] + 0.03 * small[i];
    }
    _changeStreak = diff > changeThreshold ? _changeStreak + 1 : 0;
    if (_changeStreak >= 2 && now - _lastChange >= changeCooldownS) {
      _lastChange = now;
      _changeStreak = 0;
      _baseline = Float32List.fromList(small);
      return const SceneEvent(EventKind.sceneChange, 'The view has changed significantly.', EventPriority.info);
    }
    return null;
  }
}

/// Area-averaged 32x24 greyscale thumbnail from a camera luma (Y) plane,
/// the mobile equivalent of the desktop's `cv2.resize(gray, (32, 24), INTER_AREA)`.
Float32List thumbnailFromLuma(Uint8List y, int width, int height, int rowStride) {
  const tw = SceneMonitor.thumbW, th = SceneMonitor.thumbH;
  final out = Float32List(tw * th);
  for (var ty = 0; ty < th; ty++) {
    final y0 = ty * height ~/ th, y1 = (ty + 1) * height ~/ th;
    for (var tx = 0; tx < tw; tx++) {
      final x0 = tx * width ~/ tw, x1 = (tx + 1) * width ~/ tw;
      var sum = 0, n = 0;
      // sample every 2nd pixel: plenty for a change detector, 4x cheaper
      for (var py = y0; py < y1; py += 2) {
        final row = py * rowStride;
        for (var px = x0; px < x1; px += 2) {
          sum += y[row + px];
          n++;
        }
      }
      out[ty * tw + tx] = n == 0 ? 0 : sum / n;
    }
  }
  return out;
}
