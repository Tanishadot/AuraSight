import 'dart:typed_data';

import 'package:aura_core/aura_core.dart';

/// Result of analysing one frame.
class FrameAnalysis {
  final List<ObjectView> objects; // tentative + confirmed (rejected left out)
  final List<SceneEvent> events; // only from confirmed objects
  final List<VerifyRequest> verifyRequests;
  const FrameAnalysis(this.objects, this.events, this.verifyRequests);

  List<ObjectView> get confirmed => [
        for (final o in objects)
          if (o.confirmed) o
      ];
}

/// Tracking, verification, spatial awareness and scene events, all in shared
/// Dart (aura_core). Mirrors the desktop pipeline's `_process` step.
class TrackingService {
  final AuraSettings Function() _settings;
  final double Function() _clock;
  late final ByteTracker tracker;
  late ObjectVerifier verifier;
  late final SceneMonitor monitor;
  List<String> _labels = const [];

  TrackingService(this._settings, this._clock) {
    final s = _settings();
    tracker = ByteTracker(trackThresh: s.confidenceThreshold, maxLostFrames: _lostFrames(s));
    verifier = ObjectVerifier(_settings, const [], _clock);
    monitor = SceneMonitor(_settings, _clock);
  }

  static int _lostFrames(AuraSettings s) => (2.0 * s.detectionFps).round().clamp(5, 60); // ~2 s of occlusion

  void setLabels(List<String> labels) {
    _labels = labels;
    verifier = ObjectVerifier(_settings, labels, _clock);
  }

  void applySettings(AuraSettings s) {
    tracker
      ..trackThresh = s.confidenceThreshold
      ..detThresh = s.confidenceThreshold + 0.05
      ..maxLostFrames = _lostFrames(s);
  }

  void reset() {
    tracker.reset();
    verifier.reset();
    monitor.reset();
  }

  FrameAnalysis process(List<Detection> detections, int width, int height, {Float32List? thumbnail}) {
    final tracks = tracker.update(detections);
    final up = verifier.update(tracks);
    final now = _clock();
    final objects = <ObjectView>[];
    for (final t in tracks) {
      final st = up.states[t.trackId]!;
      if (st.status == ObjectStatus.rejected) continue;
      final label = t.cls < _labels.length ? _labels[t.cls] : 'object';
      double cx(double v) => v.clamp(0.0, width.toDouble());
      double cy(double v) => v.clamp(0.0, height.toDouble());
      final b = Box(cx(t.box.x1), cy(t.box.y1), cx(t.box.x2), cy(t.box.y2));
      final first = monitor.firstSeen(t.trackId);
      objects.add(ObjectView(t.trackId, label, t.score, [b.x1 / width, b.y1 / height, b.x2 / width, b.y2 / height],
          analyzeBox(b, width, height, label), t.hits, first ?? now,
          status: st.status, statusReason: st.reason));
    }
    // Only confirmed objects can create events, announcements and alerts.
    final events = monitor.update([
      for (final o in objects)
        if (o.confirmed) o
    ], thumbnail);
    return FrameAnalysis(objects, events, up.requests);
  }
}
