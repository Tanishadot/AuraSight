import 'dart:async';
import 'dart:math' as math;

import 'package:aura_core/aura_core.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import '../platform/channels.dart';
import 'audio_orchestrator.dart';
import 'camera_service.dart';
import 'detection_service.dart';
import 'tracking_service.dart';

class LiveSnapshot {
  final List<ObjectView> objects;
  final double detectionFps;
  final double inferenceMs;
  final double targetFps;
  final String throttleReason; // why we're below the configured rate, or ''
  final DetectorInfo detector;

  /// Set when detection stopped after repeated failures; shown with a Retry button.
  final String? error;
  const LiveSnapshot({
    this.objects = const [],
    this.detectionFps = 0,
    this.inferenceMs = 0,
    this.targetFps = 0,
    this.throttleReason = '',
    this.detector = DetectorInfo.notLoaded,
    this.error,
  });
}

/// Camera frames -> detector (adaptive rate) -> tracker/verifier -> events
/// -> audio. Never blocks the UI thread: inference is native and awaited,
/// and at most one native request (detection or verification crop) is in
/// flight at a time; frames arriving in the meantime are skipped, not queued.
/// After [maxFailures] failures in a row detection stops with a visible error
/// instead of retrying forever.
class LivePipeline {
  final AuraSettings Function() _settings;
  final CameraService camera;
  final DetectionService detector;
  final TrackingService tracking;
  final AudioOrchestrator audio;
  final bool Function() vlmBusy;
  final MethodChannel _system;
  final DateTime Function() _now;

  /// Upper bound on detections per second, whatever the settings say. Kept
  /// low until the detector has been proven on the phone (see the README).
  final double maxFps;
  static const conservativeFps = 2.0;
  static const maxFailures = 3;
  int _failures = 0;

  final snapshot = ValueNotifier(const LiveSnapshot());
  StreamSubscription<CameraFrame>? _sub;
  bool _inFlight = false;
  DateTime _lastStart = DateTime.fromMillisecondsSinceEpoch(0);
  double _emaMs = 0;
  int _count = 0;
  DateTime _fpsT = DateTime.now();
  double _fps = 0;
  String _thermal = 'nominal';
  bool _lowMemory = false;
  Timer? _systemTimer;
  bool _running = false;

  /// Frames analysed / frames dropped by the scheduler since start (for tests and benchmarks).
  int processed = 0, skipped = 0;

  LivePipeline({
    required AuraSettings Function() settings,
    required this.camera,
    required this.detector,
    required this.tracking,
    required this.audio,
    required this.vlmBusy,
    MethodChannel? system,
    DateTime Function()? now,
    this.maxFps = conservativeFps,
  })  : _settings = settings,
        _system = system ?? AuraChannels.system,
        _now = now ?? DateTime.now;

  bool get running => _running;

  /// Starts analysing camera frames. The model itself is loaded by the
  /// DetectorController; until it is ready, frames are not analysed.
  Future<void> start() async {
    _running = true;
    _failures = 0;
    processed = skipped = 0;
    _emaMs = 0;
    snapshot.value = LiveSnapshot(detector: detector.info);
    tracking.reset();
    await _sub?.cancel();
    _sub = camera.frames.listen(_onFrame);
    _systemTimer?.cancel();
    _systemTimer = Timer.periodic(const Duration(seconds: 5), (_) => _pollSystem());
    unawaited(_pollSystem());
  }

  Future<void> stop() async {
    _running = false;
    await _sub?.cancel();
    _sub = null;
    _systemTimer?.cancel();
    tracking.reset();
    snapshot.value = LiveSnapshot(detector: detector.info);
  }

  Future<void> _pollSystem() async {
    try {
      final r = await _system.invokeMapMethod<String, Object?>('snapshot');
      _thermal = r?['thermal'] as String? ?? 'nominal';
      _lowMemory = r?['lowMemory'] as bool? ?? false;
    } on MissingPluginException {
      // not available in this build: assume nominal
    } on PlatformException {
      // ignore transient failures
    }
  }

  /// Adaptive detection rate: the configured fps is an upper bound, lowered
  /// when the detector is slower than that, while the scene model runs, and
  /// under thermal or memory pressure.
  (double, String) targetRate() {
    final s = _settings();
    var fps = s.detectionFps;
    var why = '';
    if (fps > maxFps) {
      fps = maxFps;
      why = 'safe start-up rate';
    }
    if (_emaMs > 0) {
      final capacity = 1000 / (_emaMs * 1.25); // leave headroom for the UI and camera
      if (capacity < fps) {
        fps = capacity;
        why = 'detector speed';
      }
    }
    if (vlmBusy() && fps > 1) {
      fps = 1;
      why = 'scene model running';
    }
    if (_thermal == 'serious' && fps > 3) {
      fps = 3;
      why = 'phone is warm';
    } else if (_thermal == 'critical' && fps > 1) {
      fps = 1;
      why = 'phone is hot';
    }
    if (_lowMemory && fps > 2) {
      fps = 2;
      why = 'low memory';
    }
    return (math.max(0.5, fps), why);
  }

  Future<void> _onFrame(CameraFrame frame) async {
    if (!_running || !detector.info.available) return;
    if (_failures >= maxFailures) return; // halted: waits for Retry (start())
    if (_inFlight) {
      skipped++; // never queue frames: a stale frame is worse than a dropped one
      return;
    }
    final (fps, why) = targetRate();
    final now = _now();
    if (now.difference(_lastStart).inMicroseconds < 1e6 / fps * 0.9) {
      skipped++;
      return;
    }
    if (tracking.labels.length != detector.info.labels.length) tracking.setLabels(detector.info.labels);
    _inFlight = true;
    _lastStart = now;
    try {
      final r = await detector.detect(frame, conf: 0.1);
      if (!_running) return; // stopped while the frame was being analysed
      _failures = 0;
      _emaMs = _emaMs == 0 ? r.inferenceMs : 0.8 * _emaMs + 0.2 * r.inferenceMs;
      final thumb = _thumbnail(frame);
      final analysis = tracking.process(r.detections, r.width, r.height, thumbnail: thumb);
      // At most one verification per frame, inside the same in-flight slot, so
      // crops never pile up behind the detector (each would hold a whole frame).
      final reqs = analysis.verifyRequests;
      if (reqs.isNotEmpty) await _verify(frame, reqs.first, r.width, r.height);
      for (final req in reqs.skip(1)) {
        tracking.verifier.cancel(req.trackId); // not a miss: asked for again on a later frame
      }
      if (!_running) return;
      audio.handleEvents(analysis.events);
      processed++;
      _count++;
      final dt = now.difference(_fpsT).inMilliseconds;
      if (dt >= 1000) {
        _fps = _count * 1000 / dt;
        _count = 0;
        _fpsT = now;
      }
      snapshot.value = LiveSnapshot(
        objects: analysis.objects,
        detectionFps: _fps,
        inferenceMs: _emaMs,
        targetFps: fps,
        throttleReason: why,
        detector: detector.info,
      );
    } on DetectorBusy {
      skipped++; // another native request is running (e.g. a diagnostics test)
    } catch (e) {
      _failures++;
      debugPrint('AuraDetector: live detection failed ($_failures in a row): $e');
      if (_failures >= maxFailures && _running) {
        final msg = 'Object detection stopped after $_failures errors in a row: $e';
        snapshot.value = LiveSnapshot(detector: detector.info, error: msg);
        unawaited(audio.log('error', msg, priority: 'error'));
      }
    } finally {
      _inFlight = false;
    }
  }

  /// Secondary check for confusable classes: re-detect on padded crops.
  Future<void> _verify(CameraFrame frame, VerifyRequest req, int w, int h) async {
    try {
      final scores = <double>[];
      for (final (pad, size) in req.crops) {
        if (!_running) {
          tracking.verifier.cancel(req.trackId);
          return;
        }
        final rect = cropRect(req.box, pad, w, h);
        if (rect == null) {
          scores.add(0);
          continue;
        }
        final dets = await detector.detectCrop(frame, rect, size);
        scores.add(cropScore(req.box, req.cls, (rect.$1.toDouble(), rect.$2.toDouble()), dets));
      }
      tracking.verifier.submit(req.trackId, scores.reduce((a, b) => a + b) / scores.length);
    } on DetectorBusy {
      tracking.verifier.cancel(req.trackId);
    } catch (_) {
      tracking.verifier.submitFailure(req.trackId);
    }
  }

  /// Luma thumbnail for scene-change detection (upright orientation doesn't
  /// matter for a frame-difference measure).
  static Float32List? _thumbnail(CameraFrame f) {
    if (f.planes.isEmpty) return null;
    final y = f.planes.first;
    return thumbnailFromLuma(y.bytes, f.width, f.height, y.rowStride);
  }

  Future<void> dispose() async {
    await stop();
    snapshot.dispose();
  }
}
