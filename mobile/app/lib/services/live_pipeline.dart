import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

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
  const LiveSnapshot({
    this.objects = const [],
    this.detectionFps = 0,
    this.inferenceMs = 0,
    this.targetFps = 0,
    this.throttleReason = '',
    this.detector = DetectorInfo.notLoaded,
  });
}

/// Camera frames -> detector (adaptive rate) -> tracker/verifier -> events
/// -> audio. Never blocks the UI thread: inference is native and awaited,
/// and at most one detection is in flight at a time (frames arriving in the
/// meantime are skipped, not queued).
class LivePipeline {
  final AuraSettings Function() _settings;
  final CameraService camera;
  final DetectionService detector;
  final TrackingService tracking;
  final AudioOrchestrator audio;
  final bool Function() vlmBusy;
  final MethodChannel _system;

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

  LivePipeline({
    required AuraSettings Function() settings,
    required this.camera,
    required this.detector,
    required this.tracking,
    required this.audio,
    required this.vlmBusy,
    MethodChannel? system,
  })  : _settings = settings,
        _system = system ?? AuraChannels.system;

  Future<void> start() async {
    final s = _settings();
    if (!detector.info.available) {
      final info = await detector.load(inputSize: s.inferenceSize);
      if (info.available) tracking.setLabels(info.labels);
    }
    snapshot.value = LiveSnapshot(detector: detector.info);
    tracking.reset();
    await _sub?.cancel();
    _sub = camera.frames.listen(_onFrame);
    _systemTimer?.cancel();
    _systemTimer = Timer.periodic(const Duration(seconds: 5), (_) => _pollSystem());
    unawaited(_pollSystem());
  }

  Future<void> stop() async {
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
    if (_inFlight || !detector.info.available) return;
    final (fps, why) = targetRate();
    final now = DateTime.now();
    if (now.difference(_lastStart).inMicroseconds < 1e6 / fps * 0.9) return;
    _inFlight = true;
    _lastStart = now;
    try {
      final r = await detector.detect(frame, conf: 0.1);
      _emaMs = _emaMs == 0 ? r.inferenceMs : 0.8 * _emaMs + 0.2 * r.inferenceMs;
      final thumb = _thumbnail(frame);
      final analysis = tracking.process(r.detections, r.width, r.height, thumbnail: thumb);
      for (final req in analysis.verifyRequests) {
        unawaited(_verify(frame, req, r.width, r.height));
      }
      audio.handleEvents(analysis.events);
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
    } catch (e) {
      unawaited(audio.log('error', 'Detection error: $e', priority: 'error'));
    } finally {
      _inFlight = false;
    }
  }

  /// Secondary check for confusable classes: re-detect on padded crops.
  Future<void> _verify(CameraFrame frame, VerifyRequest req, int w, int h) async {
    try {
      final scores = <double>[];
      for (final (pad, size) in req.crops) {
        final rect = cropRect(req.box, pad, w, h);
        if (rect == null) {
          scores.add(0);
          continue;
        }
        final dets = await detector.detectCrop(frame, rect, size);
        scores.add(cropScore(req.box, req.cls, (rect.$1.toDouble(), rect.$2.toDouble()), dets));
      }
      tracking.verifier.submit(req.trackId, scores.reduce((a, b) => a + b) / scores.length);
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
