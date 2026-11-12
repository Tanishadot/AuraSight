import 'dart:async';

import 'package:aura_core/aura_core.dart';
import 'package:aurasight/data/database.dart';
import 'package:aurasight/services/audio_orchestrator.dart';
import 'package:aurasight/services/camera_lifecycle.dart';
import 'package:aurasight/services/camera_service.dart';
import 'package:aurasight/services/detection_service.dart';
import 'package:aurasight/services/detector_model.dart';
import 'package:aurasight/services/live_pipeline.dart';
import 'package:aurasight/services/tracking_service.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';

const labels = ['person', 'bicycle', 'car', 'chair'];

/// Test double standing in for the native detector (real inference is tested
/// on the emulator); returns fixed boxes so the scheduling and tracking can be checked.
class ScriptedDetector implements DetectionService {
  Duration latency = Duration.zero;
  List<Detection> next = const [];
  int calls = 0, cropCalls = 0, inFlight = 0, maxInFlight = 0;
  bool available = true;
  Object? error; // thrown by detect when set

  Future<T> _track<T>(Future<T> Function() f) async {
    inFlight++;
    if (inFlight > maxInFlight) maxInFlight = inFlight;
    try {
      return await f();
    } finally {
      inFlight--;
    }
  }

  @override
  DetectorInfo get info => available
      ? const DetectorInfo(available: true, backend: 'cpu', inputSize: 320, labels: labels)
      : DetectorInfo.notLoaded;

  @override
  Future<DetectionResult> detect(CameraFrame frame, {double conf = 0.1}) => _track(() async {
        calls++;
        if (latency > Duration.zero) await Future<void>.delayed(latency);
        if (error != null) throw error!;
        return DetectionResult(next, 480, 640, latency.inMicroseconds / 1000);
      });

  @override
  Future<DetectorInfo> load(String modelPath, DetectorManifest manifest, {String backend = 'cpu'}) async => info;
  @override
  Future<List<Detection>> detectCrop(CameraFrame f, (int, int, int, int) r, int s, {double conf = 0.05}) =>
      _track(() async {
        cropCalls++;
        if (latency > Duration.zero) await Future<void>.delayed(latency);
        return const [];
      });
  @override
  Future<DetectionResult> detectImage(Uint8List bytes, {double conf = 0.25}) => throw UnimplementedError();
  @override
  Future<void> dispose() async {}
}

class FakeCamera implements CameraService {
  final _frames = StreamController<CameraFrame>.broadcast();
  int starts = 0, stops = 0;
  @override
  final ValueNotifier<CameraState> state = ValueNotifier(const CameraState(CameraStatus.off));
  @override
  CameraFrame? latestFrame;
  @override
  Stream<CameraFrame> get frames => _frames.stream;
  void emit() {
    final f = CameraFrame(640, 480, 90, [FramePlane(Uint8List(640 * 480), 640, 1)], DateTime.now());
    latestFrame = f;
    _frames.add(f);
  }

  @override
  Widget? buildPreview() => null;
  @override
  Future<void> start() async {
    starts++;
    state.value = const CameraState(CameraStatus.running);
  }

  @override
  Future<void> stop() async {
    stops++;
    state.value = const CameraState(CameraStatus.off);
  }

  @override
  Future<void> dispose() async {}
}

class FakeTts implements TtsEngine {
  final spoken = <String>[];
  @override
  Future<void> init() async {}
  @override
  Future<bool> speak(String text, {required double rate, required double volume, String? voiceId}) async {
    spoken.add(text);
    return true;
  }

  @override
  Future<void> interrupt() async {}
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late FakeCamera camera;
  late ScriptedDetector det;
  late LivePipeline pipeline;
  late FakeTts tts;
  late List<ActivityEntry> log;
  var settings = const AuraSettings();
  var now = DateTime(2026, 11, 12, 10);
  var vlmBusy = false;
  var clockS = 0.0;
  late AudioOrchestrator audio;

  LivePipeline build({double maxFps = 15}) => LivePipeline(
        settings: () => settings,
        camera: camera,
        detector: det,
        tracking: TrackingService(() => settings, () => clockS),
        audio: audio,
        vlmBusy: () => vlmBusy,
        now: () => now,
        maxFps: maxFps,
      );

  setUp(() async {
    settings = const AuraSettings(detectionFps: 5, minTrackFrames: 3);
    now = DateTime(2026, 11, 12, 10);
    vlmBusy = false;
    clockS = 0;
    camera = FakeCamera();
    det = ScriptedDetector();
    tts = FakeTts();
    log = [];
    double clock() => clockS;
    final queue = SpeechQueue(() => settings, tts, clock);
    audio = AudioOrchestrator(() => settings, queue, clock, log: (e) async => log.add(e), haptic: (_) async {});
    await audio.start();
    pipeline = build();
    await camera.start();
    await pipeline.start();
  });

  /// Emit one frame [ms] after the previous one and let it be processed.
  Future<void> frameAfter(int ms) async {
    now = now.add(Duration(milliseconds: ms));
    clockS += ms / 1000;
    camera.emit();
    await Future<void>.delayed(Duration.zero);
    await Future<void>.delayed(Duration.zero);
  }

  test('never queues frames while a detection is in flight', () async {
    det.latency = const Duration(milliseconds: 50);
    for (var i = 0; i < 5; i++) {
      camera.emit(); // a burst while the first detection is running
    }
    await Future<void>.delayed(const Duration(milliseconds: 80));
    expect(det.calls, 1);
    expect(pipeline.skipped, 4);
  });

  test('caps the rate at the configured detections per second', () async {
    for (var i = 0; i < 30; i++) {
      await frameAfter(100); // a 10 fps camera for 3 s
    }
    expect(det.calls, inInclusiveRange(14, 16)); // 5 per second
  });

  test('slows to 1 per second while the scene model runs', () async {
    vlmBusy = true;
    for (var i = 0; i < 30; i++) {
      await frameAfter(100);
    }
    expect(det.calls, inInclusiveRange(3, 4));
    expect(pipeline.snapshot.value.throttleReason, 'scene model running');
  });

  test('analyses nothing until the detector is ready', () async {
    det.available = false;
    for (var i = 0; i < 5; i++) {
      await frameAfter(300);
    }
    expect(det.calls, 0);
  });

  test('detections flow through tracking into spoken, positioned announcements', () async {
    // a chair on the left at middle distance (upright 480x640 frame)
    det.next = const [Detection(Box(20, 300, 140, 480), 0.82, 3)];
    for (var i = 0; i < 6; i++) {
      await frameAfter(250);
    }
    final objs = pipeline.snapshot.value.objects;
    expect(objs.single.label, 'chair');
    expect(objs.single.confirmed, isTrue);
    expect(objs.single.spatial.direction, 'on your left');
    expect(log.map((e) => e.message), contains(startsWith('Chair on your left')));
    await Future<void>.delayed(const Duration(milliseconds: 20));
    expect(tts.spoken, contains(startsWith('New: chair on your left')));
  });

  test('a stop while a frame is analysed discards its result', () async {
    det.latency = const Duration(milliseconds: 30);
    det.next = const [Detection(Box(20, 300, 140, 480), 0.82, 3)];
    camera.emit();
    await pipeline.stop();
    await Future<void>.delayed(const Duration(milliseconds: 50));
    expect(pipeline.processed, 0);
    expect(pipeline.snapshot.value.objects, isEmpty);
  });

  test('the default build caps detection at the conservative rate', () async {
    await pipeline.stop();
    pipeline = build(maxFps: LivePipeline.conservativeFps);
    await pipeline.start();
    settings = const AuraSettings(detectionFps: 6, minTrackFrames: 3);
    for (var i = 0; i < 30; i++) {
      await frameAfter(100); // a 10 fps camera for 3 s
    }
    expect(det.calls, inInclusiveRange(5, 7)); // 2 per second
    expect(pipeline.snapshot.value.throttleReason, 'safe start-up rate');
  });

  test('verification crops share the single in-flight slot with detection', () async {
    // A confusable class (tv) needs zoomed-crop checks; with a slow detector, nothing may overlap.
    const tvLabels = ['tv'];
    final tv = _TvDetector();
    await pipeline.stop();
    pipeline = LivePipeline(
      settings: () => settings,
      camera: camera,
      detector: tv,
      tracking: TrackingService(() => settings, () => clockS)..setLabels(tvLabels),
      audio: audio,
      vlmBusy: () => false,
      now: () => now,
      maxFps: 15,
    );
    await pipeline.start();
    for (var i = 0; i < 40; i++) {
      now = now.add(const Duration(milliseconds: 100));
      clockS += 0.1;
      camera.emit();
      await Future<void>.delayed(const Duration(milliseconds: 3));
    }
    await Future<void>.delayed(const Duration(milliseconds: 50));
    expect(tv.cropCalls, greaterThan(0), reason: 'the scenario must exercise verification');
    expect(tv.maxInFlight, 1);
  });

  test('a busy detector skips the frame without counting an error', () async {
    det.error = DetectorBusy();
    for (var i = 0; i < 6; i++) {
      await frameAfter(250);
    }
    expect(pipeline.snapshot.value.error, isNull);
    expect(log.where((e) => e.priority == 'error'), isEmpty);
  });

  test('repeated failures stop detection with a visible error; Retry resumes', () async {
    det.error = const DetectorFailure('model produced non-finite output');
    for (var i = 0; i < 10; i++) {
      await frameAfter(250);
    }
    expect(det.calls, LivePipeline.maxFailures, reason: 'stops calling a failing detector');
    expect(pipeline.snapshot.value.error, contains('non-finite'));
    expect(log.where((e) => e.priority == 'error'), hasLength(1));
    det.error = null;
    await pipeline.start(); // the Retry button
    expect(pipeline.snapshot.value.error, isNull);
    await frameAfter(250);
    expect(det.calls, LivePipeline.maxFailures + 1);
  });

  group('camera lifecycle', () {
    test('background stops camera and detection; returning restores both', () async {
      final lc = CameraLifecycle(camera, pipeline);
      await lc.onAppLifecycle(AppLifecycleState.inactive);
      await lc.onAppLifecycle(AppLifecycleState.paused);
      expect(camera.state.value.status, CameraStatus.off);
      expect(pipeline.running, isFalse);
      expect(camera.stops, 1);
      await lc.onAppLifecycle(AppLifecycleState.resumed);
      expect(camera.state.value.status, CameraStatus.running);
      expect(pipeline.running, isTrue);
    });

    test('a camera the user turned off stays off after returning', () async {
      final lc = CameraLifecycle(camera, pipeline);
      await lc.stop();
      await lc.onAppLifecycle(AppLifecycleState.paused);
      await lc.onAppLifecycle(AppLifecycleState.resumed);
      expect(camera.state.value.status, CameraStatus.off);
      expect(pipeline.running, isFalse);
    });
  });
}

/// Always sees a mid-confidence TV, so the verifier keeps asking for crop checks.
class _TvDetector extends ScriptedDetector {
  _TvDetector() {
    latency = const Duration(milliseconds: 10);
    next = const [Detection(Box(100, 100, 300, 300), 0.6, 0)];
  }

  @override
  DetectorInfo get info => const DetectorInfo(available: true, backend: 'cpu', inputSize: 320, labels: ['tv']);
}
