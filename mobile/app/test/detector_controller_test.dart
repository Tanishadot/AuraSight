import 'dart:io';

import 'package:aura_core/aura_core.dart';
import 'package:aurasight/services/camera_service.dart';
import 'package:aurasight/services/detection_service.dart';
import 'package:aurasight/services/detector_controller.dart';
import 'package:aurasight/services/detector_model.dart';
import 'package:aurasight/services/model_manager.dart';
import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

/// Test double for the native detector: no real inference happens here (that
/// is covered by the on-emulator integration test); it lets the controller's
/// state machine be exercised deterministically.
class FakeDetector implements DetectionService {
  bool loadOk = true;
  List<Detection> selfTestResult = [const Detection(Box(10, 10, 200, 400), 0.84, 0)];
  final calls = <String>[];
  final backends = <String>[];
  String? loadedPath;
  Object? throwOnLoad, throwOnDetect;
  Duration loadDelay = Duration.zero;
  DetectorInfo _info = DetectorInfo.notLoaded;

  @override
  DetectorInfo get info => _info;

  @override
  Future<DetectorInfo> load(String modelPath, DetectorManifest manifest, {String backend = 'cpu'}) async {
    calls.add('load');
    backends.add(backend);
    loadedPath = modelPath;
    if (loadDelay > Duration.zero) await Future<void>.delayed(loadDelay);
    if (throwOnLoad != null) throw throwOnLoad!;
    _info = loadOk
        ? DetectorInfo(
            available: true,
            backend: backend == 'cpu' ? 'cpu' : 'gpu',
            inputSize: 320,
            labels: manifest.labels,
            loadMs: 12)
        : const DetectorInfo(available: false, reason: 'unsupported op');
    return _info;
  }

  @override
  Future<DetectionResult> detectImage(Uint8List bytes, {double conf = 0.25}) async {
    calls.add('detectImage');
    if (throwOnDetect != null) throw throwOnDetect!;
    return DetectionResult(selfTestResult, 400, 600, 30);
  }

  @override
  Future<DetectionResult> detect(CameraFrame frame, {double conf = 0.1}) async {
    calls.add('detect');
    if (throwOnDetect != null) throw throwOnDetect!;
    return DetectionResult(selfTestResult, frame.uprightWidth, frame.uprightHeight, 25);
  }

  @override
  Future<List<Detection>> detectCrop(CameraFrame f, (int, int, int, int) r, int s, {double conf = 0.05}) =>
      throw UnimplementedError();
  @override
  Future<void> dispose() async {
    calls.add('dispose');
    _info = DetectorInfo.notLoaded;
  }
}

/// Stands in for the native crash journal (DetectorGuard.kt, tested on the JVM).
class FakeGuard implements DetectorGuardService {
  GuardStatus value;
  final good = <String>[];
  FakeGuard([this.value = GuardStatus.none]);
  @override
  Future<GuardStatus> status() async => value;
  @override
  Future<void> markGood(String backend) async {
    good.add(backend);
    value = GuardStatus(gpuBlocked: value.gpuBlocked, lastGood: backend);
  }

  @override
  Future<void> clearGpuBlock() async => value = GuardStatus(lastGood: value.lastGood);
}

void main() {
  late Directory dir;
  final weights = Uint8List.fromList(List.generate(50000, (i) => (i * 7) % 251));
  DetectorManifest manifest(String sha) => DetectorManifest(
        id: 'yolo-test',
        title: 'Object detector (YOLO11n)',
        purpose: 'test',
        license: 'AGPL-3.0',
        file: ModelFile('model.tflite', 'https://example.invalid/model.tflite', weights.length, sha),
        inputSize: 320,
        outputShape: const [1, 84, 2100],
        normalizedBoxes: true,
        iouThreshold: 0.5,
        labels: const ['person', 'bicycle'],
      );
  final goodSha = sha256.convert(weights).toString();
  var requests = 0;

  ModelManager mm(Uint8List served) => ModelManager(dir,
      client: () => MockClient((_) async {
            requests++;
            return http.Response.bytes(served, 200);
          }));

  /// No network at all: any request fails the test.
  ModelManager offline() => ModelManager(dir, client: () => MockClient((_) async => throw StateError('network used')));

  DetectorController controller(FakeDetector d, ModelManager m, {DetectorManifest? man, FakeGuard? guard}) =>
      DetectorController(
          detector: d,
          models: m,
          manifest: man ?? manifest(goodSha),
          selfTestImage: () async => Uint8List(4),
          guard: guard);

  setUp(() async {
    dir = await Directory.systemTemp.createTemp('aura_det');
    requests = 0;
  });
  tearDown(() => dir.delete(recursive: true));

  test('not downloaded -> downloading -> verifying -> not loaded -> (user) loading -> ready', () async {
    final d = FakeDetector();
    final guard = FakeGuard();
    final c = controller(d, mm(weights), guard: guard);
    final phases = <DetectorPhase>[];
    c.status.addListener(() => phases.add(c.status.value.phase));
    expect((await c.init()).phase, DetectorPhase.notDownloaded);
    expect(c.status.value.totalBytes, weights.length);
    expect((await c.download()).phase, DetectorPhase.notLoaded, reason: 'downloading must not load the model');
    expect(d.calls, isEmpty);
    final s = await c.load();
    expect(s.phase, DetectorPhase.ready);
    expect(d.backends, ['cpu'], reason: 'the CPU is the default backend');
    expect(guard.good, ['cpu']);
    expect(s.info!.backend, 'cpu');
    expect(s.selfTest, startsWith('person 0.84'));
    expect(
        phases,
        containsAllInOrder([
          DetectorPhase.downloading,
          DetectorPhase.verifying,
          DetectorPhase.notLoaded,
          DetectorPhase.loading,
          DetectorPhase.ready,
        ]));
    expect(d.calls, ['load', 'detectImage']);
    expect(File(d.loadedPath!).lengthSync(), weights.length);
  });

  test('a corrupted download is rejected and never loaded', () async {
    final bad = Uint8List.fromList(weights)..[100] ^= 0xff;
    final d = FakeDetector();
    final c = controller(d, mm(bad));
    await c.init();
    final s = await c.download();
    expect(s.phase, DetectorPhase.error);
    expect(s.error, contains('integrity'));
    expect(s.recovery, isNotNull);
    expect(d.calls, isEmpty);
  });

  test('a file damaged after download is caught before loading and deleted', () async {
    final d = FakeDetector();
    final c = controller(d, mm(weights));
    await c.init();
    await c.download();
    final man = manifest(goodSha);
    final f = mm(weights).fileFor(man.spec, man.file);
    final bytes = f.readAsBytesSync()..[5] ^= 0x01;
    f.writeAsBytesSync(bytes);
    d.calls.clear();
    final s = await controller(d, offline()).load();
    expect(s.phase, DetectorPhase.error);
    expect(s.error, contains('damaged'));
    expect(d.calls, isEmpty);
    expect(f.existsSync(), isFalse);
  });

  test('ready is never reported when the self-test fails', () async {
    final d = FakeDetector()..selfTestResult = [const Detection(Box(0, 0, 10, 10), 0.9, 1)]; // a bicycle, not a person
    final c = controller(d, mm(weights));
    await c.init();
    await c.download();
    final s = await c.load();
    expect(s.phase, DetectorPhase.error);
    expect(s.error, contains('self-test'));
    expect(s.error, contains('bicycle 0.90'));
    expect(d.calls.last, 'dispose');
    expect(d.info.available, isFalse);
  });

  test('a native load failure is reported with a retry path', () async {
    final d = FakeDetector()..loadOk = false;
    final c = controller(d, mm(weights));
    await c.init();
    await c.download();
    expect((await c.load()).error, 'unsupported op');
    d.loadOk = true;
    expect((await c.retry()).phase, DetectorPhase.ready); // file kept: retry only reloads
  });

  test('works offline after the first download, and auto-loads only after a good load', () async {
    final guard = FakeGuard();
    final first = controller(FakeDetector(), mm(weights), guard: guard);
    await first.download();
    expect(requests, 1);
    // Downloaded but never loaded successfully: a restart waits for the user.
    var d = FakeDetector();
    expect((await controller(d, offline(), guard: guard).init()).phase, DetectorPhase.notLoaded);
    expect(d.calls, isEmpty);
    await first.load();
    // A fresh start with no network after a good load: loads from local storage, verified locally.
    d = FakeDetector();
    final s = await controller(d, offline(), guard: guard).init();
    expect(s.phase, DetectorPhase.ready);
    expect(d.backends, ['cpu']);
    expect(requests, 1);
  });

  test('a crash during the previous load is reported and nothing is loaded automatically', () async {
    await controller(FakeDetector(), mm(weights)).download();
    final d = FakeDetector();
    final guard = FakeGuard(const GuardStatus(
        interruptedStage: 'gpu_warmup',
        interruptedBackend: 'gpu',
        gpuBlocked: 'AuraSight closed while the GPU was running the detector (step: gpu_warmup).'));
    final c = controller(d, offline(), guard: guard);
    final s = await c.init();
    expect(s.phase, DetectorPhase.error);
    expect(s.crashedStage, 'gpu_warmup');
    expect(s.error, contains('first GPU inference'));
    expect(s.recovery, contains('CPU'));
    expect(d.calls, isEmpty, reason: 'must not repeat the step that crashed');
    // Retry loads on the CPU.
    guard.value = GuardStatus(gpuBlocked: guard.value.gpuBlocked);
    expect((await c.retry()).phase, DetectorPhase.ready);
    expect(d.backends, ['cpu']);
  });

  test('a CPU crash is reported too, with a pointer to the crash report', () async {
    await controller(FakeDetector(), mm(weights)).download();
    final c = controller(FakeDetector(), offline(),
        guard: FakeGuard(const GuardStatus(interruptedStage: 'first_inference', interruptedBackend: 'cpu')));
    final s = await c.init();
    expect(s.phase, DetectorPhase.error);
    expect(s.recovery, contains('crash report'));
  });

  test('concurrent loads share one native load', () async {
    await controller(FakeDetector(), mm(weights)).download();
    final d = FakeDetector()..loadDelay = const Duration(milliseconds: 20);
    final c = controller(d, offline());
    final results = await Future.wait([c.load(), c.load(), c.retry()]);
    expect(results.map((s) => s.phase), everyElement(DetectorPhase.ready));
    expect(d.calls.where((c) => c == 'load'), hasLength(1));
    expect(d.calls.where((c) => c == 'detectImage'), hasLength(1), reason: 'one self-test per load');
  });

  test('exceptions from the native side end in a visible error, never a throw', () async {
    await controller(FakeDetector(), mm(weights)).download();
    final d = FakeDetector()..throwOnLoad = StateError('channel broke');
    final c = controller(d, offline());
    final s = await c.load();
    expect(s.phase, DetectorPhase.error);
    expect(s.error, contains('channel broke'));
    d.throwOnLoad = null;
    d.throwOnDetect = const DetectorFailure('Failed to allocate', outOfMemory: true);
    final oom = await c.retry();
    expect(oom.phase, DetectorPhase.error);
    expect(oom.error, contains('out of memory'));
    expect(d.info.available, isFalse, reason: 'released after a failed self-test');
    d.throwOnDetect = null;
    expect((await c.retry()).phase, DetectorPhase.ready);
  });

  test('trying the GPU asks for auto (GPU with CPU fallback)', () async {
    await controller(FakeDetector(), mm(weights)).download();
    final d = FakeDetector();
    final guard = FakeGuard();
    final c = controller(d, offline(), guard: guard);
    await c.load();
    final s = await c.tryGpu();
    expect(d.backends, ['cpu', 'auto']);
    expect(s.info!.backend, 'gpu');
    expect(guard.good.last, 'gpu');
    // After a good GPU load, the next start auto-loads with GPU allowed (still with CPU fallback).
    final d2 = FakeDetector();
    await controller(d2, offline(), guard: guard).init();
    expect(d2.backends, ['auto']);
  });

  test('delete returns to not downloaded', () async {
    final d = FakeDetector();
    final c = controller(d, mm(weights));
    await c.download();
    await c.load();
    await c.deleteModel();
    expect(c.status.value.phase, DetectorPhase.notDownloaded);
    expect(d.calls.last, 'dispose');
  });

  test('a build without the manifest says so', () async {
    final c = DetectorController(
        detector: FakeDetector(), models: offline(), manifest: null, selfTestImage: () async => Uint8List(0));
    final s = await c.init();
    expect(s.phase, DetectorPhase.error);
    expect(s.error, contains('missing'));
  });
}
