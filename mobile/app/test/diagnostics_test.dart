import 'dart:async';
import 'dart:io';

import 'package:aurasight/services/benchmark.dart';
import 'package:aurasight/services/camera_service.dart';
import 'package:aurasight/services/detection_service.dart';
import 'package:aurasight/services/detector_controller.dart';
import 'package:aurasight/services/detector_model.dart';
import 'package:aurasight/services/diagnostics.dart';
import 'package:aurasight/services/model_manager.dart';
import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'detector_controller_test.dart' show FakeDetector, FakeGuard;
import 'pipeline_test.dart' show FakeCamera;

/// A camera that delivers frames at about 20 fps while running.
class StreamingCamera extends FakeCamera {
  Timer? _t;
  @override
  Future<void> start() async {
    await super.start();
    _t = Timer.periodic(const Duration(milliseconds: 50), (_) => emit());
  }

  @override
  Future<void> stop() async {
    _t?.cancel();
    await super.stop();
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  late Directory dir;
  final weights = Uint8List.fromList(List.generate(4000, (i) => i % 251));
  final man = DetectorManifest(
    id: 'yolo-test',
    title: 'Object detector',
    purpose: 'test',
    license: 'AGPL-3.0',
    file: ModelFile('model.tflite', 'https://example.invalid/m', weights.length, sha256.convert(weights).toString()),
    inputSize: 320,
    outputShape: const [1, 84, 2100],
    normalizedBoxes: true,
    iouThreshold: 0.5,
    labels: const ['person', 'bicycle'],
  );

  late FakeDetector det;
  late StreamingCamera camera;
  late DiagnosticsRunner runner;
  var stopLiveCalls = 0;
  var pss = 100;

  setUp(() async {
    dir = await Directory.systemTemp.createTemp('aura_diag');
    final models = ModelManager(dir, client: () => MockClient((_) async => http.Response.bytes(weights, 200)));
    await models.install(man.spec).last;
    det = FakeDetector();
    camera = StreamingCamera();
    stopLiveCalls = 0;
    pss = 100;
    final controller = DetectorController(
        detector: det, models: models, manifest: man, selfTestImage: () async => Uint8List(4), guard: FakeGuard());
    runner = DiagnosticsRunner(
      detector: det,
      controller: controller,
      models: models,
      manifest: man,
      camera: camera,
      stopLive: () async => stopLiveCalls++,
      testImage: () async => Uint8List(4),
      snapshot: () async => SystemSnapshot(DateTime.now(), appMemMb: pss += 10, availMemMb: 3000),
      continuousFor: const Duration(milliseconds: 2200),
      frameTimeout: const Duration(seconds: 2),
    );
  });
  tearDown(() => dir.delete(recursive: true));

  test('stages run only in order', () async {
    expect(runner.canRun(DiagStage.launch), isTrue);
    expect(runner.canRun(DiagStage.loadModel), isFalse);
    expect(() => runner.run(DiagStage.testImage), throwsStateError);
    expect(det.calls, isEmpty);
  });

  test('all six stages pass in order, recording latency and memory, one request at a time', () async {
    for (final s in DiagStage.values) {
      final r = await runner.run(s);
      expect(r.ok, isTrue, reason: '${s.name}: ${r.summary}');
      expect(r.after.appMemMb, greaterThan(r.before.appMemMb!));
      expect(r.logLine, startsWith('AuraDiag: stage=${s.name} ok=true'));
    }
    final res = runner.results.value;
    expect(res[DiagStage.launch]!.latency.n, 0, reason: 'stage 1 loads nothing');
    expect(det.calls.first, 'load', reason: 'nothing native before stage 2');
    expect(res[DiagStage.loadModel]!.summary, startsWith('CPU'));
    expect(det.backends.first, 'cpu');
    expect(res[DiagStage.testImage]!.summary, contains('person 0.84'));
    expect(res[DiagStage.cameraFrame]!.latency.n, 1);
    expect(res[DiagStage.continuous]!.latency.n, inInclusiveRange(2, 3), reason: 'paced at 1 per second');
    expect(res[DiagStage.fullSelfTest]!.summary, contains('ready on CPU'));
    expect(camera.state.value.status, CameraStatus.off, reason: 'camera released after each camera stage');
    expect(stopLiveCalls, 3, reason: 'live detection stopped before the load and camera stages');
    final report = runner.report(SystemSnapshot(DateTime.now(), device: 'vivo I2214', soc: 'QTI SM6375'));
    expect(RegExp(r'\| PASS').allMatches(report), hasLength(6));
    expect(report, contains('vivo I2214'));
  });

  test('a failed stage is recorded and blocks the next one', () async {
    det.loadOk = false;
    await runner.run(DiagStage.launch);
    final r = await runner.run(DiagStage.loadModel);
    expect(r.ok, isFalse);
    expect(r.summary, contains('unsupported op'));
    expect(runner.canRun(DiagStage.testImage), isFalse);
    expect(runner.canRun(DiagStage.loadModel), isTrue, reason: 'Retry');
  });

  test('an exception inside a stage becomes a failed result, not a crash', () async {
    await runner.run(DiagStage.launch);
    await runner.run(DiagStage.loadModel);
    det.throwOnDetect = const DetectorFailure('Failed to allocate', outOfMemory: true);
    final r = await runner.run(DiagStage.testImage);
    expect(r.ok, isFalse);
    expect(r.summary, contains('Failed to allocate'));
    expect(runner.running.value, isNull);
  });

  test('crash report text names the interrupted step', () {
    const report = CrashReport(
        GuardStatus(interruptedStage: 'gpu_warmup', interruptedBackend: 'gpu', gpuBlocked: 'closed during gpu_warmup'),
        [
          {
            'reason': 'native_crash',
            'description': 'crash',
            'timestampMs': 0,
            'traceHead': '*** *** ***\nsignal 11 (SIGSEGV)'
          }
        ]);
    expect(report.hasCrash, isTrue);
    final t = report.toText();
    expect(t, contains('gpu_warmup'));
    expect(t, contains('SIGSEGV'));
  });
}
