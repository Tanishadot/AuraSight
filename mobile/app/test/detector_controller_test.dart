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
  String? loadedPath;
  DetectorInfo _info = DetectorInfo.notLoaded;

  @override
  DetectorInfo get info => _info;

  @override
  Future<DetectorInfo> load(String modelPath, DetectorManifest manifest, {String backend = 'auto'}) async {
    calls.add('load');
    loadedPath = modelPath;
    _info = loadOk
        ? DetectorInfo(available: true, backend: 'cpu', inputSize: 320, labels: manifest.labels, loadMs: 12)
        : const DetectorInfo(available: false, reason: 'unsupported op');
    return _info;
  }

  @override
  Future<DetectionResult> detectImage(Uint8List bytes, {double conf = 0.25}) async {
    calls.add('detectImage');
    return DetectionResult(selfTestResult, 400, 600, 30);
  }

  @override
  Future<DetectionResult> detect(CameraFrame frame, {double conf = 0.1}) => throw UnimplementedError();
  @override
  Future<List<Detection>> detectCrop(CameraFrame f, (int, int, int, int) r, int s, {double conf = 0.05}) =>
      throw UnimplementedError();
  @override
  Future<void> dispose() async {
    calls.add('dispose');
    _info = DetectorInfo.notLoaded;
  }
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

  DetectorController controller(FakeDetector d, ModelManager m, {DetectorManifest? man}) => DetectorController(
      detector: d, models: m, manifest: man ?? manifest(goodSha), selfTestImage: () async => Uint8List(4));

  setUp(() async {
    dir = await Directory.systemTemp.createTemp('aura_det');
    requests = 0;
  });
  tearDown(() => dir.delete(recursive: true));

  test('not downloaded -> downloading -> verifying -> loading -> ready', () async {
    final d = FakeDetector();
    final c = controller(d, mm(weights));
    final phases = <DetectorPhase>[];
    c.status.addListener(() => phases.add(c.status.value.phase));
    expect((await c.init()).phase, DetectorPhase.notDownloaded);
    expect(c.status.value.totalBytes, weights.length);
    final s = await c.download();
    expect(s.phase, DetectorPhase.ready);
    expect(s.info!.backend, 'cpu');
    expect(s.selfTest, startsWith('person 0.84'));
    expect(
        phases,
        containsAllInOrder([
          DetectorPhase.downloading,
          DetectorPhase.verifying,
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
    final s = await c.download();
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
    expect((await c.download()).error, 'unsupported op');
    d.loadOk = true;
    expect((await c.retry()).phase, DetectorPhase.ready); // file kept: retry only reloads
  });

  test('works offline after the first download', () async {
    await controller(FakeDetector(), mm(weights)).download();
    expect(requests, 1);
    // A fresh start with no network: loads from local storage, verified locally.
    final d = FakeDetector();
    final s = await controller(d, offline()).init();
    expect(s.phase, DetectorPhase.ready);
    expect(requests, 1);
  });

  test('delete returns to not downloaded', () async {
    final d = FakeDetector();
    final c = controller(d, mm(weights));
    await c.download();
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
