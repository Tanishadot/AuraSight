// On-device integration test for the Android object detector. Runs inside the
// real app on an Android device or emulator (CI: x86_64 emulator), through the
// real native LiteRT code - nothing here is simulated.
//
//   python mobile/tools/gen_integration_fixtures.py
//   cd mobile/app && flutter test integration_test/detector_test.dart -d <device>
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:aura_core/aura_core.dart';
import 'package:aurasight/services/camera_service.dart';
import 'package:aurasight/services/detection_service.dart';
import 'package:aurasight/services/detector_controller.dart';
import 'package:aurasight/services/detector_model.dart';
import 'package:aurasight/services/model_manager.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/testing.dart';
import 'package:integration_test/integration_test.dart';
import 'package:path_provider/path_provider.dart';

import 'fixtures.g.dart';

const selfTestAsset = 'assets/selftest/person_living_room.jpg';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late DetectorManifest manifest;
  late Directory root;
  final detector = NativeDetectionService();

  ModelManager online() => ModelManager(root);
  ModelManager offline() =>
      ModelManager(root, client: () => MockClient((_) async => throw StateError('network must not be used')));
  Future<Uint8List> selfTest() async => (await rootBundle.load(selfTestAsset)).buffer.asUint8List();

  setUpAll(() async {
    manifest = await DetectorManifest.load();
    root = Directory('${(await getApplicationSupportDirectory()).path}/it-models');
    if (root.existsSync()) root.deleteSync(recursive: true);
  });

  testWidgets('downloads the published model and verifies its SHA-256', (_) async {
    final events = await online().install(manifest.spec).toList();
    expect(events.last.status, ModelStatus.installed, reason: events.last.error);
    expect(events.any((e) => e.status == ModelStatus.verifying), isTrue);
    expect(await online().verifyInstalled(manifest.spec), isTrue);
    final f = online().fileFor(manifest.spec, manifest.file);
    expect(f.lengthSync(), manifest.file.bytes);
  });

  testWidgets('loads with LiteRT and passes the self-test, offline', (_) async {
    final c = DetectorController(detector: detector, models: offline(), manifest: manifest, selfTestImage: selfTest);
    final s = await c.init();
    expect(s.phase, DetectorPhase.ready, reason: '${s.error} ${s.recovery}');
    debugPrint('DETECTOR backend=${s.info!.backend} loadMs=${s.info!.loadMs.toStringAsFixed(0)} '
        'fallback=${s.info!.fallbackReason} selfTest=${s.selfTest}');
    expect(s.info!.labels.length, 80);
  });

  testWidgets('detections match the desktop detector on the fixture photos', (_) async {
    final golden = jsonDecode(desktopGoldenJson) as Map<String, Object?>;
    var refs = 0, hits = 0;
    final ious = <double>[], dScores = <double>[];
    final rows = <String>[];
    for (final g in (golden['images']! as List).cast<Map<String, Object?>>()) {
      final bytes = base64Decode(fixtureImages[g['file']]!);
      final r = await detector.detectImage(bytes, conf: 0.25);
      expect([r.width, r.height], [g['width'], g['height']]);
      final mob = [for (final d in r.detections) d];
      final used = <int>{};
      final missed = <String>[];
      for (final d in (g['detections']! as List).cast<Map<String, Object?>>()) {
        if ((d['score']! as num) < 0.4) continue;
        refs++;
        final ref = Box.fromList(d['box']! as List);
        var best = 0.0, bj = -1;
        for (var j = 0; j < mob.length; j++) {
          if (used.contains(j) || mob[j].cls != d['cls']) continue;
          final v = iou(ref, mob[j].box);
          if (v > best) {
            best = v;
            bj = j;
          }
        }
        if (best >= 0.5) {
          hits++;
          used.add(bj);
          ious.add(best);
          dScores.add(mob[bj].score - (d['score']! as num));
        } else {
          missed.add('${d['label']} ${d['score']}');
        }
      }
      final labels = mob.map((d) => '${manifest.labels[d.cls]} ${d.score.toStringAsFixed(2)}').join(', ');
      rows.add('${g['file']}: phone=[$labels] missed=[${missed.join(', ')}] '
          'model ${r.modelMs.toStringAsFixed(0)} ms');
    }
    final recall = refs == 0 ? 1.0 : hits / refs;
    final meanIou = ious.isEmpty ? 0.0 : ious.reduce((a, b) => a + b) / ious.length;
    final meanDs = dScores.isEmpty ? 0.0 : dScores.reduce((a, b) => a + b) / dScores.length;
    final maxDs = dScores.isEmpty ? 0.0 : dScores.map((e) => e.abs()).reduce(math.max);
    for (final r in rows) {
      debugPrint('PARITY $r');
    }
    debugPrint('PARITY summary: desktop detections (>=0.40)=$refs matched=$hits recall=${recall.toStringAsFixed(3)} '
        'meanIoU=${meanIou.toStringAsFixed(3)} meanScoreDiff=${meanDs.toStringAsFixed(3)} '
        'maxAbsScoreDiff=${maxDs.toStringAsFixed(3)}');
    expect(recall, greaterThanOrEqualTo(0.85));
    expect(meanIou, greaterThanOrEqualTo(0.8));
  });

  testWidgets('camera path: a rotated YUV frame gives the same detections as the photo', (_) async {
    final bytes = base64Decode(fixtureImages['person_desk_laptop_cup.jpg']!);
    final direct = await detector.detectImage(bytes, conf: 0.25);
    final frame = await _yuvFrameRotated90(bytes);
    final viaCamera = await detector.detect(frame, conf: 0.25);
    expect([viaCamera.width, viaCamera.height], [frame.uprightWidth, frame.uprightHeight]);
    final strong = [for (final d in direct.detections) if (d.score >= 0.5) d];
    expect(strong, isNotEmpty);
    for (final d in strong) {
      final match = viaCamera.detections.where((m) => m.cls == d.cls).map((m) => iou(d.box, m.box)).fold(0.0, math.max);
      expect(match, greaterThan(0.8), reason: '${manifest.labels[d.cls]} ${d.score}');
    }
    debugPrint('YUV path: ${viaCamera.detections.length} detections, '
        'pre ${viaCamera.preMs.toStringAsFixed(1)} ms, model ${viaCamera.modelMs.toStringAsFixed(1)} ms');
  });

  testWidgets('a damaged model file is refused before loading', (_) async {
    final f = offline().fileFor(manifest.spec, manifest.file);
    final bytes = f.readAsBytesSync();
    bytes[1000] ^= 0xff;
    f.writeAsBytesSync(bytes, flush: true);
    final c = DetectorController(detector: detector, models: offline(), manifest: manifest, selfTestImage: selfTest);
    final s = await c.load();
    expect(s.phase, DetectorPhase.error);
    expect(s.error, contains('damaged'));
    expect(f.existsSync(), isFalse);
  });

  tearDownAll(() async {
    await detector.dispose();
    if (root.existsSync()) root.deleteSync(recursive: true);
  });
}

/// Builds the YUV 4:2:0 frame a back camera would deliver for this photo when
/// the phone is upright: the sensor image is the photo rotated 90 degrees
/// anticlockwise, so rotating it 90 degrees clockwise restores the photo.
Future<CameraFrame> _yuvFrameRotated90(Uint8List jpeg) async {
  final codec = await ui.instantiateImageCodec(jpeg);
  final img = (await codec.getNextFrame()).image;
  final rgba = (await img.toByteData(format: ui.ImageByteFormat.rawRgba))!.buffer.asUint8List();
  final uw = img.width & ~1, uh = img.height & ~1; // even sizes for 4:2:0
  final sw = uh, sh = uw; // sensor is the upright image turned sideways
  final y = Uint8List(sw * sh), u = Uint8List((sw ~/ 2) * (sh ~/ 2)), v = Uint8List((sw ~/ 2) * (sh ~/ 2));
  for (var sy = 0; sy < sh; sy++) {
    for (var sx = 0; sx < sw; sx++) {
      // inverse of the native mapping for rotation 90: sx = uy, sy = sensorH - 1 - ux
      final ux = sh - 1 - sy, uy = sx;
      final i = (uy * img.width + ux) * 4;
      final r = rgba[i], g = rgba[i + 1], b = rgba[i + 2];
      y[sy * sw + sx] = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp(0, 255);
      if (sx.isEven && sy.isEven) {
        final ci = (sy ~/ 2) * (sw ~/ 2) + sx ~/ 2;
        u[ci] = (-0.168736 * r - 0.331264 * g + 0.5 * b + 128).round().clamp(0, 255);
        v[ci] = (0.5 * r - 0.418688 * g - 0.081312 * b + 128).round().clamp(0, 255);
      }
    }
  }
  return CameraFrame(sw, sh, 90,
      [FramePlane(y, sw, 1), FramePlane(u, sw ~/ 2, 1), FramePlane(v, sw ~/ 2, 1)], DateTime.now());
}
