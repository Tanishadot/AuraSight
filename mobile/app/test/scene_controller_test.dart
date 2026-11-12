import 'dart:async';
import 'dart:typed_data';

import 'package:aura_core/aura_core.dart';
import 'package:aurasight/data/database.dart';
import 'package:aurasight/services/audio_orchestrator.dart';
import 'package:aurasight/services/camera_service.dart';
import 'package:aurasight/services/detection_service.dart';
import 'package:aurasight/services/live_pipeline.dart';
import 'package:aurasight/services/scene_controller.dart';
import 'package:aurasight/services/scene_understanding_service.dart';
import 'package:aurasight/services/tracking_service.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';

class FakeTts implements TtsEngine {
  final spoken = <String>[];
  Duration duration = const Duration(milliseconds: 5);
  Completer<void>? _stop;
  @override
  Future<void> init() async {}
  @override
  Future<bool> speak(String text, {required double rate, required double volume, String? voiceId}) async {
    final stop = _stop = Completer<void>();
    final done = await Future.any([Future.delayed(duration, () => true), stop.future.then((_) => false)]);
    if (done) spoken.add(text);
    return done;
  }

  @override
  Future<void> interrupt() async {
    if (_stop != null && !_stop!.isCompleted) _stop!.complete();
  }
}

class FakeCamera implements CameraService {
  @override
  CameraFrame? latestFrame =
      CameraFrame(4, 4, 0, [FramePlane(Uint8List(16), 4, 1)], DateTime(2026));
  @override
  final state = ValueNotifier(const CameraState(CameraStatus.running));
  @override
  Stream<CameraFrame> get frames => const Stream.empty();
  @override
  Widget? buildPreview() => null;
  @override
  Future<void> start() async {}
  @override
  Future<void> stop() async {}
  @override
  Future<void> dispose() async {}
}

class NoDetector implements DetectionService {
  @override
  DetectorInfo get info => DetectorInfo.notLoaded;
  @override
  Future<DetectorInfo> load({int inputSize = 320, String backend = 'auto'}) async => info;
  @override
  Future<DetectionResult> detect(CameraFrame frame, {double conf = 0.1}) => throw UnimplementedError();
  @override
  Future<List<Detection>> detectCrop(CameraFrame f, (int, int, int, int) r, int s, {double conf = 0.05}) =>
      throw UnimplementedError();
  @override
  Future<void> dispose() async {}
}

/// A stand-in that streams a fixed reply, used only to test the controller's
/// streaming and de-duplication rules (the real model runs natively).
class ScriptedVlm implements SceneUnderstandingService {
  final String reply;
  ScriptedVlm(this.reply);
  @override
  VlmState get state => const VlmState(VlmStatus.ready, backend: 'test');
  @override
  Stream<VlmState> get states => const Stream.empty();
  @override
  Future<VlmState> load({required String modelPath, required String mmprojPath, int threads = 0}) async => state;
  @override
  Future<GenerationResult> generate(CameraFrame frame, String prompt,
      {int maxTokens = 80, Map<String, Object?> params = const {}, void Function(String delta)? onDelta}) async {
    for (final w in reply.split(' ')) {
      onDelta?.call('$w ');
      await Future<void>.delayed(Duration.zero);
    }
    return GenerationResult(reply, 0, 0, 0);
  }

  @override
  Future<void> cancel() async {}
  @override
  Future<void> unload() async {}
}

class UnloadedVlm extends ScriptedVlm {
  UnloadedVlm() : super('');
  @override
  VlmState get state => const VlmState(VlmStatus.unloaded);
}

ObjectView obj(int id, String label, List<double> b) {
  final box = Box.fromList(b);
  return ObjectView(id, label, 0.9, [b[0] / 640, b[1] / 480, b[2] / 640, b[3] / 480], analyzeBox(box, 640, 480, label), 5, 0);
}

Future<void> until(bool Function() c) async {
  for (var i = 0; i < 400 && !c(); i++) {
    await Future<void>.delayed(const Duration(milliseconds: 5));
  }
  expect(c(), isTrue);
}

void main() {
  late FakeTts tts;
  late SceneController scene;
  late AudioOrchestrator audio;
  var settings = const AuraSettings();

  SceneController build(SceneUnderstandingService vlm, List<ObjectView> objects) {
    tts = FakeTts();
    double clock() => DateTime.now().microsecondsSinceEpoch / 1e6;
    final queue = SpeechQueue(() => settings, tts, clock);
    audio = AudioOrchestrator(() => settings, queue, clock, log: (ActivityEntry _) async {}, haptic: (_) async {});
    final cam = FakeCamera();
    final live = LivePipeline(
      settings: () => settings,
      camera: cam,
      detector: NoDetector(),
      tracking: TrackingService(() => settings, clock),
      audio: audio,
      vlmBusy: () => false,
    );
    live.snapshot.value = LiveSnapshot(objects: objects);
    return SceneController(settings: () => settings, camera: cam, vlm: vlm, live: live, audio: audio);
  }

  setUp(() => settings = const AuraSettings());

  test('every sentence shown is spoken, and named objects are not repeated', () async {
    scene = build(
        ScriptedVlm('A woman is sitting at a desk. A laptop is open on your right. There is a cup next to it.'),
        [obj(1, 'person', [220, 20, 420, 478]), obj(2, 'laptop', [450, 200, 600, 300]), obj(3, 'chair', [20, 250, 120, 400])]);
    await audio.start();
    final v = await scene.describe();
    expect(v.source, 'vlm');
    expect(v.sentences.length, 4);
    expect(v.sentences.last, startsWith('The object detector also found a chair'));
    expect(v.text, isNot(contains('laptop slightly')));
    await until(() => tts.spoken.length == v.sentences.length);
    expect(tts.spoken, v.sentences);
    expect(v.speech, 'spoken');
  });

  test('without the scene model, falls back to the detector and says so', () async {
    scene = build(UnloadedVlm(), [obj(1, 'chair', [20, 250, 120, 400])]);
    await audio.start();
    final v = await scene.describe();
    expect(v.source, 'detector');
    expect(v.text, startsWith("The scene model isn't available. The object detector sees a chair"));
  });

  test('replay reads the same sentences; stop silences', () async {
    scene = build(ScriptedVlm('A kitchen with a table. A bottle stands on it.'), []);
    await audio.start();
    final v = await scene.describe();
    await until(() => tts.spoken.length == v.sentences.length);
    tts.spoken.clear();
    expect(scene.replay(), 'spoken');
    await until(() => tts.spoken.length == v.sentences.length);
    expect(tts.spoken, v.sentences);

    tts.duration = const Duration(seconds: 2);
    tts.spoken.clear();
    scene.replay();
    await until(() => audio.speech.isSpeaking);
    await scene.stop();
    await until(() => !audio.speech.isSpeaking);
    expect(tts.spoken, isEmpty);
  });

  test('Speak out loud off: shown, not spoken', () async {
    settings = const AuraSettings(speechEnabled: false);
    scene = build(ScriptedVlm('A quiet room with a sofa.'), []);
    await audio.start();
    final v = await scene.describe();
    expect(v.text, 'A quiet room with a sofa.');
    expect(v.speech, 'muted');
    expect(scene.replay(), 'muted');
    await Future<void>.delayed(const Duration(milliseconds: 50));
    expect(tts.spoken, isEmpty);
  });
}
