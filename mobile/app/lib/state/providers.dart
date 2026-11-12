import 'dart:async';

import 'package:aura_core/aura_core.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../data/database.dart';
import '../services/audio_orchestrator.dart';
import '../services/camera_lifecycle.dart';
import '../services/camera_service.dart';
import '../services/detection_service.dart';
import '../services/detector_controller.dart';
import '../services/detector_model.dart';
import '../services/live_pipeline.dart';
import '../services/model_manager.dart';
import '../services/scene_controller.dart';
import '../services/scene_understanding_service.dart';
import '../services/speech_service.dart';
import '../services/tracking_service.dart';

/// Overridden in main() once the database is open.
final databaseProvider = Provider<AuraDatabase>((ref) => throw UnimplementedError('databaseProvider'));
final initialSettingsProvider = Provider<AuraSettings>((ref) => const AuraSettings());
final modelManagerProvider = Provider<ModelManager>((ref) => throw UnimplementedError('modelManagerProvider'));

/// The detector manifest bundled with the app (null if the asset is missing).
final detectorManifestProvider = Provider<DetectorManifest?>((ref) => null);

/// Public-domain photo of a person, used for the detector's self-test.
const selfTestAsset = 'assets/selftest/person_living_room.jpg';

/// Monotonic clock in seconds, shared by all time-based logic.
final clockProvider = Provider<double Function()>((ref) {
  final sw = Stopwatch()..start();
  return () => sw.elapsedMicroseconds / 1e6;
});

class SettingsNotifier extends Notifier<AuraSettings> {
  @override
  AuraSettings build() => ref.read(initialSettingsProvider);

  Future<void> patch(Map<String, Object?> changes) async {
    state = state.merge(changes);
    ref.read(trackingServiceProvider).applySettings(state);
    if (changes['speechEnabled'] == false) unawaited(ref.read(audioProvider).stop());
    await ref.read(databaseProvider).saveSettings(state);
  }
}

final settingsProvider = NotifierProvider<SettingsNotifier, AuraSettings>(SettingsNotifier.new);

/// Bumped whenever the activity log changes, so lists refresh.
class ActivityTick extends Notifier<int> {
  @override
  int build() => 0;
  void bump() => state++;
}

final activityTickProvider = NotifierProvider<ActivityTick, int>(ActivityTick.new);

final cameraServiceProvider = Provider<CameraService>((ref) {
  final c = PluginCameraService();
  ref.onDispose(c.dispose);
  return c;
});

final detectionServiceProvider = Provider<DetectionService>((ref) => NativeDetectionService());
final vlmServiceProvider = Provider<SceneUnderstandingService>((ref) => NativeVlmService());
final ttsEngineProvider = Provider<NativeTtsEngine>((ref) => NativeTtsEngine());

final detectorControllerProvider = Provider<DetectorController>((ref) {
  final c = DetectorController(
    detector: ref.read(detectionServiceProvider),
    models: ref.read(modelManagerProvider),
    manifest: ref.read(detectorManifestProvider),
    selfTestImage: () async => (await rootBundle.load(selfTestAsset)).buffer.asUint8List(),
  );
  ref.onDispose(c.dispose);
  return c;
});

final cameraLifecycleProvider = Provider<CameraLifecycle>(
    (ref) => CameraLifecycle(ref.read(cameraServiceProvider), ref.read(livePipelineProvider)));

final trackingServiceProvider =
    Provider<TrackingService>((ref) => TrackingService(() => ref.read(settingsProvider), ref.read(clockProvider)));

final audioProvider = Provider<AudioOrchestrator>((ref) {
  AuraSettings settings() => ref.read(settingsProvider);
  final clock = ref.read(clockProvider);
  final queue = SpeechQueue(settings, ref.read(ttsEngineProvider), clock);
  final db = ref.read(databaseProvider);
  final audio = AudioOrchestrator(settings, queue, clock, log: (e) async {
    await db.addActivity(e);
    ref.read(activityTickProvider.notifier).bump();
  });
  unawaited(audio.start());
  ref.onDispose(queue.shutdown);
  return audio;
});

final livePipelineProvider = Provider<LivePipeline>((ref) {
  final vlm = ref.read(vlmServiceProvider);
  final p = LivePipeline(
    settings: () => ref.read(settingsProvider),
    camera: ref.read(cameraServiceProvider),
    detector: ref.read(detectionServiceProvider),
    tracking: ref.read(trackingServiceProvider),
    audio: ref.read(audioProvider),
    vlmBusy: () => vlm.state.status == VlmStatus.busy,
  );
  ref.onDispose(p.dispose);
  return p;
});

final sceneControllerProvider = Provider<SceneController>((ref) {
  final c = SceneController(
    settings: () => ref.read(settingsProvider),
    camera: ref.read(cameraServiceProvider),
    vlm: ref.read(vlmServiceProvider),
    live: ref.read(livePipelineProvider),
    audio: ref.read(audioProvider),
  );
  ref.onDispose(c.dispose);
  return c;
});

/// Speech queue events (speaking / idle / error), for indicators.
final speechEventsProvider = StreamProvider<SpeechEvent>((ref) => ref.read(audioProvider).speech.events);
