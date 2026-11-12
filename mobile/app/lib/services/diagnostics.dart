import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import '../platform/channels.dart';
import 'benchmark.dart';
import 'camera_service.dart';
import 'detection_service.dart';
import 'detector_controller.dart';
import 'detector_model.dart';
import 'model_manager.dart';

/// The staged hardware check, in order. Each stage only runs after the
/// previous one passed in this session, so a crash points at one step.
enum DiagStage {
  launch('1. App running, model not loaded'),
  loadModel('2. Load the model (CPU, camera off)'),
  testImage('3. One inference on the bundled test photo'),
  cameraFrame('4. One inference on a single camera frame'),
  continuous('5. Continuous camera inference, 1 per second'),
  fullSelfTest('6. Full detector start-up with self-test');

  final String title;
  const DiagStage(this.title);
}

class StageResult {
  final DiagStage stage;
  final bool ok;
  final String summary;
  final LatencyStats latency; // detector-reported total inference time, ms
  final SystemSnapshot before, after;
  final NativeMemory native;
  final Duration took;
  const StageResult(this.stage, this.ok, this.summary,
      {required this.latency,
      required this.before,
      required this.after,
      this.native = const NativeMemory(),
      required this.took});

  String _mem(SystemSnapshot s) => 'PSS ${s.appMemMb ?? '?'} MB, free ${s.availMemMb ?? '?'} MB';

  String get memoryLine => '${_mem(before)} → ${_mem(after)}; $native';

  String get latencyLine => latency.n == 0
      ? 'no inferences'
      : 'n=${latency.n}, median ${latency.median.toStringAsFixed(0)} ms, p95 ${latency.p95.toStringAsFixed(0)} ms, '
          'max ${latency.max.toStringAsFixed(0)} ms';

  /// One line for logcat (`adb logcat -s flutter | grep AuraDiag`).
  String get logLine => 'AuraDiag: stage=${stage.name} ok=$ok took=${took.inMilliseconds}ms '
      'latency=[$latencyLine] memory=[$memoryLine] summary=$summary';
}

/// Why previous runs of the app ended (Android's ApplicationExitInfo) plus the
/// detector crash journal; the trace of a native crash is the full tombstone.
class CrashReport {
  final GuardStatus guard;
  final List<Map<String, Object?>> exits;
  const CrashReport(this.guard, this.exits);

  static Future<CrashReport> read(DetectorGuardService guard, [MethodChannel? system]) async {
    var exits = <Map<String, Object?>>[];
    try {
      final r = await (system ?? AuraChannels.system).invokeListMethod<Object?>('exitInfo') ?? const [];
      exits = [for (final e in r) (e as Map).cast<String, Object?>()];
    } on MissingPluginException {
      // not available on this platform
    } on PlatformException {
      // ignore
    }
    return CrashReport(await guard.status(), exits);
  }

  bool get hasCrash =>
      guard.interruptedStage != null ||
      exits.any((e) =>
          const {'native_crash', 'java_crash', 'java_crash_file', 'anr', 'low_memory_kill'}.contains(e['reason']));

  String toText() {
    final b = StringBuffer('# AuraSight crash report\n\n');
    if (guard.interruptedStage != null) {
      b.writeln('Detector journal: the previous run closed during `${guard.interruptedStage}` '
          '(${describeStage(guard.interruptedStage!)}) on ${guard.interruptedBackend} at ${guard.interruptedAt}.');
    } else {
      b.writeln('Detector journal: no interrupted detector step.');
    }
    if (guard.gpuBlocked != null) b.writeln('GPU switched off: ${guard.gpuBlocked}');
    b.writeln('Last good backend: ${guard.lastGood ?? 'none'}\n');
    if (exits.isEmpty) b.writeln('No recorded process exits (Android 11+ only).');
    for (final e in exits) {
      final t =
          e['timestampMs'] == null ? '?' : DateTime.fromMillisecondsSinceEpoch((e['timestampMs']! as num).toInt());
      b.writeln('## ${e['reason']} at $t');
      b.writeln('description: ${e['description']}; status: ${e['status']}; '
          'PSS ${e['pssMb']} MB; RSS ${e['rssMb']} MB; importance ${e['importance']}');
      if (e['tracePath'] != null) b.writeln('full trace saved on the phone: ${e['tracePath']}');
      if (e['traceHead'] != null) b.writeln('```\n${e['traceHead']}\n```');
      b.writeln();
    }
    return b.toString();
  }
}

/// Runs the staged hardware check one stage at a time, recording latency and memory.
class DiagnosticsRunner {
  final DetectionService detector;
  final DetectorController controller;
  final ModelManager models;
  final DetectorManifest? manifest;
  final CameraService camera;

  /// Stops live detection and the camera (CameraLifecycle.stop).
  final Future<void> Function() stopLive;
  final Future<Uint8List> Function() testImage;
  final Future<SystemSnapshot> Function() snapshot;
  final Duration continuousFor;
  final Duration frameTimeout;

  final results = ValueNotifier<Map<DiagStage, StageResult>>(const {});
  final running = ValueNotifier<DiagStage?>(null);

  DiagnosticsRunner({
    required this.detector,
    required this.controller,
    required this.models,
    required this.manifest,
    required this.camera,
    required this.stopLive,
    required this.testImage,
    Future<SystemSnapshot> Function()? snapshot,
    this.continuousFor = const Duration(seconds: 20),
    this.frameTimeout = const Duration(seconds: 10),
  }) : snapshot = snapshot ?? SystemSnapshot.read;

  /// A stage can run once the one before it passed (stage 1 always).
  bool canRun(DiagStage s) =>
      running.value == null && (s.index == 0 || (results.value[DiagStage.values[s.index - 1]]?.ok ?? false));

  Future<StageResult> run(DiagStage stage) async {
    if (!canRun(stage)) throw StateError('Run "${DiagStage.values[stage.index - 1].title}" first.');
    running.value = stage;
    final sw = Stopwatch()..start();
    final before = await snapshot();
    final lat = <double>[];
    var native = const NativeMemory();
    bool ok;
    String summary;
    try {
      (ok, summary, native) = await _run(stage, lat);
    } catch (e) {
      ok = false;
      summary = 'failed: $e';
    }
    final r = StageResult(stage, ok, summary,
        latency: LatencyStats.of(lat), before: before, after: await snapshot(), native: native, took: sw.elapsed);
    debugPrint(r.logLine);
    results.value = {...results.value, stage: r};
    running.value = null;
    return r;
  }

  Future<(bool, String, NativeMemory)> _run(DiagStage stage, List<double> lat) async {
    final m = manifest;
    switch (stage) {
      case DiagStage.launch:
        return (true, 'app is running; detector ${controller.status.value.phase.name}', const NativeMemory());
      case DiagStage.loadModel:
        if (m == null) return (false, 'detector manifest missing from this build', const NativeMemory());
        await stopLive(); // camera off for this stage
        if (!await models.isInstalled(m.spec)) return (false, 'model not downloaded yet', const NativeMemory());
        if (!await models.verifyInstalled(m.spec)) {
          return (false, 'model file failed its SHA-256 check', const NativeMemory());
        }
        final info = await detector.load(models.fileFor(m.spec, m.file).path, m, backend: 'cpu');
        if (!info.available) return (false, 'load failed: ${info.reason}', info.memory);
        lat.add(info.warmupMs);
        return (
          true,
          '${info.backend.toUpperCase()}${info.cached ? ' (already loaded)' : ''}, load ${info.loadMs.toStringAsFixed(0)} ms, '
              'warm-up ${info.warmupMs.toStringAsFixed(0)} ms',
          info.memory
        );
      case DiagStage.testImage:
        final r = await detector.detectImage(await testImage(), conf: 0.25);
        lat.add(r.inferenceMs);
        final names = m?.labels ?? const <String>[];
        final found = r.detections
            .map((d) => '${d.cls < names.length ? names[d.cls] : d.cls} ${d.score.toStringAsFixed(2)}')
            .join(', ');
        final person = r.detections.any((d) =>
            d.cls < names.length &&
            names[d.cls] == DetectorController.selfTestLabel &&
            d.score >= DetectorController.selfTestMinScore);
        return (
          person,
          '${r.width}x${r.height}: ${found.isEmpty ? 'nothing' : found} '
              '(pre ${r.preMs.toStringAsFixed(0)} / model ${r.modelMs.toStringAsFixed(0)} / post ${r.postMs.toStringAsFixed(0)} ms)'
              '${person ? '' : ' - expected a person'}',
          r.memory
        );
      case DiagStage.cameraFrame:
        return _withCamera(() async {
          final f = await camera.frames.first.timeout(frameTimeout);
          final r = await detector.detect(f, conf: 0.25);
          lat.add(r.inferenceMs);
          return (
            true,
            'frame ${f.width}x${f.height} rot ${f.rotation}: ${r.detections.length} detections, '
                'pre ${r.preMs.toStringAsFixed(0)} / model ${r.modelMs.toStringAsFixed(0)} ms',
            r.memory
          );
        });
      case DiagStage.continuous:
        return _withCamera(() async {
          final end = DateTime.now().add(continuousFor);
          var errors = 0, consecutive = 0, busy = 0;
          var native = const NativeMemory();
          String? lastError;
          while (DateTime.now().isBefore(end)) {
            final tick = DateTime.now();
            try {
              final f = await camera.frames.first.timeout(frameTimeout);
              final r = await detector.detect(f, conf: 0.25);
              lat.add(r.inferenceMs);
              native = r.memory;
              consecutive = 0;
            } on DetectorBusy {
              busy++;
            } catch (e) {
              errors++;
              consecutive++;
              lastError = '$e';
              if (consecutive >= 3) break;
            }
            final wait = const Duration(seconds: 1) - DateTime.now().difference(tick);
            if (wait > Duration.zero) await Future<void>.delayed(wait);
          }
          final ok = consecutive < 3 && lat.isNotEmpty;
          return (
            ok,
            '${lat.length} inferences in ${continuousFor.inSeconds} s, $errors errors, $busy busy'
                '${lastError != null ? ', last error: $lastError' : ''}',
            native
          );
        });
      case DiagStage.fullSelfTest:
        final s = await controller.load();
        if (s.selfTestMs > 0) lat.add(s.selfTestMs);
        return (
          s.ready,
          s.ready ? 'ready on ${s.info!.backend.toUpperCase()}, self-test ${s.selfTest}' : '${s.error}',
          s.info?.memory ?? const NativeMemory()
        );
    }
  }

  /// Starts the camera without live detection for the stage, and stops it afterwards.
  Future<(bool, String, NativeMemory)> _withCamera(Future<(bool, String, NativeMemory)> Function() body) async {
    await stopLive(); // live detection off: the stage is the only detector user
    await camera.start();
    try {
      if (camera.state.value.status != CameraStatus.running) {
        return (
          false,
          'camera did not start: ${camera.state.value.message ?? camera.state.value.status.name}',
          const NativeMemory()
        );
      }
      return await body();
    } finally {
      await camera.stop();
    }
  }

  /// Copyable report of all stages run so far.
  String report(SystemSnapshot device) {
    final b = StringBuffer('# AuraSight detector hardware check\n\n')
      ..writeln('- Device: ${device.device ?? '?'} · ${device.soc ?? '?'} · ${device.os ?? '?'}')
      ..writeln('- RAM: ${device.totalMemMb ?? '?'} MB total')
      ..writeln('- Camera frames: ${PluginCameraService.resolution.name} resolution\n')
      ..writeln('| Stage | Result | Latency | Memory (app PSS, free) | Took |')
      ..writeln('|---|---|---|---|---|');
    for (final s in DiagStage.values) {
      final r = results.value[s];
      if (r == null) continue;
      b.writeln('| ${s.title} | ${r.ok ? 'PASS' : 'FAIL'}: ${r.summary} | ${r.latencyLine} | ${r.memoryLine} | '
          '${(r.took.inMilliseconds / 1000).toStringAsFixed(1)} s |');
    }
    return b.toString();
  }

  void dispose() {
    results.dispose();
    running.dispose();
  }
}
