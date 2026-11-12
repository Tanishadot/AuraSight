import 'dart:async';

import 'package:flutter/foundation.dart';

import 'detection_service.dart';
import 'detector_model.dart';
import 'model_manager.dart';

enum DetectorPhase { checking, notDownloaded, notLoaded, downloading, verifying, loading, ready, error }

class DetectorStatus {
  final DetectorPhase phase;
  final int receivedBytes, totalBytes;
  final String? error; // what went wrong
  final String? recovery; // what the user can do about it
  final DetectorInfo? info; // set when ready
  final String? selfTest; // e.g. "person 0.84 in 95 ms"
  final double selfTestMs;

  /// The previous run of the app closed during this native step (from the crash journal).
  final String? crashedStage;
  const DetectorStatus(this.phase,
      {this.receivedBytes = 0,
      this.totalBytes = 0,
      this.error,
      this.recovery,
      this.info,
      this.selfTest,
      this.selfTestMs = 0,
      this.crashedStage});

  bool get ready => phase == DetectorPhase.ready;
  bool get busy =>
      phase == DetectorPhase.checking ||
      phase == DetectorPhase.downloading ||
      phase == DetectorPhase.verifying ||
      phase == DetectorPhase.loading;
  double get fraction => totalBytes == 0 ? 0 : receivedBytes / totalBytes;
}

class _NoGuard implements DetectorGuardService {
  const _NoGuard();
  @override
  Future<GuardStatus> status() async => GuardStatus.none;
  @override
  Future<void> markGood(String backend) async {}
  @override
  Future<void> clearGpuBlock() async {}
}

/// The single owner of the detector model's life cycle:
/// not downloaded -> downloading -> verifying -> (not loaded) -> loading -> ready (or error).
///
/// Stability rules (after the app closed during the self-test on an iQOO Z7s):
/// - The CPU is the default backend. The GPU is only tried on request ([tryGpu]),
///   and never again after a GPU step killed the process (native crash journal).
/// - On start-up the model is only loaded automatically if the previous load and
///   self-test completed and the previous process didn't die in a detector step.
///   Otherwise the user loads it with a button, so a crash can't repeat on every launch.
/// - One load at a time; the self-test runs once per load, never on a timer.
/// - Every failure ends in a visible error with Retry, never an exception.
class DetectorController {
  final DetectionService detector;
  final ModelManager models;
  final DetectorManifest? manifest;
  final Future<Uint8List> Function() selfTestImage;
  final DetectorGuardService guard;

  /// Backend for normal loads.
  final String backend;

  final status = ValueNotifier(const DetectorStatus(DetectorPhase.checking));
  StreamSubscription<ModelProgress>? _download;
  Future<DetectorStatus>? _loading;
  GuardStatus _guard = GuardStatus.none;

  /// Minimum confidence for the self-test's expected detection.
  static const selfTestLabel = 'person';
  static const selfTestMinScore = 0.5;

  DetectorController({
    required this.detector,
    required this.models,
    required this.manifest,
    required this.selfTestImage,
    DetectorGuardService? guard,
    this.backend = 'cpu',
  }) : guard = guard ?? const _NoGuard();

  GuardStatus get guardStatus => _guard;

  void _set(DetectorStatus s) => status.value = s;

  DetectorStatus _fail(String error, String recovery, {String? crashedStage}) {
    final s = DetectorStatus(DetectorPhase.error, error: error, recovery: recovery, crashedStage: crashedStage);
    _set(s);
    return s;
  }

  DetectorStatus _idle() {
    final s = DetectorStatus(DetectorPhase.notLoaded, totalBytes: manifest?.file.bytes ?? 0);
    _set(s);
    return s;
  }

  /// On start-up. Loads straight away only when that is known to be safe.
  Future<DetectorStatus> init() async {
    final m = manifest;
    if (m == null) {
      return _fail('The detector description is missing from this build.', 'Install a newer build of AuraSight.');
    }
    _set(const DetectorStatus(DetectorPhase.checking));
    try {
      _guard = await guard.status();
      final crashed = _guard.interruptedStage;
      if (crashed != null) {
        final onGpu = _guard.interruptedBackend == 'gpu';
        debugPrint('AuraDetector: previous run closed during $crashed (${_guard.interruptedBackend})');
        return _fail(
          'AuraSight closed unexpectedly last time, during ${describeStage(crashed)}'
          '${onGpu ? ' on the GPU' : ''}.',
          onGpu
              ? 'The GPU is now switched off for the detector. Press Retry to load it on the CPU.'
              : 'Press Retry to try again. If it closes again, open Settings → Detector diagnostics '
                  'and copy the crash report.',
          crashedStage: crashed,
        );
      }
      if (!await models.isInstalled(m.spec)) {
        _set(DetectorStatus(DetectorPhase.notDownloaded, totalBytes: m.file.bytes));
        return status.value;
      }
      final good = _guard.lastGood;
      if (good != null) return await load(backend: good == 'gpu' ? 'auto' : 'cpu');
      return _idle();
    } catch (e) {
      return _fail('Could not check the object detector: $e', 'Press Retry.');
    }
  }

  /// Downloads (resuming a partial download) and verifies the SHA-256. Loading is a separate step.
  Future<DetectorStatus> download() {
    final m = manifest;
    if (m == null) return init();
    final done = Completer<DetectorStatus>();
    _download?.cancel();
    _set(DetectorStatus(DetectorPhase.downloading, totalBytes: m.file.bytes));
    _download = models.install(m.spec).listen((p) {
      switch (p.status) {
        case ModelStatus.downloading:
          _set(DetectorStatus(DetectorPhase.downloading, receivedBytes: p.receivedBytes, totalBytes: p.totalBytes));
        case ModelStatus.verifying:
          _set(DetectorStatus(DetectorPhase.verifying, receivedBytes: p.receivedBytes, totalBytes: p.totalBytes));
        case ModelStatus.installed:
          if (!done.isCompleted) done.complete(_idle()); // loading is a separate, user-started step
        case ModelStatus.failed:
          if (done.isCompleted) return;
          if ((p.error ?? '').contains('cancelled')) {
            final s = DetectorStatus(DetectorPhase.notDownloaded, totalBytes: m.file.bytes);
            _set(s);
            done.complete(s);
          } else {
            done.complete(_fail(p.error ?? 'The download failed.',
                'Check the internet connection and try again. A partial download resumes where it stopped.'));
          }
        case ModelStatus.notInstalled:
          break;
      }
    }, onError: (Object e) {
      if (!done.isCompleted) {
        done.complete(_fail('The download failed: $e', 'Check the internet connection and retry.'));
      }
    });
    return done.future;
  }

  void cancelDownload() {
    final m = manifest;
    if (m != null) models.cancel(m.spec);
  }

  /// Verifies the file, loads it natively and runs the self-test once.
  /// Concurrent calls share one load.
  Future<DetectorStatus> load({String? backend}) =>
      _loading ??= _load(backend ?? this.backend).whenComplete(() => _loading = null);

  /// Opt-in GPU attempt: GPU if it works, otherwise the CPU with the reason.
  /// If the GPU crashes the process, the next start reports it and keeps the GPU off.
  Future<DetectorStatus> tryGpu() async {
    await detector.dispose();
    return load(backend: 'auto');
  }

  Future<DetectorStatus> _load(String backend) async {
    final m = manifest;
    if (m == null) return init();
    try {
      _set(const DetectorStatus(DetectorPhase.verifying));
      if (!await models.verifyInstalled(m.spec)) {
        await models.delete(m.spec);
        return _fail('The detector file on this phone is damaged and was deleted.', 'Download it again.');
      }
      _set(const DetectorStatus(DetectorPhase.loading));
      final path = models.fileFor(m.spec, m.file).path;
      final info = await detector.load(path, m, backend: backend);
      debugPrint('AuraDetector: load backend=${info.backend} available=${info.available} '
          'loadMs=${info.loadMs.toStringAsFixed(0)} warmupMs=${info.warmupMs.toStringAsFixed(0)} '
          'cached=${info.cached} fallback=${info.fallbackReason} ${info.memory} reason=${info.reason}');
      if (!info.available) {
        return _fail(info.reason ?? 'The detector could not be loaded.',
            'Press Retry. If it keeps failing, delete the model in Settings and download it again.');
      }
      return await _selfTest(m, info);
    } catch (e) {
      await _safeDispose();
      return _fail('The detector could not be started: $e', 'Press Retry.');
    }
  }

  /// One real inference on a known photo; it must find a person.
  Future<DetectorStatus> _selfTest(DetectorManifest m, DetectorInfo info) async {
    try {
      final sw = Stopwatch()..start();
      final r = await detector.detectImage(await selfTestImage(), conf: 0.25);
      sw.stop();
      String label(int c) => c < m.labels.length ? m.labels[c] : 'class $c';
      final hits = [
        for (final d in r.detections)
          if (label(d.cls) == selfTestLabel && d.score >= selfTestMinScore) d
      ];
      debugPrint('AuraDetector: self-test ${hits.length} person(s), inference ${r.inferenceMs.toStringAsFixed(0)} ms, '
          'round trip ${sw.elapsedMilliseconds} ms, ${r.memory}');
      if (hits.isEmpty) {
        await _safeDispose();
        final seen = r.detections.map((d) => '${label(d.cls)} ${d.score.toStringAsFixed(2)}').join(', ');
        return _fail(
            'The detector loaded but failed its self-test (expected a person, found: ${seen.isEmpty ? 'nothing' : seen}).',
            'Delete the model in Settings and download it again.');
      }
      final best = hits.reduce((a, b) => a.score >= b.score ? a : b);
      await guard.markGood(info.backend);
      _guard = await guard.status();
      final s = DetectorStatus(DetectorPhase.ready,
          info: info,
          selfTestMs: r.inferenceMs,
          selfTest: '$selfTestLabel ${best.score.toStringAsFixed(2)} in ${r.inferenceMs.toStringAsFixed(0)} ms');
      _set(s);
      return s;
    } on DetectorFailure catch (e) {
      await _safeDispose();
      return _fail(
          e.outOfMemory
              ? 'The phone ran out of memory during the self-test. ${e.message}'
              : 'The detector loaded but could not run: ${e.message}',
          e.outOfMemory ? 'Close other apps, then press Retry.' : 'Press Retry, or delete and re-download the model.');
    } catch (e) {
      await _safeDispose();
      return _fail('The detector loaded but could not run: $e', 'Press Retry, or delete and re-download the model.');
    }
  }

  Future<void> _safeDispose() async {
    try {
      await detector.dispose();
    } catch (_) {}
  }

  /// Retry from wherever it failed: load (CPU) if the file is present, otherwise download.
  Future<DetectorStatus> retry() async {
    final m = manifest;
    if (m == null) return init();
    try {
      return await models.isInstalled(m.spec) ? await load() : await download();
    } catch (e) {
      return _fail('Retry failed: $e', 'Press Retry again.');
    }
  }

  /// Allows the GPU again after it was switched off by a crash.
  Future<void> clearGpuBlock() async {
    await guard.clearGpuBlock();
    _guard = await guard.status();
  }

  Future<void> deleteModel() async {
    final m = manifest;
    if (m == null) return;
    await _safeDispose();
    await models.delete(m.spec);
    _set(DetectorStatus(DetectorPhase.notDownloaded, totalBytes: m.file.bytes));
  }

  void dispose() {
    _download?.cancel();
    status.dispose();
  }
}
