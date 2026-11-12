import 'dart:async';

import 'package:flutter/foundation.dart';

import 'detection_service.dart';
import 'detector_model.dart';
import 'model_manager.dart';

enum DetectorPhase { checking, notDownloaded, downloading, verifying, loading, ready, error }

class DetectorStatus {
  final DetectorPhase phase;
  final int receivedBytes, totalBytes;
  final String? error; // what went wrong
  final String? recovery; // what the user can do about it
  final DetectorInfo? info; // set when ready
  final String? selfTest; // e.g. "person 0.84"
  const DetectorStatus(this.phase,
      {this.receivedBytes = 0, this.totalBytes = 0, this.error, this.recovery, this.info, this.selfTest});

  bool get ready => phase == DetectorPhase.ready;
  bool get busy =>
      phase == DetectorPhase.checking ||
      phase == DetectorPhase.downloading ||
      phase == DetectorPhase.verifying ||
      phase == DetectorPhase.loading;
  double get fraction => totalBytes == 0 ? 0 : receivedBytes / totalBytes;
}

/// The single owner of the detector model's life cycle:
/// not downloaded -> downloading -> verifying -> loading -> ready (or error).
///
/// "Ready" is only reported after the native model has loaded *and* found a
/// person in a bundled public-domain photo (the self-test). The SHA-256 is
/// checked after download and again before every load, without network access.
class DetectorController {
  final DetectionService detector;
  final ModelManager models;
  final DetectorManifest? manifest;
  final Future<Uint8List> Function() selfTestImage;
  final String backend;

  final status = ValueNotifier(const DetectorStatus(DetectorPhase.checking));
  StreamSubscription<ModelProgress>? _download;

  /// Minimum confidence for the self-test's expected detection.
  static const selfTestLabel = 'person';
  static const selfTestMinScore = 0.5;

  DetectorController({
    required this.detector,
    required this.models,
    required this.manifest,
    required this.selfTestImage,
    this.backend = 'auto',
  });

  void _set(DetectorStatus s) => status.value = s;

  DetectorStatus _fail(String error, String recovery) {
    final s = DetectorStatus(DetectorPhase.error, error: error, recovery: recovery);
    _set(s);
    return s;
  }

  /// On start-up: load straight away if the model is already on the phone.
  Future<DetectorStatus> init() async {
    final m = manifest;
    if (m == null) {
      return _fail('The detector description is missing from this build.', 'Install a newer build of AuraSight.');
    }
    _set(const DetectorStatus(DetectorPhase.checking));
    if (await models.isInstalled(m.spec)) return load();
    _set(DetectorStatus(DetectorPhase.notDownloaded, totalBytes: m.file.bytes));
    return status.value;
  }

  /// Downloads (resuming a partial download), verifies, then loads.
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
          load().then(done.complete);
        case ModelStatus.failed:
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
    });
    return done.future;
  }

  void cancelDownload() {
    final m = manifest;
    if (m != null) models.cancel(m.spec);
  }

  /// Verifies the file, loads it natively and runs the self-test.
  Future<DetectorStatus> load() async {
    final m = manifest!;
    _set(const DetectorStatus(DetectorPhase.verifying));
    if (!await models.verifyInstalled(m.spec)) {
      await models.delete(m.spec);
      return _fail('The detector file on this phone is damaged and was deleted.', 'Download it again.');
    }
    _set(const DetectorStatus(DetectorPhase.loading));
    final path = models.fileFor(m.spec, m.file).path;
    final info = await detector.load(path, m, backend: backend);
    if (!info.available) {
      return _fail(info.reason ?? 'The detector could not be loaded.',
          'Try again. If it keeps failing, delete the model in Settings and download it again.');
    }
    // Self-test: a real inference on a known photo must find a person.
    try {
      final r = await detector.detectImage(await selfTestImage(), conf: 0.25);
      final hits = [
        for (final d in r.detections)
          if (d.cls < m.labels.length && m.labels[d.cls] == selfTestLabel && d.score >= selfTestMinScore) d
      ];
      if (hits.isEmpty) {
        await detector.dispose();
        final seen = r.detections.map((d) => '${m.labels[d.cls]} ${d.score.toStringAsFixed(2)}').join(', ');
        return _fail(
            'The detector loaded but failed its self-test (expected a person, found: ${seen.isEmpty ? 'nothing' : seen}).',
            'Delete the model in Settings and download it again.');
      }
      final best = hits.reduce((a, b) => a.score >= b.score ? a : b);
      final s = DetectorStatus(DetectorPhase.ready,
          info: info,
          selfTest: '$selfTestLabel ${best.score.toStringAsFixed(2)} in ${r.inferenceMs.toStringAsFixed(0)} ms');
      _set(s);
      return s;
    } catch (e) {
      await detector.dispose();
      return _fail('The detector loaded but could not run: $e', 'Try again, or delete and re-download the model.');
    }
  }

  /// Retry from wherever it failed: load if the file is present, otherwise download.
  Future<DetectorStatus> retry() async {
    final m = manifest;
    if (m == null) return init();
    return await models.isInstalled(m.spec) ? load() : download();
  }

  Future<void> deleteModel() async {
    final m = manifest;
    if (m == null) return;
    await detector.dispose();
    await models.delete(m.spec);
    _set(DetectorStatus(DetectorPhase.notDownloaded, totalBytes: m.file.bytes));
  }

  void dispose() {
    _download?.cancel();
    status.dispose();
  }
}
