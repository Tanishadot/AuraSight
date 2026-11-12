import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:aura_core/aura_core.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import '../platform/channels.dart';
import 'camera_service.dart';
import 'detection_service.dart';
import 'detector_controller.dart';
import 'live_pipeline.dart';

/// Device readings from the native `aurasight/system` channel. Fields are null
/// when the platform doesn't expose them.
class SystemSnapshot {
  final String? device, os, soc, thermal;
  final int? availMemMb, appMemMb, batteryPct, batteryCurrentUa;
  final double? batteryTempC;
  final bool? charging;
  final DateTime time;
  const SystemSnapshot(this.time,
      {this.device,
      this.os,
      this.soc,
      this.thermal,
      this.availMemMb,
      this.appMemMb,
      this.batteryPct,
      this.batteryCurrentUa,
      this.batteryTempC,
      this.charging});

  static Future<SystemSnapshot> read([MethodChannel? channel]) async {
    try {
      final r = await (channel ?? AuraChannels.system).invokeMapMethod<String, Object?>('snapshot') ?? const {};
      return SystemSnapshot(DateTime.now(),
          device: r['device'] as String?,
          os: r['os'] as String?,
          soc: r['soc'] as String?,
          thermal: r['thermal'] as String?,
          availMemMb: r['availMemMb'] as int?,
          appMemMb: r['appMemMb'] as int?,
          batteryPct: r['batteryPct'] as int?,
          batteryCurrentUa: r['batteryCurrentUa'] as int?,
          batteryTempC: (r['batteryTempC'] as num?)?.toDouble(),
          charging: r['charging'] as bool?);
    } on MissingPluginException {
      return SystemSnapshot(DateTime.now());
    }
  }

  Map<String, Object?> toJson() => {
        'time': time.toIso8601String(),
        'device': device,
        'os': os,
        'soc': soc,
        'thermal': thermal,
        'availMemMb': availMemMb,
        'appMemMb': appMemMb,
        'batteryPct': batteryPct,
        'batteryCurrentUa': batteryCurrentUa,
        'batteryTempC': batteryTempC,
        'charging': charging,
      };
}

class LatencyStats {
  final int n;
  final double avg, median, p95, min, max;
  const LatencyStats(this.n, this.avg, this.median, this.p95, this.min, this.max);

  factory LatencyStats.of(List<double> v) {
    if (v.isEmpty) return const LatencyStats(0, 0, 0, 0, 0, 0);
    final s = [...v]..sort();
    double pct(double q) => s[math.min(s.length - 1, (q * (s.length - 1)).round())];
    return LatencyStats(s.length, s.reduce((a, b) => a + b) / s.length, pct(0.5), pct(0.95), s.first, s.last);
  }

  Map<String, Object?> toJson() => {'n': n, 'avg': avg, 'median': median, 'p95': p95, 'min': min, 'max': max};
  String row(String name) => '| $name | ${avg.toStringAsFixed(1)} | ${median.toStringAsFixed(1)} | '
      '${p95.toStringAsFixed(1)} | ${min.toStringAsFixed(1)} | ${max.toStringAsFixed(1)} |';
}

class BenchmarkReport {
  final DetectorStatus detector;
  final String source; // camera | test image
  final LatencyStats total, pre, model, post;
  final Duration sustained;
  final int sustainedFrames;
  final double targetFps;
  final LatencyStats sustainedLatency;
  final SystemSnapshot before, after;
  const BenchmarkReport({
    required this.detector,
    required this.source,
    required this.total,
    required this.pre,
    required this.model,
    required this.post,
    required this.sustained,
    required this.sustainedFrames,
    required this.targetFps,
    required this.sustainedLatency,
    required this.before,
    required this.after,
  });

  double get sustainedFps => sustained.inMilliseconds == 0 ? 0 : sustainedFrames * 1000 / sustained.inMilliseconds;

  Map<String, Object?> toJson() => {
        'backend': detector.info?.backend,
        'fallbackReason': detector.info?.fallbackReason,
        'loadMs': detector.info?.loadMs,
        'selfTest': detector.selfTest,
        'source': source,
        'burst': {'total': total.toJson(), 'pre': pre.toJson(), 'model': model.toJson(), 'post': post.toJson()},
        'sustained': {
          'seconds': sustained.inMilliseconds / 1000,
          'frames': sustainedFrames,
          'fps': sustainedFps,
          'targetFps': targetFps,
          'latency': sustainedLatency.toJson(),
        },
        'before': before.toJson(),
        'after': after.toJson(),
      };

  String _v(Object? v, [String unit = '']) => v == null ? 'not available' : '$v$unit';

  String toMarkdown() {
    final i = detector.info;
    final dT = (before.batteryTempC != null && after.batteryTempC != null)
        ? '${(after.batteryTempC! - before.batteryTempC!).toStringAsFixed(1)} °C'
        : 'not available';
    final dB = (before.batteryPct != null && after.batteryPct != null)
        ? '${after.batteryPct! - before.batteryPct!} %'
        : 'not available';
    return [
      '# AuraSight detector benchmark',
      '',
      '- Date: ${after.time.toIso8601String()}',
      '- Device: ${_v(before.device)} (${_v(before.soc)}), ${_v(before.os)}',
      '- Model: YOLO11n, 320×320 input, LiteRT',
      '- Delegate: ${i?.backend.toUpperCase() ?? '?'}${i?.fallbackReason != null ? ' (GPU not used: ${i!.fallbackReason})' : ''}',
      '- Model load time: ${i?.loadMs.toStringAsFixed(0) ?? '?'} ms; self-test: ${detector.selfTest}',
      '- Input: $source',
      '',
      '## Latency, back-to-back (${total.n} frames, ms)',
      '',
      '| Stage | Average | Median | p95 | Min | Max |',
      '|---|---|---|---|---|---|',
      total.row('Total'),
      pre.row('Preprocessing'),
      model.row('Model'),
      post.row('Decoding + NMS'),
      '',
      '## Sustained run',
      '',
      '- Duration: ${(sustained.inMilliseconds / 1000).toStringAsFixed(0)} s at a target of ${targetFps.toStringAsFixed(1)} detections per second',
      '- Achieved: ${sustainedFps.toStringAsFixed(2)} detections per second ($sustainedFrames frames)',
      '- Latency: median ${sustainedLatency.median.toStringAsFixed(1)} ms, p95 ${sustainedLatency.p95.toStringAsFixed(1)} ms',
      '',
      '## Device state',
      '',
      '| | Before | After |',
      '|---|---|---|',
      '| Thermal status | ${_v(before.thermal)} | ${_v(after.thermal)} |',
      '| Battery temperature | ${_v(before.batteryTempC, ' °C')} | ${_v(after.batteryTempC, ' °C')} |',
      '| Battery level | ${_v(before.batteryPct, ' %')} | ${_v(after.batteryPct, ' %')} |',
      '| Battery current | ${_v(before.batteryCurrentUa, ' µA')} | ${_v(after.batteryCurrentUa, ' µA')} |',
      '| Charging | ${_v(before.charging)} | ${_v(after.charging)} |',
      '| App memory (PSS) | ${_v(before.appMemMb, ' MB')} | ${_v(after.appMemMb, ' MB')} |',
      '| Free system memory | ${_v(before.availMemMb, ' MB')} | ${_v(after.availMemMb, ' MB')} |',
      '',
      '- Temperature change: $dT; battery change: $dB${(before.charging ?? false) ? ' (charging: battery impact not meaningful)' : ''}',
      '- Battery temperature is the closest reading Android gives apps to device temperature.',
    ].join('\n');
  }
}

/// Measures the detector on this phone. Pauses live detection while it runs so
/// the two don't compete, and restores it afterwards.
class BenchmarkRunner {
  final DetectionService detector;
  final DetectorController controller;
  final CameraService camera;
  final LivePipeline pipeline;
  final Future<Uint8List> Function() testImage;
  final AuraSettings Function() settings;
  final MethodChannel? system;

  final progress = ValueNotifier<String>('');
  bool _cancel = false;

  BenchmarkRunner({
    required this.detector,
    required this.controller,
    required this.camera,
    required this.pipeline,
    required this.testImage,
    required this.settings,
    this.system,
  });

  void cancel() => _cancel = true;

  Future<BenchmarkReport> run({int burst = 100, Duration sustained = const Duration(seconds: 60)}) async {
    if (!controller.status.value.ready) throw StateError('The detector is not ready.');
    _cancel = false;
    final wasRunning = pipeline.running;
    if (wasRunning) await pipeline.stop();
    try {
      final img = await testImage();
      final useCamera = camera.latestFrame != null;
      Future<DetectionResult> once() {
        final f = camera.latestFrame;
        return f != null ? detector.detect(f, conf: 0.1) : detector.detectImage(img, conf: 0.1);
      }

      final before = await SystemSnapshot.read(system);
      progress.value = 'Warming up…';
      for (var i = 0; i < 5; i++) {
        await once();
      }
      final tot = <double>[], pre = <double>[], mod = <double>[], post = <double>[];
      for (var i = 0; i < burst && !_cancel; i++) {
        final r = await once();
        tot.add(r.inferenceMs);
        pre.add(r.preMs);
        mod.add(r.modelMs);
        post.add(r.postMs);
        if (i % 10 == 0) progress.value = 'Latency test: $i of $burst';
      }
      // Sustained: paced at the configured detection rate, like live use.
      final fps = settings().detectionFps;
      final period = Duration(microseconds: (1e6 / fps).round());
      final sw = Stopwatch()..start();
      final lat = <double>[];
      var frames = 0;
      while (sw.elapsed < sustained && !_cancel) {
        final t0 = sw.elapsed;
        final r = await once();
        lat.add(r.inferenceMs);
        frames++;
        if (frames % 10 == 0) {
          progress.value = 'Sustained test: ${sw.elapsed.inSeconds} of ${sustained.inSeconds} s';
        }
        final wait = period - (sw.elapsed - t0);
        if (wait > Duration.zero) await Future<void>.delayed(wait);
      }
      sw.stop();
      final after = await SystemSnapshot.read(system);
      final report = BenchmarkReport(
        detector: controller.status.value,
        source: useCamera ? 'live camera frames' : 'bundled test image (camera off)',
        total: LatencyStats.of(tot),
        pre: LatencyStats.of(pre),
        model: LatencyStats.of(mod),
        post: LatencyStats.of(post),
        sustained: sw.elapsed,
        sustainedFrames: frames,
        targetFps: fps,
        sustainedLatency: LatencyStats.of(lat),
        before: before,
        after: after,
      );
      progress.value = 'Done';
      return report;
    } finally {
      if (wasRunning) await pipeline.start();
    }
  }

  /// Saves the report next to the app's data and echoes it to the system log
  /// (`adb logcat -s flutter`), so it can be collected without the UI.
  static Future<File> save(BenchmarkReport r, Directory dir) async {
    await dir.create(recursive: true);
    final stamp = r.after.time.toIso8601String().replaceAll(':', '-').split('.').first;
    await File('${dir.path}/bench-$stamp.json').writeAsString(const JsonEncoder.withIndent(' ').convert(r.toJson()));
    final md = File('${dir.path}/bench-$stamp.md');
    await md.writeAsString(r.toMarkdown());
    for (final line in r.toMarkdown().split('\n')) {
      debugPrint('AuraBench: $line');
    }
    return md;
  }
}
