import 'package:aura_core/aura_core.dart';
import 'package:flutter/services.dart';

import '../platform/channels.dart';
import 'camera_service.dart';
import 'detector_model.dart';

class DetectorInfo {
  final bool available;
  final String backend; // gpu | cpu | none
  final int inputSize;
  final List<String> labels;
  final String? reason;
  final double loadMs;

  /// Why the GPU wasn't used, when the detector fell back to the CPU.
  final String? fallbackReason;

  /// The first inference (warm-up on a grey image) during load.
  final double warmupMs;

  /// True when the native side reused an already-loaded model instead of loading again.
  final bool cached;
  final NativeMemory memory;
  const DetectorInfo({
    required this.available,
    this.backend = 'none',
    this.inputSize = 0,
    this.labels = const [],
    this.reason,
    this.loadMs = 0,
    this.fallbackReason,
    this.warmupMs = 0,
    this.cached = false,
    this.memory = const NativeMemory(),
  });

  static const notLoaded = DetectorInfo(available: false, reason: 'Not loaded yet.');
}

/// Process memory as the native side saw it after a call (cheap readings, no PSS walk).
class NativeMemory {
  final double? nativeHeapMb, javaHeapMb;
  const NativeMemory({this.nativeHeapMb, this.javaHeapMb});

  factory NativeMemory.fromMap(Map<String, Object?> r) => NativeMemory(
      nativeHeapMb: (r['nativeHeapMb'] as num?)?.toDouble(), javaHeapMb: (r['javaHeapMb'] as num?)?.toDouble());

  @override
  String toString() => nativeHeapMb == null
      ? 'memory n/a'
      : 'native heap ${nativeHeapMb!.toStringAsFixed(1)} MB, java heap ${javaHeapMb?.toStringAsFixed(1)} MB';
}

class DetectionResult {
  final List<Detection> detections; // upright pixel coordinates
  final int width, height; // upright frame size
  final double inferenceMs; // total: preprocessing + model + decoding
  final double preMs, modelMs, postMs;
  final NativeMemory memory;
  const DetectionResult(this.detections, this.width, this.height, this.inferenceMs,
      {this.preMs = 0, this.modelMs = 0, this.postMs = 0, this.memory = const NativeMemory()});
}

/// The native side refused a request because another detection is still
/// running (requests are never queued). Callers skip the frame.
class DetectorBusy implements Exception {
  @override
  String toString() => 'DetectorBusy: another detection is still running';
}

/// A detector call failed; [outOfMemory] when the phone ran out of memory.
class DetectorFailure implements Exception {
  final String message;
  final bool outOfMemory;
  const DetectorFailure(this.message, {this.outOfMemory = false});
  @override
  String toString() => message;
}

/// What the native crash journal recorded (see DetectorGuard.kt).
class GuardStatus {
  /// The native step the previous process died in, e.g. `gpu_warmup` (null if it didn't).
  final String? interruptedStage, interruptedBackend;
  final DateTime? interruptedAt;
  final String? gpuBlocked; // why the GPU is switched off
  final String? lastGood; // backend whose load + self-test last completed
  const GuardStatus(
      {this.interruptedStage, this.interruptedBackend, this.interruptedAt, this.gpuBlocked, this.lastGood});

  static const none = GuardStatus();

  factory GuardStatus.fromMap(Map<String, Object?> r) {
    final i = (r['interrupted'] as Map?)?.cast<String, Object?>();
    return GuardStatus(
      interruptedStage: i?['stage'] as String?,
      interruptedBackend: i?['backend'] as String?,
      interruptedAt: i?['atMs'] == null ? null : DateTime.fromMillisecondsSinceEpoch((i!['atMs']! as num).toInt()),
      gpuBlocked: r['gpuBlocked'] as String?,
      lastGood: r['lastGood'] as String?,
    );
  }
}

/// Plain-language names for the journalled native steps.
String describeStage(String stage) => switch (stage) {
      'map_model' => 'opening the model file',
      'gpu_compat_check' => 'checking GPU support',
      'gpu_delegate_create' => 'starting the GPU delegate',
      'gpu_interpreter_create' => 'preparing the model for the GPU',
      'gpu_warmup' => 'the first GPU inference (warm-up)',
      'cpu_interpreter_create' => 'preparing the model for the CPU',
      'cpu_warmup' => 'the first CPU inference (warm-up)',
      'tensor_shapes' => 'reading the model tensor shapes',
      'first_inference' => 'the first detection on an image',
      _ => stage,
    };

/// The native crash journal and the GPU switch (Android only; elsewhere a no-op).
abstract class DetectorGuardService {
  Future<GuardStatus> status();
  Future<void> markGood(String backend);
  Future<void> clearGpuBlock();
}

class NativeDetectorGuard implements DetectorGuardService {
  final MethodChannel _ch;
  NativeDetectorGuard([MethodChannel? channel]) : _ch = channel ?? AuraChannels.detector;

  @override
  Future<GuardStatus> status() async {
    try {
      final r = await _ch.invokeMapMethod<String, Object?>('guardStatus');
      return r == null ? GuardStatus.none : GuardStatus.fromMap(r);
    } on MissingPluginException {
      return GuardStatus.none;
    } on PlatformException {
      return GuardStatus.none;
    }
  }

  @override
  Future<void> markGood(String backend) async {
    try {
      await _ch.invokeMethod<void>('markGood', {'backend': backend});
    } on MissingPluginException {
      // not available on this platform
    }
  }

  @override
  Future<void> clearGpuBlock() async {
    try {
      await _ch.invokeMethod<void>('clearGpuBlock');
    } on MissingPluginException {
      // not available on this platform
    }
  }
}

/// On-device object detection (YOLO11n). Shared interface; inference runs
/// natively (LiteRT on Android, Core ML on iOS) on a background thread.
abstract class DetectionService {
  DetectorInfo get info;

  /// Loads the model file and runs the native self-check (shapes, finite output).
  /// [backend]: "cpu" (default, safe), "auto" (GPU, falling back to CPU) or "gpu".
  Future<DetectorInfo> load(String modelPath, DetectorManifest manifest, {String backend = 'cpu'});
  Future<DetectionResult> detect(CameraFrame frame, {double conf = 0.1});

  /// Re-detect inside [rect] (upright pixels); boxes are in crop coordinates.
  Future<List<Detection>> detectCrop(CameraFrame frame, (int, int, int, int) rect, int size, {double conf = 0.05});

  /// Detect on an encoded image (JPEG/PNG); used by the self-test and benchmarks.
  Future<DetectionResult> detectImage(Uint8List bytes, {double conf = 0.25});
  Future<void> dispose();
}

class NativeDetectionService implements DetectionService {
  final MethodChannel _ch;
  DetectorInfo _info = DetectorInfo.notLoaded;

  NativeDetectionService([MethodChannel? channel]) : _ch = channel ?? AuraChannels.detector;

  @override
  DetectorInfo get info => _info;

  @override
  Future<DetectorInfo> load(String modelPath, DetectorManifest manifest, {String backend = 'cpu'}) async {
    // Frames must not be sent while the native side swaps models.
    _info = DetectorInfo.notLoaded;
    try {
      final r = await _ch.invokeMapMethod<String, Object?>('load', {
        'modelPath': modelPath,
        'inputSize': manifest.inputSize,
        'numClasses': manifest.labels.length,
        'normalized': manifest.normalizedBoxes,
        'iou': manifest.iouThreshold,
        'backend': backend,
      });
      _info = DetectorInfo(
        available: true,
        backend: r!['backend']! as String,
        inputSize: r['inputSize']! as int,
        labels: manifest.labels,
        loadMs: (r['loadMs'] as num?)?.toDouble() ?? 0,
        fallbackReason: r['fallbackReason'] as String?,
        warmupMs: (r['warmupMs'] as num?)?.toDouble() ?? 0,
        cached: r['cached'] as bool? ?? false,
        memory: NativeMemory.fromMap(r),
      );
    } on MissingPluginException {
      _info = const DetectorInfo(available: false, reason: "This build doesn't include the on-device detector.");
    } on PlatformException catch (e) {
      _info = DetectorInfo(available: false, reason: e.message ?? e.code);
    }
    return _info;
  }

  @override
  Future<DetectionResult> detect(CameraFrame frame, {double conf = 0.1}) async =>
      _result((await _call('detect', {'frame': frame.toChannel(), 'conf': conf}))!);

  /// Maps native error codes to [DetectorBusy] / [DetectorFailure].
  Future<Map<String, Object?>?> _call(String method, Map<String, Object?> args) async {
    try {
      return await _ch.invokeMapMethod<String, Object?>(method, args);
    } on PlatformException catch (e) {
      if (e.code == 'busy') throw DetectorBusy();
      throw DetectorFailure(e.message ?? e.code, outOfMemory: e.code == 'oom');
    }
  }

  @override
  Future<List<Detection>> detectCrop(CameraFrame frame, (int, int, int, int) rect, int size,
      {double conf = 0.05}) async {
    final r = await _call('detectCrop', {
      'frame': frame.toChannel(),
      'rect': [rect.$1, rect.$2, rect.$3, rect.$4],
      'size': size,
      'conf': conf,
    });
    return _parse(r!);
  }

  @override
  Future<DetectionResult> detectImage(Uint8List bytes, {double conf = 0.25}) async =>
      _result((await _call('detectImage', {'bytes': bytes, 'conf': conf}))!);

  static DetectionResult _result(Map<String, Object?> r) {
    double n(String k) => (r[k] as num?)?.toDouble() ?? 0;
    return DetectionResult(_parse(r), r['width']! as int, r['height']! as int, n('inferenceMs'),
        preMs: n('preMs'), modelMs: n('modelMs'), postMs: n('postMs'), memory: NativeMemory.fromMap(r));
  }

  static List<Detection> _parse(Map<String, Object?> r) {
    final boxes = r['boxes']! as List, scores = r['scores']! as List, classes = r['classes']! as List;
    return [
      for (var i = 0; i < boxes.length; i++)
        Detection(Box.fromList(boxes[i] as List), (scores[i] as num).toDouble(), classes[i] as int)
    ];
  }

  @override
  Future<void> dispose() async {
    _info = DetectorInfo.notLoaded;
    try {
      await _ch.invokeMethod<void>('dispose');
    } on MissingPluginException {
      // nothing loaded
    } on PlatformException {
      // already released
    }
  }
}
