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
  const DetectorInfo({
    required this.available,
    this.backend = 'none',
    this.inputSize = 0,
    this.labels = const [],
    this.reason,
    this.loadMs = 0,
    this.fallbackReason,
  });

  static const notLoaded = DetectorInfo(available: false, reason: 'Not loaded yet.');
}

class DetectionResult {
  final List<Detection> detections; // upright pixel coordinates
  final int width, height; // upright frame size
  final double inferenceMs; // total: preprocessing + model + decoding
  final double preMs, modelMs, postMs;
  const DetectionResult(this.detections, this.width, this.height, this.inferenceMs,
      {this.preMs = 0, this.modelMs = 0, this.postMs = 0});
}

/// On-device object detection (YOLO11n). Shared interface; inference runs
/// natively (LiteRT on Android, Core ML on iOS) on a background thread.
abstract class DetectionService {
  DetectorInfo get info;

  /// Loads the model file and runs the native self-check (shapes, finite output).
  Future<DetectorInfo> load(String modelPath, DetectorManifest manifest, {String backend = 'auto'});
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
  Future<DetectorInfo> load(String modelPath, DetectorManifest manifest, {String backend = 'auto'}) async {
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
      _result((await _ch.invokeMapMethod<String, Object?>('detect', {'frame': frame.toChannel(), 'conf': conf}))!);

  @override
  Future<List<Detection>> detectCrop(CameraFrame frame, (int, int, int, int) rect, int size,
      {double conf = 0.05}) async {
    final r = await _ch.invokeMapMethod<String, Object?>('detectCrop', {
      'frame': frame.toChannel(),
      'rect': [rect.$1, rect.$2, rect.$3, rect.$4],
      'size': size,
      'conf': conf,
    });
    return _parse(r!);
  }

  @override
  Future<DetectionResult> detectImage(Uint8List bytes, {double conf = 0.25}) async =>
      _result((await _ch.invokeMapMethod<String, Object?>('detectImage', {'bytes': bytes, 'conf': conf}))!);

  static DetectionResult _result(Map<String, Object?> r) {
    double n(String k) => (r[k] as num?)?.toDouble() ?? 0;
    return DetectionResult(_parse(r), r['width']! as int, r['height']! as int, n('inferenceMs'),
        preMs: n('preMs'), modelMs: n('modelMs'), postMs: n('postMs'));
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
    try {
      await _ch.invokeMethod<void>('dispose');
    } on MissingPluginException {
      // nothing loaded
    }
    _info = DetectorInfo.notLoaded;
  }
}
