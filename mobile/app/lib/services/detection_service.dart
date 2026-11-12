import 'package:aura_core/aura_core.dart';
import 'package:flutter/services.dart';

import '../platform/channels.dart';
import 'camera_service.dart';

class DetectorInfo {
  final bool available;
  final String backend; // gpu | npu | cpu | ane | none
  final int inputSize;
  final List<String> labels;
  final String? reason;
  final double loadMs;
  const DetectorInfo({
    required this.available,
    this.backend = 'none',
    this.inputSize = 0,
    this.labels = const [],
    this.reason,
    this.loadMs = 0,
  });

  static const notLoaded = DetectorInfo(available: false, reason: 'Not loaded yet.');
}

class DetectionResult {
  final List<Detection> detections; // upright pixel coordinates
  final int width, height; // upright frame size
  final double inferenceMs;
  const DetectionResult(this.detections, this.width, this.height, this.inferenceMs);
}

/// On-device object detection (YOLO11n). Shared interface; the inference
/// runs natively (LiteRT on Android, Core ML on iOS).
abstract class DetectionService {
  DetectorInfo get info;
  Future<DetectorInfo> load({int inputSize = 320, String backend = 'auto'});
  Future<DetectionResult> detect(CameraFrame frame, {double conf = 0.1});

  /// Re-detect inside [rect] (upright pixels) at input [size]; boxes are in crop coordinates.
  Future<List<Detection>> detectCrop(CameraFrame frame, (int, int, int, int) rect, int size, {double conf = 0.05});
  Future<void> dispose();
}

class NativeDetectionService implements DetectionService {
  final MethodChannel _ch;
  DetectorInfo _info = DetectorInfo.notLoaded;

  NativeDetectionService([MethodChannel? channel]) : _ch = channel ?? AuraChannels.detector;

  @override
  DetectorInfo get info => _info;

  @override
  Future<DetectorInfo> load({int inputSize = 320, String backend = 'auto'}) async {
    try {
      final r = await _ch.invokeMapMethod<String, Object?>('load', {'inputSize': inputSize, 'backend': backend});
      _info = DetectorInfo(
        available: true,
        backend: r!['backend']! as String,
        inputSize: r['inputSize']! as int,
        labels: (r['labels']! as List).cast<String>(),
        loadMs: (r['loadMs'] as num?)?.toDouble() ?? 0,
      );
    } on MissingPluginException {
      _info = const DetectorInfo(available: false, reason: "This build doesn't include the on-device detector yet.");
    } on PlatformException catch (e) {
      _info = DetectorInfo(available: false, reason: e.message ?? e.code);
    }
    return _info;
  }

  @override
  Future<DetectionResult> detect(CameraFrame frame, {double conf = 0.1}) async {
    final r = await _ch.invokeMapMethod<String, Object?>('detect', {'frame': frame.toChannel(), 'conf': conf});
    return DetectionResult(_parse(r!), r['width']! as int, r['height']! as int, (r['inferenceMs']! as num).toDouble());
  }

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
