import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart';

/// One camera frame in YUV 4:2:0, as delivered by the camera.
class CameraFrame {
  final int width, height; // sensor orientation (not yet rotated)
  final int rotation; // clockwise degrees to make it upright
  final List<FramePlane> planes;
  final DateTime time;
  const CameraFrame(this.width, this.height, this.rotation, this.planes, this.time);

  int get uprightWidth => rotation % 180 == 0 ? width : height;
  int get uprightHeight => rotation % 180 == 0 ? height : width;

  Map<String, Object?> toChannel() => {
        'width': width,
        'height': height,
        'rotation': rotation,
        'format': 'yuv420',
        'planes': [for (final p in planes) p.toChannel()],
      };
}

class FramePlane {
  final Uint8List bytes;
  final int rowStride;
  final int pixelStride;
  const FramePlane(this.bytes, this.rowStride, this.pixelStride);
  Map<String, Object?> toChannel() => {'bytes': bytes, 'rowStride': rowStride, 'pixelStride': pixelStride};
}

enum CameraStatus { off, starting, running, noPermission, unavailable, error }

class CameraState {
  final CameraStatus status;
  final String? message;
  const CameraState(this.status, [this.message]);
}

/// Camera access: live preview plus a stream of frames for analysis.
abstract class CameraService {
  ValueListenable<CameraState> get state;
  Stream<CameraFrame> get frames;
  CameraFrame? get latestFrame;

  /// Preview widget while running, otherwise null.
  Widget? buildPreview();
  Future<void> start();
  Future<void> stop();
  Future<void> dispose();
}

/// Implementation on the official `camera` plugin (CameraX on Android,
/// AVFoundation on iOS).
class PluginCameraService implements CameraService {
  /// Low (about 320x240) while the detector is being proven on phones: the
  /// model's input is 320x320 anyway, and every frame is copied from the
  /// camera into Dart and then into native code, so smaller frames mean
  /// much less memory churn. The preview is softer as a result.
  static const resolution = ResolutionPreset.low;
  final _state = ValueNotifier(const CameraState(CameraStatus.off));
  final _frames = StreamController<CameraFrame>.broadcast();
  CameraController? _controller;
  CameraFrame? _latest;

  @override
  ValueListenable<CameraState> get state => _state;
  @override
  Stream<CameraFrame> get frames => _frames.stream;
  @override
  CameraFrame? get latestFrame => _latest;

  @override
  Widget? buildPreview() {
    final c = _controller;
    if (c == null || !c.value.isInitialized) return null;
    return CameraPreview(c);
  }

  @override
  Future<void> start() async {
    if (_state.value.status == CameraStatus.running || _state.value.status == CameraStatus.starting) return;
    _state.value = const CameraState(CameraStatus.starting);
    try {
      final cams = await availableCameras();
      if (cams.isEmpty) {
        _state.value = const CameraState(CameraStatus.unavailable, 'No camera found on this device.');
        return;
      }
      final back = cams.firstWhere((c) => c.lensDirection == CameraLensDirection.back, orElse: () => cams.first);
      final c = CameraController(back, resolution, enableAudio: false, imageFormatGroup: ImageFormatGroup.yuv420);
      await c.initialize();
      _controller = c;
      await c.startImageStream((img) => _onImage(img, back.sensorOrientation));
      _state.value = const CameraState(CameraStatus.running);
    } on CameraException catch (e) {
      final denied = e.code.contains('AccessDenied') || e.code.contains('Permission');
      _state.value = denied
          ? const CameraState(CameraStatus.noPermission, 'Camera permission was denied. Allow it in system settings.')
          : CameraState(CameraStatus.error, e.description ?? e.code);
    } catch (e) {
      _state.value = CameraState(CameraStatus.error, '$e');
    }
  }

  void _onImage(CameraImage img, int sensorOrientation) {
    final frame = CameraFrame(
      img.width,
      img.height,
      sensorOrientation, // portrait-locked UI: the sensor rotation makes it upright
      [for (final p in img.planes) FramePlane(p.bytes, p.bytesPerRow, p.bytesPerPixel ?? 1)],
      DateTime.now(),
    );
    _latest = frame;
    if (_frames.hasListener) _frames.add(frame);
  }

  @override
  Future<void> stop() async {
    final c = _controller;
    _controller = null;
    _latest = null;
    if (c != null) {
      try {
        if (c.value.isStreamingImages) await c.stopImageStream();
      } catch (_) {}
      await c.dispose();
    }
    _state.value = const CameraState(CameraStatus.off);
  }

  @override
  Future<void> dispose() async {
    await stop();
    await _frames.close();
  }
}
