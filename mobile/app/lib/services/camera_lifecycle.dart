import 'package:flutter/widgets.dart';

import 'camera_service.dart';
import 'live_pipeline.dart';

/// Releases the camera and stops detection when the app leaves the
/// foreground, and restores them when it comes back, but only if they were
/// running. The camera is a shared, power-hungry resource; Android also
/// revokes it from background apps.
class CameraLifecycle {
  final CameraService camera;
  final LivePipeline pipeline;
  bool _resumeOnReturn = false;
  Future<void> _pending = Future.value();

  CameraLifecycle(this.camera, this.pipeline);

  bool get willResume => _resumeOnReturn;

  /// Start the camera and detection together.
  Future<void> start() => _serial(() async {
        await camera.start();
        if (camera.state.value.status == CameraStatus.running) await pipeline.start();
      });

  /// Stop detection, then the camera.
  Future<void> stop() => _serial(() async {
        _resumeOnReturn = false;
        await pipeline.stop();
        await camera.stop();
      });

  Future<void> onAppLifecycle(AppLifecycleState s) => _serial(() async {
        switch (s) {
          case AppLifecycleState.inactive:
          case AppLifecycleState.hidden:
          case AppLifecycleState.paused:
          case AppLifecycleState.detached:
            if (camera.state.value.status == CameraStatus.running) {
              _resumeOnReturn = true;
              await pipeline.stop();
              await camera.stop();
            }
          case AppLifecycleState.resumed:
            if (_resumeOnReturn) {
              _resumeOnReturn = false;
              await camera.start();
              if (camera.state.value.status == CameraStatus.running) await pipeline.start();
            }
        }
      });

  /// Lifecycle events can arrive faster than the camera opens or closes; run them in order.
  Future<void> _serial(Future<void> Function() action) => _pending = _pending.then((_) => action(), onError: (_) => action());
}
