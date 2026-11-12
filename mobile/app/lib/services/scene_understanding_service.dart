import 'dart:async';

import 'package:flutter/services.dart';

import '../platform/channels.dart';
import 'camera_service.dart';

enum VlmStatus { notInstalled, unloaded, loading, ready, busy, unavailable, error }

class VlmState {
  final VlmStatus status;
  final String? detail;
  final String backend;
  const VlmState(this.status, {this.detail, this.backend = 'none'});
}

class GenerationResult {
  final String text;
  final double imageMs, totalMs;
  final int tokens;
  const GenerationResult(this.text, this.imageMs, this.totalMs, this.tokens);
}

/// Thrown when no scene model can run (not installed, not built, or failed).
class VlmUnavailable implements Exception {
  final String message;
  const VlmUnavailable(this.message);
  @override
  String toString() => message;
}

/// Local vision-language model (SmolVLM via llama.cpp/mtmd, natively).
/// Runs only on request or on selected frames, never on every frame.
abstract class SceneUnderstandingService {
  VlmState get state;
  Stream<VlmState> get states;
  Future<VlmState> load({required String modelPath, required String mmprojPath, int threads = 0});

  /// Generates text for [frame]; [onDelta] receives streamed text pieces.
  Future<GenerationResult> generate(CameraFrame frame, String prompt,
      {int maxTokens = 80, Map<String, Object?> params = const {}, void Function(String delta)? onDelta});
  Future<void> cancel();
  Future<void> unload();
}

class NativeVlmService implements SceneUnderstandingService {
  final MethodChannel _ch;
  final EventChannel _tokens;
  final _states = StreamController<VlmState>.broadcast();
  VlmState _state = const VlmState(VlmStatus.unloaded);
  int _nextId = 0;
  int? _activeId;

  NativeVlmService({MethodChannel? channel, EventChannel? tokens})
      : _ch = channel ?? AuraChannels.vlm,
        _tokens = tokens ?? AuraChannels.vlmTokens;

  void _set(VlmState s) {
    _state = s;
    _states.add(s);
  }

  @override
  VlmState get state => _state;
  @override
  Stream<VlmState> get states => _states.stream;

  @override
  Future<VlmState> load({required String modelPath, required String mmprojPath, int threads = 0}) async {
    _set(const VlmState(VlmStatus.loading));
    try {
      final r = await _ch.invokeMapMethod<String, Object?>(
          'load', {'model': modelPath, 'mmproj': mmprojPath, 'threads': threads, 'gpu': true});
      _set(VlmState(VlmStatus.ready, backend: r?['backend'] as String? ?? 'cpu'));
    } on MissingPluginException {
      _set(const VlmState(VlmStatus.unavailable, detail: "This build doesn't include the on-device scene model yet."));
    } on PlatformException catch (e) {
      _set(VlmState(VlmStatus.error, detail: e.message ?? e.code));
    }
    return _state;
  }

  @override
  Future<GenerationResult> generate(CameraFrame frame, String prompt,
      {int maxTokens = 80, Map<String, Object?> params = const {}, void Function(String delta)? onDelta}) async {
    if (_state.status != VlmStatus.ready) throw VlmUnavailable(_state.detail ?? 'The scene model is not loaded.');
    final id = ++_nextId;
    _activeId = id;
    _set(VlmState(VlmStatus.busy, backend: _state.backend));
    StreamSubscription<dynamic>? sub;
    if (onDelta != null) {
      sub = _tokens.receiveBroadcastStream().listen((e) {
        final m = e as Map<Object?, Object?>;
        if (m['requestId'] == id) onDelta(m['delta']! as String);
      });
    }
    try {
      final r = await _ch.invokeMapMethod<String, Object?>('generate', {
        'requestId': id,
        'frame': frame.toChannel(),
        'prompt': prompt,
        'maxTokens': maxTokens,
        'params': params,
      });
      return GenerationResult(r!['text']! as String, (r['imageMs']! as num).toDouble(),
          (r['totalMs']! as num).toDouble(), r['tokens']! as int);
    } on PlatformException catch (e) {
      throw VlmUnavailable(e.message ?? e.code);
    } finally {
      await sub?.cancel();
      _activeId = null;
      if (_state.status == VlmStatus.busy) _set(VlmState(VlmStatus.ready, backend: _state.backend));
    }
  }

  @override
  Future<void> cancel() async {
    final id = _activeId;
    if (id != null) await _ch.invokeMethod<void>('cancel', {'requestId': id});
  }

  @override
  Future<void> unload() async {
    try {
      await _ch.invokeMethod<void>('unload');
    } on MissingPluginException {
      // nothing loaded
    }
    _set(const VlmState(VlmStatus.unloaded));
  }
}
