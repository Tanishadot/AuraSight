/// Contracts between the shared Flutter code and the native inference
/// modules (Kotlin on Android, Swift on iOS). Both platforms implement the
/// same channel names, methods and argument maps.
///
/// Frames are passed as the camera's YUV 4:2:0 planes, never re-encoded:
///   `{width, height, rotation, format: "yuv420", planes: [{bytes, rowStride, pixelStride}]}`
/// `rotation` is the clockwise rotation (0/90/180/270) needed to make the
/// image upright. Native code returns boxes in *upright* pixel coordinates.
library;

import 'package:flutter/services.dart';

class AuraChannels {
  /// Object detector (Android: LiteRT, iOS: Core ML).
  ///  - `load` {inputSize, backend: "auto"|"gpu"|"npu"|"cpu"}
  ///      -> {backend, inputSize, labels: [String], loadMs}
  ///  - `detect` {frame, conf} -> {boxes: [[x1,y1,x2,y2]], scores, classes, inferenceMs, width, height}
  ///  - `detectCrop` {frame, rect: [x1,y1,x2,y2], size, conf} -> same as detect, in crop coordinates
  ///  - `dispose`
  static const detector = MethodChannel('aurasight/detector');

  /// Vision-language model (llama.cpp + mtmd on both platforms).
  ///  - `load` {model, mmproj, threads, gpu} -> {backend, loadMs}
  ///  - `generate` {requestId, frame, prompt, maxTokens, params} -> {text, imageMs, totalMs, tokens}
  ///  - `cancel` {requestId}
  ///  - `unload`
  static const vlm = MethodChannel('aurasight/vlm');

  /// Streamed tokens: {requestId, delta}
  static const vlmTokens = EventChannel('aurasight/vlm/tokens');

  /// Text-to-speech (Android TextToSpeech, iOS AVSpeechSynthesizer).
  ///  - `init` -> {engine, offlineVoices: int}
  ///  - `voices` -> [{id, name, language, offline: bool}]
  ///  - `speak` {text, rate, volume, voiceId} -> bool (true finished, false interrupted)
  ///  - `stop`
  static const tts = MethodChannel('aurasight/tts');

  /// Device state: thermal and memory pressure.
  ///  - `snapshot` -> {thermal: "nominal"|"fair"|"serious"|"critical", availMemMb, lowMemory: bool}
  static const system = MethodChannel('aurasight/system');
}

/// True when a channel has no native implementation in this build.
bool isMissingPlugin(Object e) => e is MissingPluginException;
