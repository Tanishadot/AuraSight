import 'dart:async';

import 'package:aura_core/aura_core.dart';
import 'package:flutter/foundation.dart';

import 'audio_orchestrator.dart';
import 'camera_service.dart';
import 'live_pipeline.dart';
import 'scene_understanding_service.dart';

enum SceneStage { idle, analysing, streaming }

class SceneView {
  final SceneStage stage;
  final List<String> sentences; // exactly what is shown and spoken
  final String source; // vlm | detector | ''
  final double latencyMs;
  final double? firstSentenceMs;
  final String speech; // spoken | muted | unavailable | ''
  final String? error;
  const SceneView({
    this.stage = SceneStage.idle,
    this.sentences = const [],
    this.source = '',
    this.latencyMs = 0,
    this.firstSentenceMs,
    this.speech = '',
    this.error,
  });

  String get text => sentences.join(' ');
  bool get hasDescription => sentences.isNotEmpty && stage == SceneStage.idle;
}

/// Describe / replay / stop, with the desktop's rules: every sentence shown
/// is spoken, in order; a new description replaces one still playing; the
/// detector only adds objects the model did not mention; routine
/// announcements wait while a description is generated or read.
class SceneController {
  final AuraSettings Function() _settings;
  final CameraService camera;
  final SceneUnderstandingService vlm;
  final LivePipeline live;
  final AudioOrchestrator audio;

  final view = ValueNotifier(const SceneView());
  int _seq = 0;
  final _silenced = <int>{};

  SceneController(
      {required AuraSettings Function() settings,
      required this.camera,
      required this.vlm,
      required this.live,
      required this.audio})
      : _settings = settings;

  String _speechOutcome() {
    if (!_settings().speechEnabled) return 'muted';
    if (audio.speech.state == SpeechState.error) return 'unavailable';
    return 'spoken';
  }

  Future<SceneView> describe() async {
    final frame = camera.latestFrame;
    if (frame == null) {
      view.value = const SceneView(error: 'Turn the camera on first.');
      return view.value;
    }
    if (view.value.stage != SceneStage.idle) return view.value; // one at a time
    final seq = ++_seq;
    final sentences = <String>[];
    final t0 = DateTime.now();
    double? firstMs;
    audio.speech.cancel('description');
    audio.describing++;
    final objects = [
      for (final o in live.snapshot.value.objects)
        if (o.confirmed) o
    ];
    audio.announcer.noteDescribed({for (final o in objects) o.label});
    view.value = const SceneView(stage: SceneStage.analysing);

    void emit(String s) {
      sentences.add(s);
      firstMs ??= DateTime.now().difference(t0).inMicroseconds / 1000;
      if (!_silenced.contains(seq)) {
        audio.speech.say(s, priority: SpeechPriority.user, kind: 'description', dedupe: false);
      }
      view.value = SceneView(stage: SceneStage.streaming, sentences: List.of(sentences));
    }

    var source = 'detector';
    try {
      if (vlm.state.status != VlmStatus.ready) {
        throw VlmUnavailable(vlm.state.detail ?? 'The scene model is not loaded.');
      }
      final hints = {for (final o in objects) spokenLabel(o.label)}.toList()..sort();
      final splitter = _SentenceSplitter(emit);
      final r = await vlm.generate(frame, describePrompt(hints),
          maxTokens: _settings().vlmMaxTokens, params: generationParams, onDelta: splitter.feed);
      splitter.finish(r.text);
      source = 'vlm';
      if (sentences.isEmpty) emit("I'm not sure what I'm looking at.");
      final extra = [
        for (final o in objects)
          if (!mentionedIn(o.label, sentences.join(' '))) o
      ];
      if (extra.isNotEmpty) emit('The object detector also found ${summarizeObjects(extra)}.');
    } on VlmUnavailable catch (e) {
      final det = summarizeObjects(objects);
      emit(det.isEmpty
          ? "The scene model isn't available, and the object detector doesn't see any objects it recognises."
          : "The scene model isn't available. The object detector sees $det.");
      unawaited(audio.log('vlm', 'Scene model unavailable: $e', priority: 'warning'));
    } finally {
      audio.describing--;
      audio.announcer.noteDescribed({for (final o in objects) o.label});
    }
    final result = SceneView(
      sentences: sentences,
      source: source,
      latencyMs: DateTime.now().difference(t0).inMicroseconds / 1000,
      firstSentenceMs: firstMs,
      speech: _speechOutcome(),
    );
    view.value = result;
    unawaited(audio.log('description', result.text));
    return result;
  }

  /// Read the last description again, exactly as shown.
  String replay() {
    final last = view.value;
    if (!last.hasDescription) return 'none';
    audio.speech.cancel('description');
    final outcome = _speechOutcome();
    if (outcome == 'spoken') {
      for (final s in last.sentences) {
        audio.speech.say(s, priority: SpeechPriority.user, kind: 'description', dedupe: false);
      }
    }
    return outcome;
  }

  /// Silence everything now, including the rest of a description still being generated.
  Future<void> stop() async {
    _silenced
      ..clear()
      ..add(_seq);
    await audio.stop();
  }

  void dispose() => view.dispose();
}

/// Prompt and sampling settings proven on the desktop with SmolVLM-500M
/// (see backend/vision/analyzer.py): a short, single instruction works;
/// multi-clause prompts produce one-word answers.
String describePrompt(List<String> hints) {
  const base = 'Describe this image for a blind person in two or three short sentences.';
  return hints.isEmpty ? base : 'An object detector found: ${hints.join(', ')}.\n$base';
}

const generationParams = <String, Object?>{
  'temperature': 0.1,
  'top_p': 0.9,
  'repeat_penalty': 1.15,
  'frequency_penalty': 0.3,
  'presence_penalty': 0.3,
};

/// Splits streamed text into sentences as they complete.
class _SentenceSplitter {
  final void Function(String) onSentence;
  final int maxSentences;
  String _buf = '';
  int _count = 0;
  _SentenceSplitter(this.onSentence) : maxSentences = 3;

  static final _end = RegExp(r'[.!?](?=\s)');

  void feed(String delta) {
    _buf += delta;
    while (_count < maxSentences) {
      final m = _end.firstMatch(_buf);
      if (m == null) break;
      _emit(_buf.substring(0, m.end));
      _buf = _buf.substring(m.end);
    }
  }

  void finish(String finalText) {
    if (_count == 0 && _buf.trim().isEmpty) _buf = finalText; // no streaming: use the full text
    feed(' ');
    final rest = _buf.trim();
    if (rest.isNotEmpty && _count < maxSentences) _emit(RegExp(r'[.!?]$').hasMatch(rest) ? rest : '$rest.');
    _buf = '';
  }

  void _emit(String raw) {
    final s = raw.trim().replaceAll(RegExp(r'\s+'), ' ');
    if (s.isEmpty) return;
    _count++;
    onSentence(s);
  }
}
