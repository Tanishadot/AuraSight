import 'package:aura_core/aura_core.dart';
import 'package:flutter/services.dart';

import '../platform/channels.dart';

class Voice {
  final String id, name, language;
  final bool offline;
  const Voice(this.id, this.name, this.language, this.offline);
}

/// Platform text-to-speech (Android TextToSpeech, iOS AVSpeechSynthesizer)
/// behind aura_core's [TtsEngine], so the shared [SpeechQueue] decides what
/// is said and when.
class NativeTtsEngine implements TtsEngine {
  final MethodChannel _ch;
  String engineName = 'unknown';
  int offlineVoices = 0;

  NativeTtsEngine([MethodChannel? channel]) : _ch = channel ?? AuraChannels.tts;

  @override
  Future<void> init() async {
    try {
      final r = await _ch.invokeMapMethod<String, Object?>('init');
      engineName = r?['engine'] as String? ?? 'unknown';
      offlineVoices = r?['offlineVoices'] as int? ?? 0;
    } on MissingPluginException {
      throw StateError("this build doesn't include the speech module yet");
    } on PlatformException catch (e) {
      throw StateError(e.message ?? e.code);
    }
    if (offlineVoices == 0) {
      throw StateError('no offline voice is installed. Install one in the system text-to-speech settings');
    }
  }

  Future<List<Voice>> voices() async {
    try {
      final list = await _ch.invokeListMethod<Map<Object?, Object?>>('voices') ?? const [];
      return [
        for (final v in list)
          Voice(v['id']! as String, v['name']! as String, v['language']! as String, v['offline']! as bool)
      ];
    } on MissingPluginException {
      return const [];
    }
  }

  @override
  Future<bool> speak(String text, {required double rate, required double volume, String? voiceId}) async {
    final done =
        await _ch.invokeMethod<bool>('speak', {'text': text, 'rate': rate, 'volume': volume, 'voiceId': voiceId});
    return done ?? false;
  }

  @override
  Future<void> interrupt() async {
    try {
      await _ch.invokeMethod<void>('stop');
    } on MissingPluginException {
      // nothing playing
    }
  }
}
