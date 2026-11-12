/// Prioritised, non-overlapping, interruptible speech queue over a native
/// text-to-speech engine. Port of backend/speech/service.py.
///
///  * One utterance plays at a time.
///  * A higher-priority request interrupts lower-priority speech; a USER or
///    URGENT sentence cut off that way is read again afterwards (stop and
///    cancel discard it).
///  * The queue is bounded; when full the least important, oldest item goes.
///  * Time-sensitive items carry `maxAge` and are dropped when stale.
///  * Identical text already queued or playing is not queued twice, unless
///    `dedupe: false` (explicit replays).
library;

import 'dart:async';

import 'settings.dart';

typedef AuraSettingsGetter = AuraSettings Function();

enum SpeechPriority {
  low, // departures, ambient info
  normal, // new objects, automatic descriptions
  user, // answers to things the user explicitly asked for
  urgent, // obstacle / approaching-object warnings
}

/// Anything that can be asked to speak (the queue, or a test spy).
abstract interface class Speaker {
  bool say(String text, {SpeechPriority priority, String kind, double? maxAge, bool dedupe});
}

/// Platform TTS: Android TextToSpeech, iOS AVSpeechSynthesizer.
abstract interface class TtsEngine {
  /// Prepares the engine; throws if no offline voice is usable.
  Future<void> init();

  /// Speaks [text] and completes when playback ends: true if it finished,
  /// false if [interrupt] cut it off. Throws on engine failure.
  Future<bool> speak(String text, {required double rate, required double volume, String? voiceId});

  /// Stops current playback immediately.
  Future<void> interrupt();
}

enum SpeechState { notStarted, ready, speaking, error }

class SpeechEvent {
  final String state; // ready | speaking | idle | stopped | error
  final String? text;
  final String? kind;
  final String? error;
  final bool interrupted;
  const SpeechEvent(this.state, {this.text, this.kind, this.error, this.interrupted = false});
}

class _Item {
  final SpeechPriority priority;
  final int seq;
  final String text;
  final String kind;
  final double created;
  final double? maxAge;
  bool cancelled = false;
  _Item(this.priority, this.seq, this.text, this.kind, this.created, this.maxAge);

  /// Higher priority first, then FIFO.
  int compareTo(_Item o) => o.priority.index != priority.index ? o.priority.index - priority.index : seq - o.seq;
}

class SpeechQueue implements Speaker {
  final AuraSettingsGetter _settings;
  final TtsEngine _engine;
  final double Function() _clock;
  final int maxQueue;

  final List<_Item> _queue = [];
  _Item? _current;
  int _seq = 0;
  Completer<void>? _wake;
  bool _running = false;
  final _events = StreamController<SpeechEvent>.broadcast();

  SpeechState state = SpeechState.notStarted;
  String? error;
  int spokenCount = 0;

  SpeechQueue(this._settings, this._engine, this._clock, {this.maxQueue = 10});

  Stream<SpeechEvent> get events => _events.stream;
  bool get isSpeaking => _current != null;
  int get queued => _queue.length;

  /// Kind of the utterance playing now, or of the next queued one.
  String? get currentKind => _current?.kind ?? (_queue.isEmpty ? null : _queue.first.kind);

  Future<void> start() async {
    if (_running) return;
    try {
      await _engine.init();
      state = SpeechState.ready;
      _events.add(const SpeechEvent('ready'));
    } catch (e) {
      state = SpeechState.error;
      error = 'No offline voice available: $e';
      _events.add(SpeechEvent('error', error: error));
      return;
    }
    _running = true;
    unawaited(_run());
  }

  Future<void> shutdown() async {
    _running = false;
    await stopAll();
    _wake?.complete();
    await _events.close();
  }

  @override
  bool say(String text,
      {SpeechPriority priority = SpeechPriority.normal, String kind = 'info', double? maxAge, bool dedupe = true}) {
    text = text.split(RegExp(r'\s+')).where((w) => w.isNotEmpty).join(' ');
    if (text.isEmpty) return false;
    if (!_settings().speechEnabled || state == SpeechState.error) return false;
    if (dedupe && (_queue.any((i) => i.text == text) || _current?.text == text)) return false;
    _insert(_Item(priority, _seq++, text, kind, _clock(), maxAge));
    while (_queue.length > maxQueue) {
      // drop the least important, oldest item
      var worst = _queue.first;
      for (final i in _queue) {
        if (i.priority.index < worst.priority.index ||
            (i.priority.index == worst.priority.index && i.seq < worst.seq)) {
          worst = i;
        }
      }
      _queue.remove(worst);
    }
    final cur = _current;
    if (cur != null && priority.index > cur.priority.index) unawaited(_engine.interrupt());
    _signal();
    return true;
  }

  /// Drop queued items of one kind and cut off the current one if it is of
  /// that kind. Other speech is untouched. Returns how many were removed.
  int cancel(String kind) {
    final before = _queue.length;
    _queue.removeWhere((i) => i.kind == kind);
    var removed = before - _queue.length;
    final cur = _current;
    if (cur != null && cur.kind == kind) {
      cur.cancelled = true;
      unawaited(_engine.interrupt());
      removed++;
    }
    return removed;
  }

  /// Silence immediately and clear everything queued.
  Future<void> stopAll() async {
    _queue.clear();
    _current?.cancelled = true;
    await _engine.interrupt();
    _events.add(const SpeechEvent('stopped'));
  }

  void _insert(_Item item) {
    var i = 0;
    while (i < _queue.length && _queue[i].compareTo(item) <= 0) {
      i++;
    }
    _queue.insert(i, item);
  }

  void _signal() {
    final w = _wake;
    if (w != null && !w.isCompleted) w.complete();
  }

  Future<void> _run() async {
    while (_running) {
      if (_queue.isEmpty) {
        _wake = Completer<void>();
        await _wake!.future;
        continue;
      }
      final item = _queue.removeAt(0);
      if (item.cancelled || (item.maxAge != null && _clock() - item.created > item.maxAge!)) continue;
      _current = item;
      final s = _settings();
      state = SpeechState.speaking;
      _events.add(SpeechEvent('speaking', text: item.text, kind: item.kind));
      var completed = true;
      String? failed;
      try {
        completed = await _engine.speak(item.text, rate: s.speechRate, volume: s.speechVolume, voiceId: s.voiceId);
        spokenCount++;
      } catch (e) {
        failed = e.toString();
        error = failed;
      }
      _current = null;
      state = SpeechState.ready;
      if (!completed && failed == null && !item.cancelled && item.priority.index >= SpeechPriority.user.index) {
        // Pre-empted by an urgent warning (not stopped by the user): read the
        // sentence again afterwards so nothing asked for is lost.
        _insert(_Item(item.priority, item.seq, item.text, item.kind, _clock(), null));
      }
      if (failed != null) _events.add(SpeechEvent('error', error: failed, kind: item.kind));
      _events.add(SpeechEvent('idle', kind: item.kind, interrupted: !completed));
    }
  }
}
