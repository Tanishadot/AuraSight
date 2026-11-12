// Behaviour of the speech queue, mirroring tests/test_speech.py on the desktop.
import 'dart:async';

import 'package:aura_core/aura_core.dart';
import 'package:test/test.dart';

/// Fake platform TTS: each utterance "plays" for [duration] unless interrupted.
class FakeTts implements TtsEngine {
  Duration duration;
  bool failInit = false;
  bool fail = false;
  final spoken = <String>[];
  final interrupted = <String>[];
  int active = 0, maxActive = 0;
  Completer<void>? _stop;

  FakeTts([this.duration = const Duration(milliseconds: 50)]);

  @override
  Future<void> init() async {
    if (failInit) throw StateError('no voice');
  }

  @override
  Future<bool> speak(String text, {required double rate, required double volume, String? voiceId}) async {
    if (fail) throw StateError('audio device unavailable');
    active++;
    if (active > maxActive) maxActive = active;
    final stop = _stop = Completer<void>();
    final done = await Future.any([Future.delayed(duration, () => true), stop.future.then((_) => false)]);
    active--;
    (done ? spoken : interrupted).add(text);
    return done;
  }

  @override
  Future<void> interrupt() async {
    final s = _stop;
    if (s != null && !s.isCompleted) s.complete();
  }
}

double _now() => DateTime.now().microsecondsSinceEpoch / 1e6;

Future<SpeechQueue> make(FakeTts tts, [AuraSettings s = const AuraSettings()]) async {
  final q = SpeechQueue(() => s, tts, _now);
  await q.start();
  return q;
}

Future<void> until(bool Function() cond, {int ms = 3000}) async {
  final end = DateTime.now().add(Duration(milliseconds: ms));
  while (!cond()) {
    if (DateTime.now().isAfter(end)) fail('condition not met within $ms ms');
    await Future<void>.delayed(const Duration(milliseconds: 5));
  }
}

const ms = Duration(milliseconds: 1);

void main() {
  test('speaks in order without overlap', () async {
    final tts = FakeTts();
    final q = await make(tts);
    for (final t in ['one', 'two', 'three']) {
      q.say(t);
    }
    await until(() => tts.spoken.length == 3);
    expect(tts.spoken, ['one', 'two', 'three']);
    expect(tts.maxActive, 1);
  });

  test('priority order and urgent interrupts', () async {
    final tts = FakeTts(ms * 300);
    final q = await make(tts);
    q.say('long low priority sentence', priority: SpeechPriority.low);
    await until(() => q.isSpeaking);
    q.say('normal');
    q.say('Caution: chair close, ahead.', priority: SpeechPriority.urgent);
    await until(() => tts.spoken.length == 2);
    expect(tts.interrupted, ['long low priority sentence']);
    expect(tts.spoken, ['Caution: chair close, ahead.', 'normal']);
  });

  test('duplicates are not queued unless dedupe is off', () async {
    final tts = FakeTts(ms * 200);
    final q = await make(tts);
    expect(q.say('same'), isTrue);
    await until(() => q.isSpeaking);
    expect(q.say('same'), isFalse);
    expect(q.say('same', dedupe: false), isTrue);
    await until(() => tts.spoken.length == 2);
  });

  test('stale items are dropped', () async {
    final tts = FakeTts(ms * 300);
    final q = await make(tts);
    q.say('blocking', priority: SpeechPriority.user);
    await until(() => q.isSpeaking);
    q.say('stale info', maxAge: 0.05);
    await until(() => tts.spoken.contains('blocking'));
    await Future<void>.delayed(ms * 100);
    expect(tts.spoken, isNot(contains('stale info')));
  });

  test('queue is bounded', () async {
    final tts = FakeTts(ms * 500);
    final q = await make(tts);
    q.say('busy', priority: SpeechPriority.user);
    await until(() => q.isSpeaking);
    for (var i = 0; i < 20; i++) {
      q.say('item $i', priority: SpeechPriority.low);
    }
    expect(q.queued, lessThanOrEqualTo(10));
    await q.stopAll();
  });

  test('stopAll silences and clears', () async {
    final tts = FakeTts(ms * 500);
    final q = await make(tts);
    q.say('a long announcement');
    q.say('next');
    await until(() => q.isSpeaking);
    await q.stopAll();
    await until(() => !q.isSpeaking);
    await Future<void>.delayed(ms * 100);
    expect(tts.spoken, isEmpty);
    expect(tts.interrupted, ['a long announcement']);
  });

  test('cancel(kind) only silences that kind', () async {
    final tts = FakeTts(ms * 300);
    final q = await make(tts);
    q.say('Sentence one.', priority: SpeechPriority.user, kind: 'description');
    q.say('Sentence two.', priority: SpeechPriority.user, kind: 'description');
    q.say('Your answer.', priority: SpeechPriority.user, kind: 'answer');
    await until(() => q.isSpeaking && q.currentKind == 'description');
    expect(q.cancel('description'), 2);
    await until(() => tts.spoken.length == 1);
    expect(tts.spoken, ['Your answer.']);
    expect(tts.interrupted, ['Sentence one.']);
  });

  test('description cut off by a warning is read again afterwards', () async {
    final tts = FakeTts(ms * 200);
    final q = await make(tts);
    q.say('A woman sits at a desk.', priority: SpeechPriority.user, kind: 'description');
    q.say('A laptop is on your right.', priority: SpeechPriority.user, kind: 'description');
    await until(() => q.isSpeaking);
    q.say('Caution: chair close, ahead.', priority: SpeechPriority.urgent, kind: 'alert');
    await until(() => tts.spoken.length == 3);
    expect(tts.spoken, ['Caution: chair close, ahead.', 'A woman sits at a desk.', 'A laptop is on your right.']);
  });

  test('disabled speech drops everything', () async {
    final q = await make(FakeTts(), const AuraSettings(speechEnabled: false));
    expect(q.say('hello'), isFalse);
  });

  test('engine init failure is reported', () async {
    final tts = FakeTts()..failInit = true;
    final q = await make(tts);
    expect(q.state, SpeechState.error);
    expect(q.error, contains('no voice'));
    expect(q.say('hello'), isFalse);
  });

  test('speak error is reported and the queue keeps working', () async {
    final tts = FakeTts(ms * 10)..fail = true;
    final q = await make(tts);
    final events = <SpeechEvent>[];
    q.events.listen(events.add);
    q.say('broken', kind: 'description');
    await until(() => events.any((e) => e.state == 'error' && e.kind == 'description'));
    tts.fail = false;
    q.say('works again');
    await until(() => tts.spoken.contains('works again'));
  });
}
