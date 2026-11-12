import 'dart:async';

import 'package:aura_core/aura_core.dart';
import 'package:flutter/services.dart';

import '../data/database.dart';

/// Decides what the user hears and feels: one prioritised speech queue for
/// everything (alerts, announcements, descriptions, answers), plus haptics.
/// Every event is also written to the activity log.
class AudioOrchestrator {
  final AuraSettings Function() _settings;
  final SpeechQueue speech;
  late final Announcer announcer;
  final Future<void> Function(ActivityEntry) _log;
  final Future<void> Function(HapticKind) _haptic;

  /// Descriptions in progress; while > 0 routine announcements are held.
  int describing = 0;

  AudioOrchestrator(
    this._settings,
    this.speech,
    double Function() clock, {
    required Future<void> Function(ActivityEntry) log,
    Future<void> Function(HapticKind)? haptic,
  })  : _log = log,
        _haptic = haptic ?? defaultHaptic {
    announcer = Announcer(_settings, speech, clock, hold: () => describing > 0 || speech.currentKind == 'description');
  }

  Future<void> start() => speech.start();

  /// Scene events from the tracker: log all, speak what matters, buzz on warnings.
  void handleEvents(List<SceneEvent> events) {
    if (events.isEmpty) return;
    final now = DateTime.now();
    for (final e in events) {
      unawaited(_log(ActivityEntry(time: now, kind: e.kindName, message: e.message, priority: e.priority.name)));
    }
    announcer.handle(events);
    final s = _settings();
    if (s.hapticsEnabled && s.obstacleAlerts && events.any((e) => e.priority == EventPriority.warning)) {
      unawaited(_haptic(HapticKind.warning));
    }
  }

  /// Speak an answer the user asked for.
  bool sayAnswer(String text) => speech.say(text, priority: SpeechPriority.user, kind: 'answer');

  Future<void> log(String kind, String message, {String priority = 'info'}) =>
      _log(ActivityEntry(time: DateTime.now(), kind: kind, message: message, priority: priority));

  Future<void> stop() => speech.stopAll();

  Future<void> buzz(HapticKind kind) async {
    if (_settings().hapticsEnabled) await _haptic(kind);
  }
}

enum HapticKind { tap, found, warning }

Future<void> defaultHaptic(HapticKind k) => switch (k) {
      HapticKind.tap => HapticFeedback.selectionClick(),
      HapticKind.found => HapticFeedback.mediumImpact(),
      HapticKind.warning => HapticFeedback.heavyImpact(),
    };
