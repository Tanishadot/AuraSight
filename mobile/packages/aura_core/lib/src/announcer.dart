/// Decides which scene events are worth *speaking*, and how.
/// Port of backend/assistant/announcer.py.
///
/// Every event still goes to the activity log (visual + screen reader); only
/// the spoken channel is filtered, grouped and prioritised here.
library;

import 'scene_monitor.dart';
import 'spatial.dart';
import 'speech_queue.dart';

class Announcer {
  final AuraSettingsGetter _settings;
  final Speaker _speech;
  final double Function() _clock;

  /// True while a scene description is being generated or read aloud:
  /// routine announcements would otherwise land between its sentences.
  final bool Function() _hold;
  final Map<String, double> _labelLast = {};

  Announcer(this._settings, this._speech, this._clock, {bool Function()? hold}) : _hold = hold ?? (() => false);

  void reset() => _labelLast.clear();

  /// A scene description is covering these classes: treat them as just
  /// announced so "New: laptop" doesn't repeat what the description says.
  void noteDescribed(Iterable<String> labels) {
    final now = _clock();
    for (final l in labels) {
      _labelLast[l] = now;
    }
  }

  /// Speak what matters; returns the spoken phrases (for tests and logging).
  List<String> handle(List<SceneEvent> events) {
    final s = _settings();
    if (s.announcementsMuted || events.isEmpty) return const [];
    final now = _clock();
    final spoken = <String>[];

    final warnings = [
      for (final e in events)
        if (e.kind == EventKind.obstacle || e.kind == EventKind.approaching) e
    ];
    if (s.obstacleAlerts && warnings.isNotEmpty) {
      final text = warnings.take(2).map((e) => e.message).join(' ');
      if (_speech.say(text, priority: SpeechPriority.urgent, kind: 'alert', maxAge: 2.5)) spoken.add(text);
      for (final e in warnings) {
        if (e.label != null) _labelLast[e.label!] = now;
      }
    }

    if (_hold()) return spoken; // warnings above still interrupt; the rest is in the activity log

    if (s.announceNewObjects) {
      final fresh = <SceneEvent>[];
      for (final e in events) {
        if (e.kind != EventKind.entered || e.label == null) continue;
        final last = _labelLast[e.label!] ?? -1e9;
        if (now - last < s.announceCooldownS) continue;
        _labelLast[e.label!] = now;
        fresh.add(e);
      }
      if (fresh.isNotEmpty) {
        final parts = <String>[];
        for (final e in fresh.take(3)) {
          var where = e.spatial?.direction ?? '';
          if (s.verbosity.name == 'detailed' && e.spatial != null) where = e.spatial!.phrase();
          parts.add('${spokenLabel(e.label!)} $where'.trim());
        }
        var text = 'New: ${parts.join(', ')}';
        if (fresh.length > 3) text += ', and ${fresh.length - 3} more';
        text += '.';
        if (_speech.say(text, priority: SpeechPriority.normal, kind: 'new_objects', maxAge: 4.0)) spoken.add(text);
      }
    }

    if (s.announceDepartures) {
      final gone = [
        for (final e in events)
          if (e.kind == EventKind.left && e.label != null) spokenLabel(e.label!)
      ];
      if (gone.isNotEmpty) {
        var text = '${gone.take(3).join(', ')} gone.';
        text = text[0].toUpperCase() + text.substring(1);
        if (_speech.say(text, priority: SpeechPriority.low, kind: 'departure', maxAge: 4.0)) spoken.add(text);
      }
    }

    if (s.announceSceneChanges && events.any((e) => e.kind == EventKind.sceneChange)) {
      const text = 'The view has changed.';
      if (_speech.say(text, priority: SpeechPriority.normal, kind: 'scene_change', maxAge: 3.0)) spoken.add(text);
    }
    return spoken;
  }
}

/// Detector-only summary, e.g. "a laptop ahead, near; 2 chairs on your right".
String summarizeObjects(List<ObjectView> objects, {int maxGroups = 6, bool detailed = true}) {
  if (objects.isEmpty) return '';
  const order = {Distance.near: 0, Distance.middle: 1, Distance.far: 2};
  // closest first, then most central; stable like Python's sort
  final indexed = [for (var i = 0; i < objects.length; i++) (i, objects[i])];
  indexed.sort((a, b) {
    final da = order[a.$2.spatial.distance]!, db = order[b.$2.spatial.distance]!;
    if (da != db) return da.compareTo(db);
    final ca = (a.$2.spatial.cx - 0.5).abs(), cb = (b.$2.spatial.cx - 0.5).abs();
    if (ca != cb) return ca.compareTo(cb);
    return a.$1.compareTo(b.$1);
  });
  final groups = <String, List<ObjectView>>{};
  for (final (_, o) in indexed) {
    groups.putIfAbsent('${o.label}\u0000${o.spatial.direction}', () => []).add(o);
  }
  final parts = <String>[];
  for (final entry in groups.entries.take(maxGroups)) {
    final items = entry.value;
    final label = items.first.label, direction = items.first.spatial.direction;
    final name = spokenLabel(label);
    final n = items.length;
    final head = n == 1 ? '${article(name)} $name' : '$n ${plural(name, n)}';
    final tail = detailed ? '$direction, ${items.first.spatial.distancePhrase}' : direction;
    parts.add('$head $tail');
  }
  var text = parts.join('; ');
  final extra = groups.length - maxGroups;
  if (extra > 0) text += '; and $extra more';
  return text;
}

// Words a scene description may use for a detector class. Used only to avoid
// repeating an object the description already named.
const _synonyms = <String, List<String>>{
  'person': [
    'person', 'people', 'man', 'men', 'woman', 'women', 'boy', 'girl', 'child', 'children', //
    'kid', 'kids', 'someone', 'lady', 'guy', 'baby'
  ],
  'tv': ['tv', 'tvs', 'television', 'monitor', 'monitors', 'screen', 'screens'],
  'dining table': ['table', 'tables', 'desk', 'desks'],
  'cell phone': ['phone', 'phones', 'smartphone', 'mobile'],
  'couch': ['couch', 'sofa', 'sofas', 'couches'],
  'potted plant': ['plant', 'plants', 'flower', 'flowers', 'pot'],
  'laptop': ['laptop', 'laptops', 'computer', 'notebook'],
  'cup': ['cup', 'cups', 'mug', 'mugs'],
  'wine glass': ['glass', 'glasses', 'wine'],
  'car': ['car', 'cars', 'vehicle', 'vehicles'],
  'bicycle': ['bicycle', 'bicycles', 'bike', 'bikes'],
  'motorcycle': ['motorcycle', 'motorbike', 'scooter'],
  'refrigerator': ['refrigerator', 'fridge'],
  'handbag': ['handbag', 'bag', 'purse'],
  'backpack': ['backpack', 'bag', 'rucksack'],
  'keyboard': ['keyboard'],
  'mouse': ['mouse'],
  'book': ['book', 'books'],
  'bottle': ['bottle', 'bottles'],
  'chair': ['chair', 'chairs', 'stool', 'seat'],
  'bed': ['bed', 'beds'],
  'dog': ['dog', 'dogs', 'puppy'],
  'cat': ['cat', 'cats', 'kitten'],
};

final _word = RegExp('[a-z]+');

/// True if a description already names this detector class (or a synonym).
bool mentionedIn(String label, String text) {
  final lower = text.toLowerCase();
  final syn = _synonyms[label];
  if (syn == null && label.contains(' ')) {
    // multi-word class: needs the whole phrase
    return lower.contains(label.toLowerCase()) || lower.contains(plural(label, 2).toLowerCase());
  }
  final words = {for (final m in _word.allMatches(lower)) m.group(0)!};
  final names = syn ?? [...label.toLowerCase().split(' '), plural(label, 2).toLowerCase()];
  return names.any(words.contains);
}
