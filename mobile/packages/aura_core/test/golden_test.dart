// Parity with the desktop implementation: every expectation here was produced
// by running backend/ (Python) on the same inputs - see mobile/tools/gen_golden.py.
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:aura_core/aura_core.dart';
import 'package:test/test.dart';

const w = 640, h = 480;
const names = ['person', 'chair', 'tv', 'cup', 'laptop', 'refrigerator', 'dining table', 'bottle'];

dynamic golden(String name) => jsonDecode(File('test/golden/$name.json').readAsStringSync());

double d(dynamic v) => (v as num).toDouble();

/// Desktop settings keys (snake_case) -> AuraSettings.
AuraSettings settingsFrom(Map<String, dynamic> py) {
  String camel(String k) => k.replaceAllMapped(RegExp('_([a-z])'), (m) => m[1]!.toUpperCase());
  return const AuraSettings().merge({for (final e in py.entries) camel(e.key): e.value});
}

class Clock {
  double t = 100.0;
  double call() => t;
}

ObjectView view(int id, String label, List<dynamic> box) {
  final b = Box.fromList(box);
  return ObjectView(id, label, 0.9, [b.x1 / w, b.y1 / h, b.x2 / w, b.y2 / h], analyzeBox(b, w, h, label), 1, 0);
}

void main() {
  group('tracker parity', () {
    for (final sc in golden('tracker') as List) {
      test(sc['name'] as String, () {
        final p = sc['params'] as Map;
        final tr = ByteTracker(maxLostFrames: (p['max_lost_frames'] as int?) ?? 15);
        final frames = sc['frames'] as List;
        final expected = sc['expected'] as List;
        for (var f = 0; f < frames.length; f++) {
          final fr = frames[f] as Map;
          final boxes = fr['boxes'] as List, scores = fr['scores'] as List, classes = fr['classes'] as List;
          final out = tr.update([
            for (var i = 0; i < boxes.length; i++)
              Detection(Box.fromList(boxes[i] as List), d(scores[i]), classes[i] as int)
          ]);
          final exp = expected[f] as List;
          expect(out.map((o) => o.trackId).toList(), [for (final e in exp) e['id']], reason: 'frame $f ids');
          for (var i = 0; i < exp.length; i++) {
            final e = exp[i] as Map, o = out[i];
            expect([o.cls, o.hits, o.ageFrames], [e['cls'], e['hits'], e['age']], reason: 'frame $f #$i');
            expect(o.score, closeTo(d(e['score']), 1e-6));
            final eb = e['box'] as List;
            for (var k = 0; k < 4; k++) {
              expect(o.box.toList()[k], closeTo(d(eb[k]), 1e-3), reason: 'frame $f #$i box[$k]');
            }
          }
        }
      });
    }
  });

  test('spatial parity', () {
    final g = golden('spatial') as Map;
    for (final c in g['analyze'] as List) {
      final s = analyzeBox(Box.fromList(c['box'] as List), w, h, c['label'] as String);
      final why = '${c['label']} ${c['box']}';
      expect(s.horizontal, c['horizontal'], reason: why);
      expect(s.direction, c['direction'], reason: why);
      expect(s.distance.name, c['distance'], reason: why);
      expect(s.inPath, c['in_path'], reason: why);
      expect(s.cx, closeTo(d(c['cx']), 1e-9));
      expect(s.area, closeTo(d(c['area']), 1e-9));
      expect(s.phrase(), c['phrase']);
    }
    for (final c in g['plural'] as List) {
      expect(plural(c['label'] as String, c['n'] as int), c['out']);
    }
    for (final c in g['spoken'] as List) {
      expect(spokenLabel(c['label'] as String), c['out']);
    }
  });

  group('verifier parity', () {
    for (final sc in golden('verifier') as List) {
      test(sc['name'] as String, () {
        final clock = Clock();
        final ver = ObjectVerifier(() => const AuraSettings(), names, clock.call);
        final table = {
          for (final e in (sc['crop_scores'] as Map).entries)
            int.parse(e.key as String): [for (final v in e.value as List) d(v)]
        };
        final calls = <int, int>{};
        final frames = sc['frames'] as List, expected = sc['expected'] as List;
        for (var f = 0; f < frames.length; f++) {
          clock.t += 0.2;
          final tracks = [
            for (final t in frames[f] as List)
              TrackOutput(t['id'] as int, const Box(100, 100, 300, 260), d(t['score']), t['cls'] as int, 5, 5)
          ];
          final up = ver.update(tracks);
          for (final r in up.requests) {
            final i = calls[r.trackId] ?? 0;
            calls[r.trackId] = i + 1;
            final seq = table[r.trackId] ?? [0.0];
            ver.submit(r.trackId, seq[i < seq.length ? i : seq.length - 1]);
          }
          final exp = expected[f] as Map;
          expect(up.states.keys.map((k) => '$k').toSet(), exp.keys.toSet(), reason: 'frame $f');
          for (final e in exp.entries) {
            final st = up.states[int.parse(e.key as String)]!;
            expect(st.status.name, e.value['status'], reason: 'frame $f track ${e.key}');
            expect(st.reason, e.value['reason'], reason: 'frame $f track ${e.key}');
          }
        }
      });
    }
  });

  group('scene monitor parity', () {
    for (final sc in golden('monitor') as List) {
      test(sc['name'] as String, () {
        final s = settingsFrom((sc['settings'] as Map).cast<String, dynamic>());
        final clock = Clock();
        final mon = SceneMonitor(() => s, clock.call);
        final frames = sc['frames'] as List, expected = sc['expected'] as List;
        final thumbs = sc['thumbs'] as List?;
        for (var f = 0; f < frames.length; f++) {
          clock.t += 0.2;
          final views = [for (final o in frames[f] as List) view(o['id'] as int, o['label'] as String, o['box'] as List)];
          Float32List? thumb;
          if (thumbs != null) {
            thumb = Float32List.fromList([
              for (final row in thumbs[f] as List)
                for (final v in row as List) d(v)
            ]);
          }
          final evs = mon.update(views, thumb);
          final exp = expected[f] as Map;
          expect(
              evs.map((e) => [e.kindName, e.message, e.priority.name, e.label, e.trackId]).toList(),
              [
                for (final e in exp['events'] as List) [e['kind'], e['message'], e['priority'], e['label'], e['track_id']]
              ],
              reason: 'frame $f');
          expect(views.map((v) => v.approaching).toList(), exp['approaching'], reason: 'frame $f approaching');
        }
      });
    }
  });

  group('announcer parity', () {
    final g = golden('announcer') as Map;
    SceneEvent event(Map<dynamic, dynamic> e) {
      final sp = e['spatial'] as Map?;
      return SceneEvent(
        e['kind'] == 'scene_change' ? EventKind.sceneChange : EventKind.values.byName(e['kind'] as String),
        e['message'] as String,
        EventPriority.values.byName(e['priority'] as String),
        label: e['label'] as String?,
        trackId: e['track_id'] as int?,
        spatial: sp == null
            ? null
            : SpatialInfo(sp['horizontal'] as String, sp['direction'] as String,
                Distance.values.byName(sp['distance'] as String), sp['in_path'] as bool, d(sp['cx']), d(sp['area'])),
      );
    }

    final batches = [for (final b in g['batches'] as List) [for (final e in b as List) event(e as Map)]];
    for (final run in g['runs'] as List) {
      test('handle ${run['settings']}', () {
        final s = settingsFrom((run['settings'] as Map).cast<String, dynamic>());
        final clock = Clock();
        final spy = _Spy();
        final a = Announcer(() => s, spy, clock.call);
        final returned = <List<String>>[];
        for (final b in batches) {
          clock.t += 2.0;
          returned.add(a.handle(b));
        }
        expect(returned, run['returned']);
        expect(spy.said, run['said']);
      });
    }

    test('summarizeObjects', () {
      for (final c in g['summaries'] as List) {
        final objs = [for (final o in c['objects'] as List) view(o['id'] as int, o['label'] as String, o['box'] as List)];
        expect(summarizeObjects(objs, maxGroups: c['max_groups'] as int, detailed: c['detailed'] as bool), c['out']);
      }
    });

    test('mentionedIn', () {
      for (final c in g['mentioned'] as List) {
        expect(mentionedIn(c['label'] as String, c['text'] as String), c['out'],
            reason: '${c['label']} in "${c['text']}"');
      }
    });
  });
}

class _Spy implements Speaker {
  final said = <Map<String, dynamic>>[];
  @override
  bool say(String text,
      {SpeechPriority priority = SpeechPriority.normal, String kind = 'info', double? maxAge, bool dedupe = true}) {
    said.add({'text': text, 'priority': priority.index, 'kind': kind, 'max_age': maxAge});
    return true;
  }
}
