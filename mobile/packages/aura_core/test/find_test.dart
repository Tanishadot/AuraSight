import 'package:aura_core/aura_core.dart';
import 'package:test/test.dart';

const w = 640, h = 480;

ObjectView obj(int id, String label, List<double> box) {
  final b = Box.fromList(box);
  return ObjectView(id, label, 0.9, [b.x1 / w, b.y1 / h, b.x2 / w, b.y2 / h], analyzeBox(b, w, h, label), 5, 0);
}

void main() {
  late double t;
  late FindGuide guide;
  setUp(() {
    t = 0;
    guide = FindGuide('cup', () => t);
  });

  test('guides to the nearest matching object and says when it is centred', () {
    final far = obj(1, 'cup', [600, 100, 620, 120]);
    final near = obj(2, 'cup', [280, 200, 400, 330]);
    final g = guide.update([obj(3, 'chair', [0, 0, 100, 200]), far, near]);
    expect(g.target!.trackId, 2);
    expect(g.centred, isTrue);
    expect(g.message, 'Cup straight ahead, near.');
  });

  test('direction for an off-centre object', () {
    final g = guide.update([obj(1, 'cup', [440, 200, 480, 250])]);
    expect(g.centred, isFalse);
    expect(g.message, startsWith('Cup slightly right'));
  });

  test('speaks new information at once and repeats only after the gap', () {
    final g = guide.update([obj(1, 'cup', [440, 200, 480, 250])]);
    expect(guide.speakable(g), g.message);
    t = 1;
    expect(guide.speakable(guide.update([obj(1, 'cup', [440, 200, 480, 250])])), isNull);
    t = 5;
    expect(guide.speakable(guide.update([obj(1, 'cup', [440, 200, 480, 250])])), isNotNull);
    t = 5.5;
    final moved = guide.update([obj(1, 'cup', [20, 200, 60, 250])]);
    expect(guide.speakable(moved), startsWith('Cup on your left'));
  });

  test('"not in view" waits two seconds', () {
    expect(guide.speakable(guide.update([])), isNull);
    t = 1;
    expect(guide.speakable(guide.update([])), isNull);
    t = 2.1;
    expect(guide.speakable(guide.update([])), startsWith('No cup in view yet'));
  });

  test('uses friendly names', () {
    final g = FindGuide('cell phone', () => 0).update([obj(1, 'cell phone', [300, 200, 340, 260])]);
    expect(g.message, startsWith('Phone straight ahead'));
  });
}
