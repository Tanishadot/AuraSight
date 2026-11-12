/// Approximate, *relative* spatial descriptions from bounding-box geometry.
/// Port of backend/spatial/positioning.py.
///
/// There is no depth sensor and no camera calibration, so AuraSight never
/// states metres. "Near / middle distance / far" is a heuristic based on how
/// much of the frame an object fills, normalised by a rough per-class size prior.
library;

import 'dart:math' as math;

import 'geometry.dart';

const _large = {
  'person', 'couch', 'bed', 'dining table', 'refrigerator', 'car', 'bus', 'truck', 'train', //
  'motorcycle', 'bicycle', 'horse', 'cow', 'elephant', 'bear', 'giraffe', 'zebra', 'bench',
  'airplane', 'boat',
};
const _medium = {
  'chair': 0.55, 'toilet': 0.5, 'tv': 0.5, 'dog': 0.45, 'suitcase': 0.5, 'potted plant': 0.45, //
  'oven': 0.55, 'sink': 0.45, 'microwave': 0.4, 'backpack': 0.4, 'laptop': 0.4, 'umbrella': 0.5,
  'fire hydrant': 0.45, 'stop sign': 0.4, 'sheep': 0.6, 'cat': 0.35, 'skateboard': 0.35,
  'surfboard': 0.6, 'teddy bear': 0.35, 'handbag': 0.35, 'keyboard': 0.3, 'clock': 0.3,
  'vase': 0.3, 'traffic light': 0.35, 'parking meter': 0.45,
};
const _smallDefault = 0.22; // bottle, cup, phone, remote, mouse, book, cutlery, fruit...

/// Classes that can physically block a walking path at floor level.
const obstacleClasses = {
  'person', 'chair', 'couch', 'bed', 'dining table', 'bench', 'bicycle', 'motorcycle', 'car', //
  'bus', 'truck', 'dog', 'cat', 'suitcase', 'potted plant', 'toilet', 'refrigerator',
  'fire hydrant', 'backpack', 'skateboard', 'stop sign', 'parking meter', 'horse', 'cow',
};

double sizePrior(String label) => _large.contains(label) ? 1.0 : (_medium[label] ?? _smallDefault);

enum Distance { near, middle, far }

class SpatialInfo {
  final String horizontal; // left | centre | right
  final String direction; // on your left | slightly left | ahead | slightly right | on your right
  final Distance distance;
  final bool inPath; // inside the central walking corridor
  final double cx; // normalised centre x
  final double area; // normalised box area

  const SpatialInfo(this.horizontal, this.direction, this.distance, this.inPath, this.cx, this.area);

  String get distancePhrase => switch (distance) {
        Distance.near => 'near',
        Distance.middle => 'middle distance',
        Distance.far => 'far away',
      };

  String phrase() => '$direction, $distancePhrase';
}

(String, String) horizontalZone(double cx) {
  if (cx < 0.2) return ('left', 'on your left');
  if (cx < 0.4) return ('left', 'slightly left');
  if (cx <= 0.6) return ('centre', 'ahead');
  if (cx <= 0.8) return ('right', 'slightly right');
  return ('right', 'on your right');
}

Distance distanceBucket(Box b, num frameW, num frameH, String label) {
  final hf = b.height / frameH;
  final wf = b.width / frameW;
  final s = math.max(hf, wf);
  final norm = s / sizePrior(label);
  final touchesBottom = b.y2 >= 0.97 * frameH;
  if (norm >= 0.85 || (touchesBottom && s >= 0.45)) return Distance.near;
  if (norm >= 0.4) return Distance.middle;
  return Distance.far;
}

SpatialInfo analyzeBox(Box b, num frameW, num frameH, String label) {
  final cx = ((b.x1 + b.x2) / 2) / frameW;
  final (horizontal, direction) = horizontalZone(cx);
  // Walking corridor: the central 40 % of the image width.
  final overlap = math.max(0.0, math.min(b.x2 / frameW, 0.7) - math.max(b.x1 / frameW, 0.3));
  final inPath = overlap >= 0.2 || (cx >= 0.3 && cx <= 0.7);
  final area = (b.width * b.height) / (frameW * frameH);
  return SpatialInfo(horizontal, direction, distanceBucket(b, frameW, frameH, label), inPath, cx, area);
}

const _irregularPlural = {
  'person': 'people', 'mouse': 'mice', 'knife': 'knives', 'bus': 'buses', 'couch': 'couches', //
  'wine glass': 'wine glasses', 'skis': 'skis', 'sheep': 'sheep', 'toothbrush': 'toothbrushes',
  'sandwich': 'sandwiches', 'dining table': 'tables', 'tv': 'TVs', 'cell phone': 'phones',
  'hair drier': 'hair driers', 'glass': 'glasses',
};

String plural(String label, int n) => n == 1 ? label : (_irregularPlural[label] ?? '${label}s');

const _spoken = {
  'dining table': 'table', 'cell phone': 'phone', 'tv': 'TV or monitor', //
  'potted plant': 'plant', 'couch': 'sofa',
};

/// Friendlier names for speech.
String spokenLabel(String label) => _spoken[label] ?? label;

String article(String word) => 'aeiou'.contains(word.isEmpty ? '_' : word[0].toLowerCase()) ? 'an' : 'a';
