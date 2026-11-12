import 'dart:math' as math;
import 'dart:typed_data';

/// Axis-aligned box in pixel coordinates (x1, y1) top-left, (x2, y2) bottom-right.
class Box {
  final double x1, y1, x2, y2;
  const Box(this.x1, this.y1, this.x2, this.y2);

  factory Box.fromList(List<dynamic> v) =>
      Box((v[0] as num).toDouble(), (v[1] as num).toDouble(), (v[2] as num).toDouble(), (v[3] as num).toDouble());

  double get width => x2 - x1;
  double get height => y2 - y1;
  double get area => width * height;

  List<double> toList() => [x1, y1, x2, y2];

  /// The same box with each coordinate rounded to float32, as the desktop
  /// tracker stores boxes.
  Box toF32() => Box(f32(x1), f32(y1), f32(x2), f32(y2));

  @override
  String toString() => 'Box($x1, $y1, $x2, $y2)';
}

/// Intersection over union of two boxes (0 when they don't overlap).
double iou(Box a, Box b) {
  final ix = math.max(0.0, math.min(a.x2, b.x2) - math.max(a.x1, b.x1));
  final iy = math.max(0.0, math.min(a.y2, b.y2) - math.max(a.y1, b.y1));
  final inter = ix * iy;
  return inter / math.max(a.area + b.area - inter, 1e-6);
}

final Float32List _f32 = Float32List(1);

/// Rounds a double to the nearest float32 value.
double f32(double v) {
  _f32[0] = v;
  return _f32[0];
}
