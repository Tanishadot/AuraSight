import 'dart:math' as math;

import 'package:aurasight/theme/aura_theme.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

double _lum(Color c) {
  double ch(double v) => v <= 0.03928 ? v / 12.92 : math.pow((v + 0.055) / 1.055, 2.4).toDouble();
  return 0.2126 * ch(c.r) + 0.7152 * ch(c.g) + 0.0722 * ch(c.b);
}

double contrast(Color a, Color b) {
  final la = _lum(a), lb = _lum(b);
  return (math.max(la, lb) + 0.05) / (math.min(la, lb) + 0.05);
}

void main() {
  for (final b in Brightness.values) {
    test('text contrast meets WCAG AA in ${b.name} theme', () {
      final cs = auraTheme(b).colorScheme;
      final pairs = {
        'onSurface/surface': (cs.onSurface, cs.surface),
        'onSurfaceVariant/surface': (cs.onSurfaceVariant, cs.surface),
        'onSurface/raised': (cs.onSurface, cs.surfaceContainerHighest),
        'onSurfaceVariant/raised': (cs.onSurfaceVariant, cs.surfaceContainerHighest),
        'onPrimary/primary': (cs.onPrimary, cs.primary),
        'onPrimaryContainer/primaryContainer': (cs.onPrimaryContainer, cs.primaryContainer),
        'primary/surface': (cs.primary, cs.surface),
        'error/surface': (cs.error, cs.surface),
      };
      for (final e in pairs.entries) {
        expect(contrast(e.value.$1, e.value.$2), greaterThanOrEqualTo(4.5), reason: e.key);
      }
      // body text aims for AAA
      expect(contrast(cs.onSurface, cs.surface), greaterThanOrEqualTo(7));
    });
  }
}
