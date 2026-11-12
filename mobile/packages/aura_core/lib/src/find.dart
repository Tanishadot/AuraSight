/// "Find something": guides the user towards one object class using the
/// confirmed detections, with rate-limited spoken updates.
library;

import 'scene_monitor.dart';
import 'spatial.dart';

class FindGuidance {
  final ObjectView? target;
  final String message;

  /// The object is roughly straight ahead (centre zone).
  final bool centred;
  const FindGuidance(this.target, this.message, this.centred);

  bool get found => target != null;
}

class FindGuide {
  final String label;
  final double Function() _clock;

  /// Minimum gap between repeated identical messages.
  final double repeatS;
  String? _lastSpoken;
  double _lastAt = -1e9;
  double _lostSince = -1;

  FindGuide(this.label, this._clock, {this.repeatS = 4.0});

  /// Best match: nearest first, then most central.
  FindGuidance update(List<ObjectView> confirmed) {
    final matches = [
      for (final o in confirmed)
        if (o.label == label) o
    ];
    final name = spokenLabel(label);
    if (matches.isEmpty) {
      if (_lostSince < 0) _lostSince = _clock();
      return FindGuidance(null, 'No $name in view yet. Move the phone slowly from side to side.', false);
    }
    _lostSince = -1;
    const order = {Distance.near: 0, Distance.middle: 1, Distance.far: 2};
    matches.sort((a, b) {
      final d = order[a.spatial.distance]!.compareTo(order[b.spatial.distance]!);
      return d != 0 ? d : (a.spatial.cx - 0.5).abs().compareTo((b.spatial.cx - 0.5).abs());
    });
    final t = matches.first;
    final centred = t.spatial.horizontal == 'centre';
    final where = centred ? 'straight ahead' : t.spatial.direction;
    final msg = '${name[0].toUpperCase()}${name.substring(1)} $where, ${t.spatial.distancePhrase}.';
    return FindGuidance(t, msg, centred);
  }

  /// Returns the text to speak now, or null to stay quiet: new information is
  /// spoken at once, unchanged information at most every [repeatS] seconds.
  /// "Not in view" is only said after the object has been missing for 2 s.
  String? speakable(FindGuidance g) {
    final now = _clock();
    if (!g.found && (_lostSince < 0 || now - _lostSince < 2.0)) return null;
    if (g.message != _lastSpoken || now - _lastAt >= repeatS) {
      _lastSpoken = g.message;
      _lastAt = now;
      return g.message;
    }
    return null;
  }
}
