/// Turns raw tracks into *confirmed* objects. Port of backend/detection/verify.py.
///
/// YOLO11n confuses some architecture with appliance and screen classes: a
/// bright window looks like a lit "tv", a flat door like a "refrigerator". An
/// object is therefore confirmed only after three independent checks:
///
///  1. Temporal stability: tracked for `minTrackFrames` and present in most
///     frames since it appeared.
///  2. Class-aware confidence: the mean recent score must reach the base
///     threshold, or a higher one for [confusableClasses].
///  3. Secondary verification for confusable classes that are not strongly
///     confident: the detector is re-run on zoomed crops around the box.
///
/// On mobile, crop re-detection is a native call and therefore asynchronous:
/// [ObjectVerifier.update] returns [VerifyRequest]s, the caller runs the crops
/// through the detector and reports back with [ObjectVerifier.submit].
library;

import 'dart:collection';

import 'geometry.dart';
import 'settings.dart';
import 'tracking/bytetrack.dart';

const confusableClasses = {'tv', 'refrigerator', 'oven', 'microwave', 'toaster', 'sink', 'toilet'};

enum ObjectStatus { tentative, confirmed, rejected }

class ObjectState {
  final int firstFrame;
  final Queue<double> scores = Queue();
  int seenFrames = 0;
  ObjectStatus status = ObjectStatus.tentative;
  int verifyAttempts = 0;
  double nextVerifyAt = 0;
  List<double> verifyScores = [];
  String reason = 'collecting evidence';
  bool pending = false;

  ObjectState(this.firstFrame);

  double? get verifyScore => verifyScores.isEmpty ? null : _mean(verifyScores);
}

/// A crop re-detection the caller must run: re-detect class [cls] around [box].
class VerifyRequest {
  final int trackId;
  final Box box;
  final int cls;

  /// Padded crops (padding fraction, detector input size); the result is the
  /// mean of the best same-class score overlapping the box in each crop.
  final List<(double, int)> crops;
  const VerifyRequest(this.trackId, this.box, this.cls, this.crops);
}

class VerifierUpdate {
  final Map<int, ObjectState> states;
  final List<VerifyRequest> requests;
  const VerifierUpdate(this.states, this.requests);
}

/// Mean with NumPy's summation order (pairwise, 8 accumulators from 8
/// elements up), so thresholds compare exactly as on the desktop.
double _mean(Iterable<double> values) {
  final a = values.toList(growable: false);
  final n = a.length;
  if (n == 0) return 0;
  double sum;
  if (n < 8) {
    sum = 0.0;
    for (final x in a) {
      sum += x;
    }
  } else {
    final r = a.sublist(0, 8);
    var i = 8;
    for (; i < n - (n % 8); i += 8) {
      for (var j = 0; j < 8; j++) {
        r[j] += a[i + j];
      }
    }
    sum = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) {
      sum += a[i];
    }
  }
  return sum / n;
}

class ObjectVerifier {
  static const cropConfigs = [(0.3, 320), (0.6, 320)];
  static const verifyAccept = 0.45;
  static const verifyFastAccept = 0.60;
  static const verifyFastReject = 0.20;
  static const secondPassS = 0.4;
  static const verifyRetryS = 3.0;
  static const maxVerifyRounds = 3;
  static const minPresence = 0.6;
  static const demoteMargin = 0.05;

  final AuraSettings Function() _settings;
  final List<String> names;
  final double Function() _clock;
  final bool canVerify;

  Map<int, ObjectState> _states = {};
  int _frame = 0;
  int verificationsRun = 0;

  ObjectVerifier(this._settings, this.names, this._clock, {this.canVerify = true});

  void reset() {
    _states = {};
    _frame = 0;
    verificationsRun = 0;
  }

  VerifierUpdate update(List<TrackOutput> tracks) {
    final s = _settings();
    _frame += 1;
    final now = _clock();
    final visible = <int, ObjectState>{};
    final requests = <VerifyRequest>[];
    for (final t in tracks) {
      final st = _states.putIfAbsent(t.trackId, () => ObjectState(_frame));
      st.scores.addLast(t.score);
      if (st.scores.length > 10) st.scores.removeFirst();
      st.seenFrames += 1;
      final req = _evaluate(t, st, s, now);
      if (req != null) requests.add(req);
      visible[t.trackId] = st;
    }
    // forget tracks gone for a while (tracker IDs are never reused)
    if (_frame % 50 == 0) {
      final live = {for (final t in tracks) t.trackId};
      _states.removeWhere((tid, st) => !live.contains(tid) && _frame - st.firstFrame > 300);
    }
    return VerifierUpdate(visible, requests);
  }

  VerifyRequest? _evaluate(TrackOutput t, ObjectState st, AuraSettings s, double now) {
    final label = names[t.cls];
    final span = _frame - st.firstFrame + 1;
    final mean = _mean(st.scores);
    final confusable = confusableClasses.contains(label);
    final base = s.confidenceThreshold;
    final need = confusable ? (base > s.confusableThreshold ? base : s.confusableThreshold) : base;

    if (st.status == ObjectStatus.confirmed) {
      // Confusable classes must keep earning it: a glass panel that briefly
      // looked like a TV is demoted when its confidence sinks again.
      if (confusable && st.scores.length >= 5 && mean < need - demoteMargin) {
        st.status = ObjectStatus.tentative;
        st.reason = 'confidence fell to ${mean.toStringAsFixed(2)}';
        st.verifyScores = [];
        st.verifyAttempts = 0;
      }
      return null;
    }
    if (st.seenFrames < s.minTrackFrames) {
      st.reason = 'seen ${st.seenFrames}/${s.minTrackFrames} frames';
      return null;
    }
    if (st.seenFrames / span < minPresence) {
      st.reason = 'flickering';
      return null;
    }
    if (mean < need) {
      st.reason = 'mean confidence ${mean.toStringAsFixed(2)} < ${need.toStringAsFixed(2)}';
      return null;
    }
    if (!confusable || mean >= s.strongConfidence) {
      st.status = ObjectStatus.confirmed;
      st.reason = 'stable and confident';
      return null;
    }
    // Ambiguous confusable class -> zoomed-crop re-detection (secondary check).
    if (!canVerify || st.pending || now < st.nextVerifyAt) return null;
    if (st.verifyAttempts >= 2 * maxVerifyRounds) return null; // gave up
    st.verifyAttempts += 1;
    st.pending = true;
    return VerifyRequest(t.trackId, t.box, t.cls, cropConfigs);
  }

  /// Report the crop re-detection score for a request returned by [update].
  /// Returns the track's state, or null if the track has been forgotten.
  ObjectState? submit(int trackId, double score) {
    final st = _states[trackId];
    if (st == null || !st.pending) return st;
    st.pending = false;
    final now = _clock();
    st.verifyScores.add(score);
    verificationsRun += 1;
    final v = st.verifyScore!;
    final n = st.verifyScores.length;
    if (v >= verifyFastAccept || (n >= 2 && v >= verifyAccept)) {
      st.status = ObjectStatus.confirmed;
      st.reason = 'verified on zoomed crops (${v.toStringAsFixed(2)})';
    } else if (v < verifyFastReject || n >= 2) {
      st.status = ObjectStatus.rejected;
      st.reason = 'not confirmed on zoomed crops (${v.toStringAsFixed(2)})';
      st.verifyScores = []; // a later round starts fresh (camera may have moved)
      st.nextVerifyAt = now + verifyRetryS;
    } else {
      st.reason = 'checking (${v.toStringAsFixed(2)})';
      st.nextVerifyAt = now + secondPassS;
    }
    return st;
  }

  /// The detector found nothing usable (or failed): count it as a miss.
  ObjectState? submitFailure(int trackId) => submit(trackId, 0.0);

  /// The check could not run at all (e.g. the detector was busy): release the
  /// request without counting it as a miss; it is asked for again shortly.
  void cancel(int trackId) {
    final st = _states[trackId];
    if (st == null || !st.pending) return;
    st.pending = false;
    if (st.verifyAttempts > 0) st.verifyAttempts -= 1;
    st.nextVerifyAt = _clock() + secondPassS;
  }
}

/// Best same-class score overlapping the target box among detections made on
/// a crop. [cropOrigin] is the crop's top-left in frame pixels. Port of
/// `ObjectVerifier._crop_score`.
double cropScore(Box target, int cls, (double, double) cropOrigin, List<Detection> cropDetections) {
  final t =
      Box(target.x1 - cropOrigin.$1, target.y1 - cropOrigin.$2, target.x2 - cropOrigin.$1, target.y2 - cropOrigin.$2);
  var best = 0.0;
  for (final d in cropDetections) {
    if (d.cls == cls && iou(t, d.box) > 0.4 && d.score > best) best = d.score;
  }
  return best;
}

/// The padded crop rectangle for [box] (integer pixels, clipped to the frame),
/// or null when it is too small to re-detect in. Port of the crop maths in
/// `ObjectVerifier._crop_score`.
(int, int, int, int)? cropRect(Box box, double pad, int frameW, int frameH) {
  final bw = box.width, bh = box.height;
  final x1 = (box.x1 - pad * bw).clamp(0, double.infinity).floor();
  final y1 = (box.y1 - pad * bh).clamp(0, double.infinity).floor();
  final x2 = (box.x2 + pad * bw).clamp(double.negativeInfinity, frameW.toDouble()).floor();
  final y2 = (box.y2 + pad * bh).clamp(double.negativeInfinity, frameH.toDouble()).floor();
  if (x2 - x1 < 16 || y2 - y1 < 16) return null;
  return (x1, y1, x2, y2);
}
