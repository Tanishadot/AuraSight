import 'package:aura_core/aura_core.dart';
import 'package:test/test.dart';

void main() {
  test('a cancelled verification is not a miss and is asked for again', () {
    var t = 0.0;
    const settings = AuraSettings(minTrackFrames: 1);
    final v = ObjectVerifier(() => settings, const ['tv'], () => t);
    // A confusable class, confident enough to need a check but not strongly confident.
    final score = (settings.confusableThreshold + settings.strongConfidence) / 2;
    TrackOutput track() => TrackOutput(1, const Box(10, 10, 200, 200), score, 0, 3, 3);
    final first = v.update([track()]);
    expect(first.requests, hasLength(1));
    v.cancel(1);
    final st = first.states[1]!;
    expect(st.status, isNot(ObjectStatus.rejected));
    expect(st.verifyAttempts, 0);
    expect(v.verificationsRun, 0);
    t += ObjectVerifier.secondPassS + 0.01;
    expect(v.update([track()]).requests, hasLength(1));
  });

  test('a failed verification (score 0) still rejects', () {
    var t = 0.0;
    const settings = AuraSettings(minTrackFrames: 1);
    final v = ObjectVerifier(() => settings, const ['tv'], () => t);
    final score = (settings.confusableThreshold + settings.strongConfidence) / 2;
    v.update([TrackOutput(1, const Box(10, 10, 200, 200), score, 0, 3, 3)]);
    expect(v.submitFailure(1)!.status, ObjectStatus.rejected);
  });
}
