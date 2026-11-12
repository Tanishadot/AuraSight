import 'dart:async';

import 'package:aura_core/aura_core.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../services/audio_orchestrator.dart';
import '../services/camera_service.dart';
import '../services/live_pipeline.dart';
import '../state/providers.dart';
import '../widgets/common.dart';

/// Things people commonly look for, as detector classes.
const commonTargets = [
  'cup', 'bottle', 'cell phone', 'chair', 'person', 'laptop', 'book', 'remote', //
  'backpack', 'handbag', 'keyboard', 'tv', 'dining table', 'couch', 'bed', 'toilet', 'sink',
];

class FindScreen extends ConsumerStatefulWidget {
  const FindScreen({super.key});
  @override
  ConsumerState<FindScreen> createState() => _FindScreenState();
}

class _FindScreenState extends ConsumerState<FindScreen> {
  FindGuide? _guide;
  FindGuidance? _last;
  bool _wasCentred = false;

  LivePipeline get _pipeline => ref.read(livePipelineProvider);

  @override
  void initState() {
    super.initState();
    _pipeline.snapshot.addListener(_onSnapshot);
  }

  @override
  void dispose() {
    _pipeline.snapshot.removeListener(_onSnapshot);
    super.dispose();
  }

  void _start(String label) {
    setState(() {
      _guide = FindGuide(label, ref.read(clockProvider));
      _last = null;
      _wasCentred = false;
    });
    ref.read(audioProvider).sayAnswer('Looking for ${article(spokenLabel(label))} ${spokenLabel(label)}.');
  }

  void _stop() {
    setState(() => _guide = null);
    unawaited(ref.read(audioProvider).stop());
  }

  void _onSnapshot() {
    final g = _guide;
    if (g == null || !mounted) return;
    final confirmed = [for (final o in _pipeline.snapshot.value.objects) if (o.confirmed) o];
    final guidance = g.update(confirmed);
    final say = g.speakable(guidance);
    final audio = ref.read(audioProvider);
    if (say != null) audio.speech.say(say, priority: SpeechPriority.user, kind: 'find', maxAge: 2.0);
    if (guidance.centred && !_wasCentred) unawaited(audio.buzz(HapticKind.found));
    _wasCentred = guidance.centred;
    setState(() => _last = guidance);
  }

  @override
  Widget build(BuildContext context) {
    final camera = ref.watch(cameraServiceProvider);
    final labels = _pipeline.detector.info.labels;
    final targets = [for (final t in commonTargets) if (labels.isEmpty || labels.contains(t)) t];
    final text = Theme.of(context).textTheme;
    return ValueListenableBuilder<CameraState>(
      valueListenable: camera.state,
      builder: (context, cam, _) {
        final running = cam.status == CameraStatus.running;
        return ScreenBody(children: [
          if (!running)
            const StatusCard(
                icon: Icons.videocam_off_rounded,
                title: 'Camera is off',
                body: 'Start the camera on the Live tab, then choose what to find.'),
          if (_guide != null) ...[
            Semantics(
              liveRegion: true,
              child: Text(_last?.message ?? 'Looking…', style: text.headlineMedium),
            ),
            const SizedBox(height: 20),
            HeroButton(label: 'Stop finding', icon: Icons.close_rounded, onPressed: _stop),
          ] else ...[
            const SectionHeader('What are you looking for?'),
            Wrap(spacing: 10, runSpacing: 10, children: [
              for (final t in targets)
                ActionChip(
                  label: Padding(
                    padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 4),
                    child: Text(spokenLabel(t), style: text.titleMedium),
                  ),
                  onPressed: running ? () => _start(t) : null,
                ),
            ]),
            const SizedBox(height: 16),
            Text(
              'AuraSight tells you where the object is and buzzes when it is straight ahead. '
              'It can only find things the object detector knows.',
              style: text.bodyMedium,
            ),
          ],
        ]);
      },
    );
  }
}
