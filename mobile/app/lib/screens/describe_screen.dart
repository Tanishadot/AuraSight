import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../services/scene_controller.dart';
import '../services/scene_understanding_service.dart';
import '../state/providers.dart';
import '../widgets/common.dart';

class DescribeScreen extends ConsumerWidget {
  const DescribeScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final scene = ref.watch(sceneControllerProvider);
    final vlm = ref.watch(vlmServiceProvider);
    final settings = ref.watch(settingsProvider);
    final speech = ref.watch(speechEventsProvider).valueOrNull;
    final reading = speech?.state == 'speaking' && speech?.kind == 'description';
    final ttsError = speech?.state == 'error' && speech?.kind == 'description' ? speech?.error : null;

    return ValueListenableBuilder<SceneView>(
      valueListenable: scene.view,
      builder: (context, v, _) {
        final generating = v.stage != SceneStage.idle;
        final text = Theme.of(context).textTheme;
        return ScreenBody(children: [
          HeroButton(
            label: generating ? 'Looking…' : 'Describe scene',
            icon: Icons.auto_awesome_rounded,
            busy: generating,
            hint: 'Describes what the camera sees, aloud',
            onPressed: () => scene.describe(),
          ),
          const SizedBox(height: 20),
          Semantics(
            liveRegion: generating,
            child: Container(
              constraints: const BoxConstraints(minHeight: 140),
              child: v.error != null
                  ? Text(v.error!, style: text.bodyLarge!.copyWith(color: Theme.of(context).colorScheme.error))
                  : v.stage == SceneStage.analysing
                      ? Text('Looking at the scene…', style: text.bodyLarge)
                      : v.sentences.isEmpty
                          ? Text('Press Describe scene for a spoken description of what the camera sees.',
                              style: text.bodyLarge)
                          : Text(v.text, style: text.bodyLarge!.copyWith(fontSize: 22, height: 1.5)),
            ),
          ),
          if (v.hasDescription) ...[
            const SizedBox(height: 8),
            Text(
              '${v.source == 'vlm' ? 'Scene model + detector' : 'Object detector only'}'
              '${v.latencyMs > 0 ? ' · ${(v.latencyMs / 1000).toStringAsFixed(1)} s' : ''}',
              style: text.bodyMedium!.copyWith(color: Theme.of(context).colorScheme.onSurfaceVariant),
            ),
          ],
          const SizedBox(height: 16),
          Wrap(spacing: 12, runSpacing: 12, crossAxisAlignment: WrapCrossAlignment.center, children: [
            SecondaryButton(
              label: 'Replay',
              icon: Icons.replay_rounded,
              semanticLabel: 'Read the description aloud again',
              onPressed: v.hasDescription && settings.speechEnabled ? scene.replay : null,
            ),
            SecondaryButton(
              label: 'Stop',
              icon: Icons.stop_rounded,
              semanticLabel: 'Stop speaking',
              onPressed: reading || generating ? scene.stop : null,
            ),
            if (reading) const SpeakingIndicator(label: 'Reading aloud'),
          ]),
          if (!settings.speechEnabled) ...[
            const SizedBox(height: 16),
            const StatusCard(
                icon: Icons.volume_off_rounded,
                title: 'Speak out loud is off',
                body: 'Descriptions are shown, not spoken. Turn speech on in Settings.'),
          ],
          if (ttsError != null) ...[
            const SizedBox(height: 16),
            StatusCard(
                icon: Icons.error_outline_rounded,
                title: "The description couldn't be read aloud",
                body: '$ttsError. The text above is complete; try Replay.',
                tone: Tone.danger),
          ],
          const SizedBox(height: 16),
          StreamBuilder<VlmState>(
            stream: vlm.states,
            initialData: vlm.state,
            builder: (context, s) => _VlmStatus(s.data!),
          ),
        ]);
      },
    );
  }
}

class _VlmStatus extends StatelessWidget {
  final VlmState s;
  const _VlmStatus(this.s);

  @override
  Widget build(BuildContext context) {
    final (title, body, tone) = switch (s.status) {
      VlmStatus.ready || VlmStatus.busy => ('Scene model ready', 'Runs on this phone (${s.backend}).', Tone.good),
      VlmStatus.loading => ('Loading the scene model…', null, Tone.neutral),
      VlmStatus.notInstalled => (
          'Scene model not downloaded',
          'Download it once in Settings → Models. Until then, descriptions come from the object detector.',
          Tone.warn
        ),
      VlmStatus.unloaded => (
          'Scene model not loaded',
          'Descriptions come from the object detector until the scene model is installed and loaded.',
          Tone.neutral
        ),
      VlmStatus.unavailable || VlmStatus.error => (
          'Scene model unavailable',
          '${s.detail ?? ''} Descriptions come from the object detector.',
          Tone.warn
        ),
    };
    return StatusCard(icon: Icons.memory_rounded, title: title, body: body, tone: tone);
  }
}
