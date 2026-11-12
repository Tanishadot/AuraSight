import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../services/scene_understanding_service.dart';
import '../state/providers.dart';
import '../widgets/common.dart';

/// Ask Aura: questions about the camera view, answered on the phone by the
/// scene model, grounded in the detector's confirmed objects.
///
/// Answering needs the on-device scene model (Phase 6) and the question
/// intents ported from the desktop (backend/assistant/qa.py, Phase 7). Until
/// then this screen says so instead of producing made-up answers.
class AskScreen extends ConsumerWidget {
  const AskScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final vlm = ref.watch(vlmServiceProvider);
    return StreamBuilder<VlmState>(
      stream: vlm.states,
      initialData: vlm.state,
      builder: (context, s) {
        final ready = s.data!.status == VlmStatus.ready;
        return ScreenBody(children: [
          TextField(
            enabled: false,
            style: Theme.of(context).textTheme.bodyLarge,
            decoration: InputDecoration(
              labelText: 'Your question',
              hintText: 'For example: Is there a cup on the table?',
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(18)),
              contentPadding: const EdgeInsets.all(18),
            ),
          ),
          const SizedBox(height: 16),
          const HeroButton(label: 'Ask', icon: Icons.send_rounded, onPressed: null),
          const SizedBox(height: 16),
          StatusCard(
            icon: Icons.construction_rounded,
            title: 'Ask Aura is not ready in this build',
            body: ready
                ? 'The scene model is loaded, but question answering arrives in the next update.'
                : 'Asking questions needs the on-device scene model, which this build does not include yet. '
                    "Use \"What's around me?\" on the Live tab for the detector's view.",
          ),
        ]);
      },
    );
  }
}
