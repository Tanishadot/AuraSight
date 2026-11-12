import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../services/detector_controller.dart';
import '../state/providers.dart';
import 'common.dart';

String _mb(int bytes) => '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';

/// The detector's real state with the action that moves it forward:
/// Not downloaded -> Downloading -> Verifying -> Loading -> Ready, or Error + Retry.
/// [compact] hides the card once the detector is ready (Live screen).
class DetectorCard extends ConsumerWidget {
  final bool compact;
  const DetectorCard({super.key, this.compact = false});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final c = ref.watch(detectorControllerProvider);
    final size = ref.watch(detectorManifestProvider)?.file.bytes ?? 0;
    return ValueListenableBuilder<DetectorStatus>(
      valueListenable: c.status,
      builder: (context, s, _) {
        final text = Theme.of(context).textTheme;
        if (compact && s.ready) {
          final cs = Theme.of(context).colorScheme;
          return Row(children: [
            ExcludeSemantics(child: Icon(Icons.check_circle_rounded, color: cs.primary)),
            const SizedBox(width: 8),
            Expanded(
              child: Text('Object detector ready · ${s.info!.backend.toUpperCase()}',
                  style: text.titleMedium!.copyWith(color: cs.primary)),
            ),
          ]);
        }
        switch (s.phase) {
          case DetectorPhase.checking:
            return const StatusCard(icon: Icons.hourglass_top_rounded, title: 'Checking the object detector…');
          case DetectorPhase.notDownloaded:
            return StatusCard(
              icon: Icons.download_rounded,
              title: 'Object detector not downloaded',
              body: 'AuraSight needs a one-time ${_mb(size)} download to recognise objects. '
                  'After that, detection works offline on this phone.',
              tone: Tone.warn,
              action: FilledButton.icon(
                onPressed: c.download,
                icon: const Icon(Icons.download_rounded),
                label: Text('Download (${_mb(size)})'),
              ),
            );
          case DetectorPhase.downloading:
          case DetectorPhase.verifying:
            final verifying = s.phase == DetectorPhase.verifying;
            final pct = (s.fraction * 100).round();
            return StatusCard(
              icon: verifying ? Icons.verified_outlined : Icons.downloading_rounded,
              title: verifying ? 'Checking the download…' : 'Downloading the object detector…',
              body: verifying
                  ? 'Comparing its SHA-256 checksum with the published one.'
                  : '${_mb(s.receivedBytes)} of ${_mb(s.totalBytes)}',
              action: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Semantics(
                  label: verifying ? 'Checking' : 'Download progress',
                  value: '$pct percent',
                  child: LinearProgressIndicator(value: verifying && s.totalBytes == 0 ? null : s.fraction, minHeight: 10),
                ),
                if (!verifying) ...[
                  const SizedBox(height: 8),
                  TextButton(onPressed: c.cancelDownload, child: const Text('Cancel download')),
                ],
              ]),
            );
          case DetectorPhase.loading:
            return const StatusCard(
              icon: Icons.memory_rounded,
              title: 'Loading the object detector…',
              body: 'Starting it on this phone and running a self-test.',
            );
          case DetectorPhase.ready:
            final i = s.info!;
            return StatusCard(
              icon: Icons.check_circle_outline_rounded,
              title: 'Object detector ready',
              body: 'Running on the ${i.backend.toUpperCase()} · loaded in ${i.loadMs.toStringAsFixed(0)} ms · '
                  'self-test: ${s.selfTest}'
                  '${i.fallbackReason != null ? '\nUsing the CPU because: ${i.fallbackReason}' : ''}',
              tone: Tone.good,
            );
          case DetectorPhase.error:
            return StatusCard(
              icon: Icons.error_outline_rounded,
              title: 'Object detector unavailable',
              body: '${s.error}\n${s.recovery ?? ''}'.trim(),
              tone: Tone.danger,
              action: Wrap(spacing: 10, runSpacing: 10, children: [
                FilledButton.icon(onPressed: c.retry, icon: const Icon(Icons.refresh_rounded), label: const Text('Retry')),
                if (!compact)
                  OutlinedButton(
                    onPressed: c.deleteModel,
                    child: Text('Delete and download again', style: text.labelLarge),
                  ),
              ]),
            );
        }
      },
    );
  }
}
