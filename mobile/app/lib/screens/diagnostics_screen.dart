import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../services/benchmark.dart';
import '../services/diagnostics.dart';
import '../state/providers.dart';
import '../widgets/common.dart';
import '../widgets/detector_card.dart';

/// Staged detector hardware check and crash reports (Settings → Detector diagnostics).
class DiagnosticsScreen extends ConsumerStatefulWidget {
  const DiagnosticsScreen({super.key});
  @override
  ConsumerState<DiagnosticsScreen> createState() => _DiagnosticsScreenState();
}

class _DiagnosticsScreenState extends ConsumerState<DiagnosticsScreen> {
  late final DiagnosticsRunner _runner;
  CrashReport? _crash;
  SystemSnapshot? _device;

  @override
  void initState() {
    super.initState();
    _runner = DiagnosticsRunner(
      detector: ref.read(detectionServiceProvider),
      controller: ref.read(detectorControllerProvider),
      models: ref.read(modelManagerProvider),
      manifest: ref.read(detectorManifestProvider),
      camera: ref.read(cameraServiceProvider),
      stopLive: ref.read(cameraLifecycleProvider).stop,
      testImage: () async => (await rootBundle.load(selfTestAsset)).buffer.asUint8List(),
    );
    _refresh();
  }

  Future<void> _refresh() async {
    final crash = await CrashReport.read(ref.read(detectorGuardProvider));
    final device = await SystemSnapshot.read();
    if (mounted) {
      setState(() {
        _crash = crash;
        _device = device;
      });
    }
  }

  @override
  void dispose() {
    _runner.dispose();
    super.dispose();
  }

  Future<void> _copy(String text, String what) async {
    await Clipboard.setData(ClipboardData(text: text));
    if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('$what copied')));
  }

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final controller = ref.watch(detectorControllerProvider);
    return Scaffold(
      appBar: AppBar(title: const Text('Detector diagnostics')),
      body: ValueListenableBuilder(
        valueListenable: _runner.running,
        builder: (context, running, _) => ValueListenableBuilder(
          valueListenable: _runner.results,
          builder: (context, results, _) => ScreenBody(children: [
            Text(
              'Run the stages in order. Each one only starts after the previous one passed, '
              'so if the app closes you know which step did it. Reopen this screen afterwards: '
              'the crash report below says why the app closed.',
              style: text.bodyLarge,
            ),
            const SectionHeader('Hardware check'),
            for (final s in DiagStage.values)
              _StageTile(
                  s,
                  results[s],
                  running == s,
                  _runner.canRun(s)
                      ? () async {
                          await _runner.run(s);
                          await _refresh();
                        }
                      : null),
            const SizedBox(height: 12),
            SecondaryButton(
              label: 'Copy hardware check report',
              icon: Icons.copy_rounded,
              onPressed: results.isEmpty
                  ? null
                  : () => _copy(_runner.report(_device ?? SystemSnapshot(DateTime.now())), 'Report'),
            ),
            const SectionHeader('Detector'),
            const DetectorCard(),
            const SizedBox(height: 12),
            if (controller.guardStatus.gpuBlocked != null) ...[
              StatusCard(
                icon: Icons.block_rounded,
                title: 'GPU switched off',
                body: controller.guardStatus.gpuBlocked,
                tone: Tone.warn,
                action: OutlinedButton(
                  onPressed: () async {
                    await controller.clearGpuBlock();
                    setState(() {});
                  },
                  child: const Text('Allow the GPU again'),
                ),
              ),
              const SizedBox(height: 12),
            ],
            SecondaryButton(
              label: 'Try the GPU (may close the app)',
              icon: Icons.memory_rounded,
              onPressed: (results[DiagStage.fullSelfTest]?.ok ?? false) && controller.guardStatus.gpuBlocked == null
                  ? () => controller.tryGpu()
                  : null,
            ),
            Text(
              'Only after stage 6 passed. If the GPU crashes the app, the next start reports it and keeps the GPU off.',
              style: text.bodyMedium,
            ),
            const SectionHeader('Crash report'),
            if (_crash == null)
              const LinearProgressIndicator()
            else ...[
              StatusCard(
                icon: _crash!.hasCrash ? Icons.report_problem_rounded : Icons.check_circle_outline_rounded,
                title: _crash!.hasCrash ? 'A previous run ended abnormally' : 'No crash recorded',
                body: _crash!.exits.isEmpty
                    ? 'No process exits recorded.'
                    : _crash!.exits.take(3).map((e) => '${e['reason']}: ${e['description'] ?? ''}').join('\n'),
                tone: _crash!.hasCrash ? Tone.danger : Tone.good,
              ),
              const SizedBox(height: 12),
              SecondaryButton(
                label: 'Copy crash report',
                icon: Icons.copy_all_rounded,
                onPressed: () => _copy(_crash!.toText(), 'Crash report'),
              ),
            ],
          ]),
        ),
      ),
    );
  }
}

class _StageTile extends StatelessWidget {
  final DiagStage stage;
  final StageResult? result;
  final bool running;
  final VoidCallback? onRun;
  const _StageTile(this.stage, this.result, this.running, this.onRun);

  @override
  Widget build(BuildContext context) {
    final r = result;
    final cs = Theme.of(context).colorScheme;
    return Card(
      margin: const EdgeInsets.only(bottom: 10),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            ExcludeSemantics(
              child: Icon(
                r == null ? Icons.radio_button_unchecked : (r.ok ? Icons.check_circle_rounded : Icons.cancel_rounded),
                color: r == null ? cs.onSurfaceVariant : (r.ok ? cs.primary : cs.error),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(child: Text(stage.title, style: Theme.of(context).textTheme.titleMedium)),
          ]),
          if (r != null) ...[
            const SizedBox(height: 6),
            Text('${r.ok ? 'Passed' : 'Failed'}: ${r.summary}'),
            Text('Latency: ${r.latencyLine}'),
            Text('Memory: ${r.memoryLine}'),
          ],
          const SizedBox(height: 8),
          FilledButton(
            onPressed: running ? null : onRun,
            child: Text(running ? 'Running…' : (r == null ? 'Run' : 'Run again')),
          ),
        ]),
      ),
    );
  }
}
