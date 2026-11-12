import 'dart:io';

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import '../services/benchmark.dart';
import '../services/detector_controller.dart';
import '../state/providers.dart';
import '../widgets/common.dart';
import '../widgets/detector_card.dart';

class BenchmarkScreen extends ConsumerStatefulWidget {
  const BenchmarkScreen({super.key});
  @override
  ConsumerState<BenchmarkScreen> createState() => _BenchmarkScreenState();
}

class _BenchmarkScreenState extends ConsumerState<BenchmarkScreen> {
  BenchmarkRunner? _runner;
  BenchmarkReport? _report;
  String? _savedTo;
  String? _error;
  int _seconds = 60;

  Future<void> _run() async {
    final runner = BenchmarkRunner(
      detector: ref.read(detectionServiceProvider),
      controller: ref.read(detectorControllerProvider),
      camera: ref.read(cameraServiceProvider),
      pipeline: ref.read(livePipelineProvider),
      testImage: () async => (await rootBundle.load(selfTestAsset)).buffer.asUint8List(),
      settings: () => ref.read(settingsProvider),
    );
    setState(() {
      _runner = runner;
      _report = null;
      _error = null;
      _savedTo = null;
    });
    try {
      final r = await runner.run(sustained: Duration(seconds: _seconds));
      final docs = await getApplicationDocumentsDirectory();
      final f = await BenchmarkRunner.save(r, Directory(p.join(docs.path, 'benchmarks')));
      if (!mounted) return;
      setState(() {
        _report = r;
        _savedTo = f.path;
      });
      ref.read(audioProvider).sayAnswer('Benchmark finished.');
    } catch (e) {
      if (mounted) setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _runner = null);
    }
  }

  @override
  Widget build(BuildContext context) {
    final status = ref.watch(detectorControllerProvider).status;
    final running = _runner != null;
    return Scaffold(
      appBar: AppBar(title: const Text('Detector benchmark')),
      body: ValueListenableBuilder<DetectorStatus>(
        valueListenable: status,
        builder: (context, s, _) => ScreenBody(children: [
          Text(
            'Measures load time, latency (average, median, p95), sustained detection rate, temperature, '
            'battery and memory on this phone. Start the camera first to measure on live frames; '
            'otherwise a bundled photo is used. For battery readings, unplug the charger.',
            style: Theme.of(context).textTheme.bodyLarge,
          ),
          const SizedBox(height: 16),
          if (!s.ready) const DetectorCard(),
          SegmentedButton<int>(
            segments: const [
              ButtonSegment(value: 30, label: Text('30 s')),
              ButtonSegment(value: 60, label: Text('1 min')),
              ButtonSegment(value: 300, label: Text('5 min')),
            ],
            selected: {_seconds},
            onSelectionChanged: running ? null : (v) => setState(() => _seconds = v.first),
          ),
          const SizedBox(height: 16),
          HeroButton(
            label: running ? 'Running…' : 'Run benchmark',
            icon: Icons.speed_rounded,
            busy: running,
            onPressed: s.ready ? _run : null,
          ),
          if (running) ...[
            const SizedBox(height: 12),
            ValueListenableBuilder<String>(
              valueListenable: _runner!.progress,
              builder: (_, t, __) => Semantics(liveRegion: true, child: Text(t)),
            ),
            TextButton(onPressed: _runner!.cancel, child: const Text('Stop early')),
          ],
          if (_error != null) ...[
            const SizedBox(height: 16),
            StatusCard(icon: Icons.error_outline_rounded, title: 'Benchmark failed', body: _error, tone: Tone.danger),
          ],
          if (_report != null) ...[
            const SectionHeader('Result'),
            SelectableText(_report!.toMarkdown(), style: const TextStyle(fontFamily: 'monospace', fontSize: 14)),
            const SizedBox(height: 12),
            Text('Saved to $_savedTo', style: Theme.of(context).textTheme.bodyMedium),
            const SizedBox(height: 12),
            SecondaryButton(
              label: 'Copy report',
              icon: Icons.copy_rounded,
              onPressed: () {
                Clipboard.setData(ClipboardData(text: _report!.toMarkdown()));
                ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Report copied')));
              },
            ),
          ],
        ]),
      ),
    );
  }
}
