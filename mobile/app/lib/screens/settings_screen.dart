import 'dart:async';

import 'package:aura_core/aura_core.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../services/model_manager.dart';
import '../services/speech_service.dart';
import '../state/providers.dart';
import '../widgets/common.dart';

final _voicesProvider = FutureProvider<List<Voice>>((ref) => ref.read(ttsEngineProvider).voices());

class SettingsScreen extends ConsumerWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final s = ref.watch(settingsProvider);
    final n = ref.read(settingsProvider.notifier);
    return Scaffold(
      appBar: AppBar(title: const Text('Settings and models')),
      body: ScreenBody(children: [
        const SectionHeader('Speech'),
        _Toggle('Speak out loud', 'Descriptions, answers and alerts are spoken.', s.speechEnabled,
            (v) => n.patch({'speechEnabled': v})),
        _SliderTile('Speech rate', s.speechRate, 0.5, 2.0, 15, (v) => '${v.toStringAsFixed(1)}×',
            (v) => n.patch({'speechRate': double.parse(v.toStringAsFixed(1))})),
        _SliderTile('Volume', s.speechVolume, 0, 1, 20, (v) => '${(v * 100).round()}%',
            (v) => n.patch({'speechVolume': double.parse(v.toStringAsFixed(2))})),
        _VoicePicker(selected: s.voiceId, onChanged: (id) => n.patch({'voiceId': id})),
        const SizedBox(height: 8),
        SecondaryButton(
          label: 'Test voice',
          icon: Icons.volume_up_rounded,
          onPressed: () => ref.read(audioProvider).sayAnswer(
              'This is AuraSight. Speech rate ${s.speechRate.toStringAsFixed(1)}, volume ${(s.speechVolume * 100).round()} percent.'),
        ),
        const SectionHeader('Announcements'),
        _Toggle('Mute announcements', 'Safety warnings are still spoken when obstacle alerts are on.',
            s.announcementsMuted, (v) => n.patch({'announcementsMuted': v})),
        _Toggle('Obstacle alerts', 'Warns about people and objects close ahead.', s.obstacleAlerts,
            (v) => n.patch({'obstacleAlerts': v})),
        _Toggle('New objects', 'Says when something new comes into view.', s.announceNewObjects,
            (v) => n.patch({'announceNewObjects': v})),
        _Toggle('Objects leaving', null, s.announceDepartures, (v) => n.patch({'announceDepartures': v})),
        _Toggle('Big view changes', null, s.announceSceneChanges, (v) => n.patch({'announceSceneChanges': v})),
        _Toggle('Detailed positions', 'Adds near, middle distance or far.', s.verbosity == Verbosity.detailed,
            (v) => n.patch({'verbosity': v ? 'detailed' : 'brief'})),
        _SliderTile('Repeat the same object after', s.announceCooldownS, 0, 120, 24, (v) => '${v.round()} s',
            (v) => n.patch({'announceCooldownS': v.roundToDouble()})),
        const SectionHeader('Feedback'),
        _Toggle('Vibration', 'Buzzes on warnings and when a searched object is straight ahead.', s.hapticsEnabled,
            (v) => n.patch({'hapticsEnabled': v})),
        const SectionHeader('Detection'),
        _SliderTile('Highest detection rate', s.detectionFps, 1, 15, 14, (v) => '${v.round()} per second',
            (v) => n.patch({'detectionFps': v.roundToDouble()})),
        _SliderTile('Confidence needed', s.confidenceThreshold, 0.3, 0.8, 10, (v) => v.toStringAsFixed(2),
            (v) => n.patch({'confidenceThreshold': double.parse(v.toStringAsFixed(2))})),
        Text('The rate drops automatically when the phone is warm, low on memory or describing a scene.',
            style: Theme.of(context).textTheme.bodyMedium),
        const SectionHeader('Models'),
        for (final m in modelCatalog) _ModelTile(m),
        const SectionHeader('Privacy'),
        Text(
          'Everything AuraSight sees and hears is processed on this phone. The internet is only used to '
          'download models you choose to install. No images, audio or text are sent anywhere.',
          style: Theme.of(context).textTheme.bodyLarge,
        ),
        const SectionHeader('Licences'),
        Text(
          'Object detection uses YOLO11n by Ultralytics (AGPL-3.0). Scene descriptions use SmolVLM by Hugging Face '
          '(Apache-2.0) running on llama.cpp (MIT).',
          style: Theme.of(context).textTheme.bodyMedium,
        ),
        const SizedBox(height: 8),
        SecondaryButton(
            label: 'Open-source licences', icon: Icons.description_outlined, onPressed: () => showLicensePage(context: context)),
      ]),
    );
  }
}

class _Toggle extends StatelessWidget {
  final String title;
  final String? subtitle;
  final bool value;
  final ValueChanged<bool> onChanged;
  const _Toggle(this.title, this.subtitle, this.value, this.onChanged);

  @override
  Widget build(BuildContext context) => SwitchListTile(
        contentPadding: EdgeInsets.zero,
        title: Text(title),
        subtitle: subtitle == null ? null : Text(subtitle!),
        value: value,
        onChanged: onChanged,
      );
}

class _SliderTile extends StatelessWidget {
  final String label;
  final double value, min, max;
  final int divisions;
  final String Function(double) format;
  final ValueChanged<double> onChanged;
  const _SliderTile(this.label, this.value, this.min, this.max, this.divisions, this.format, this.onChanged);

  @override
  Widget build(BuildContext context) {
    final v = value.clamp(min, max);
    return Padding(
      padding: const EdgeInsets.only(top: 8),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        ExcludeSemantics(
          child: Row(children: [
            Expanded(child: Text(label, style: Theme.of(context).textTheme.titleMedium)),
            Text(format(v), style: Theme.of(context).textTheme.titleMedium),
          ]),
        ),
        Slider(
          value: v,
          min: min,
          max: max,
          divisions: divisions,
          label: format(v),
          semanticFormatterCallback: (x) => '$label ${format(x)}',
          onChanged: onChanged,
        ),
      ]),
    );
  }
}

class _VoicePicker extends ConsumerWidget {
  final String? selected;
  final ValueChanged<String?> onChanged;
  const _VoicePicker({required this.selected, required this.onChanged});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final voices = ref.watch(_voicesProvider);
    return voices.when(
      loading: () => const SizedBox.shrink(),
      error: (_, __) => const SizedBox.shrink(),
      data: (list) {
        final offline = [for (final v in list) if (v.offline) v];
        if (offline.isEmpty) {
          return const Padding(
            padding: EdgeInsets.only(top: 8),
            child: StatusCard(
              icon: Icons.record_voice_over_outlined,
              title: 'No offline voice found',
              body: 'Install a voice for your language in the system text-to-speech settings so speech works offline.',
              tone: Tone.warn,
            ),
          );
        }
        return Padding(
          padding: const EdgeInsets.only(top: 8),
          child: DropdownButtonFormField<String?>(
            initialValue: offline.any((v) => v.id == selected) ? selected : null,
            isExpanded: true,
            decoration: InputDecoration(
              labelText: 'Voice (offline voices only)',
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(18)),
            ),
            items: [
              const DropdownMenuItem(value: null, child: Text('System default')),
              for (final v in offline) DropdownMenuItem(value: v.id, child: Text('${v.name} (${v.language})')),
            ],
            onChanged: onChanged,
          ),
        );
      },
    );
  }
}

class _ModelTile extends ConsumerStatefulWidget {
  final ModelSpec spec;
  const _ModelTile(this.spec);
  @override
  ConsumerState<_ModelTile> createState() => _ModelTileState();
}

class _ModelTileState extends ConsumerState<_ModelTile> {
  ModelProgress? _progress;
  bool? _installed;
  StreamSubscription<ModelProgress>? _sub;

  ModelManager get _mm => ref.read(modelManagerProvider);

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  Future<void> _refresh() async {
    final ok = await _mm.isInstalled(widget.spec);
    if (mounted) setState(() => _installed = ok);
  }

  void _install() {
    _sub = _mm.install(widget.spec).listen((p) {
      if (!mounted) return;
      setState(() => _progress = p);
      if (p.status == ModelStatus.installed || p.status == ModelStatus.failed) {
        _refresh();
        ref.read(audioProvider).sayAnswer(p.status == ModelStatus.installed
            ? '${widget.spec.title} is installed.'
            : 'The download failed. ${p.error ?? ''}');
      }
    });
  }

  @override
  void dispose() {
    _sub?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final m = widget.spec;
    final mb = (m.totalBytes / (1024 * 1024)).round();
    final p = _progress;
    final active = p != null && (p.status == ModelStatus.downloading || p.status == ModelStatus.verifying);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(m.title, style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 4),
          Text('${m.purpose} $mb MB · ${m.license}.', style: Theme.of(context).textTheme.bodyMedium),
          const SizedBox(height: 12),
          if (active) ...[
            Semantics(
              label: p.status == ModelStatus.verifying ? 'Checking the download' : 'Downloading',
              value: '${(p.fraction * 100).round()} percent',
              child: LinearProgressIndicator(value: p.fraction, minHeight: 10),
            ),
            const SizedBox(height: 8),
            Row(children: [
              Expanded(
                child: Text(p.status == ModelStatus.verifying
                    ? 'Checking integrity…'
                    : '${(p.receivedBytes / 1048576).round()} of $mb MB'),
              ),
              TextButton(onPressed: () => _mm.cancel(m), child: const Text('Cancel')),
            ]),
          ] else if (_installed == true)
            Row(children: [
              const Expanded(child: Text('Installed and verified')),
              SecondaryButton(
                label: 'Delete',
                icon: Icons.delete_outline_rounded,
                onPressed: () async {
                  await _mm.delete(m);
                  await _refresh();
                },
              ),
            ])
          else ...[
            if (p?.status == ModelStatus.failed)
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Text(p!.error ?? 'Download failed.', style: TextStyle(color: Theme.of(context).colorScheme.error)),
              ),
            FilledButton.icon(
              onPressed: _installed == null ? null : _install,
              icon: const Icon(Icons.download_rounded),
              label: Text('Download ($mb MB, Wi-Fi recommended)'),
            ),
          ],
        ]),
      ),
    );
  }
}
