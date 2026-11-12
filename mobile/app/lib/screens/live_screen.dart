import 'package:aura_core/aura_core.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../services/camera_service.dart';
import '../services/live_pipeline.dart';
import '../state/providers.dart';
import '../widgets/common.dart';
import '../widgets/detector_card.dart';

/// Home: live camera, detected objects and the two everyday actions.
class LiveScreen extends ConsumerWidget {
  const LiveScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final camera = ref.watch(cameraServiceProvider);
    final pipeline = ref.watch(livePipelineProvider);
    final lifecycle = ref.watch(cameraLifecycleProvider);
    return ValueListenableBuilder<CameraState>(
      valueListenable: camera.state,
      builder: (context, cam, _) => ValueListenableBuilder<LiveSnapshot>(
        valueListenable: pipeline.snapshot,
        builder: (context, snap, _) {
          final running = cam.status == CameraStatus.running;
          final confirmed = [
            for (final o in snap.objects)
              if (o.confirmed) o
          ];
          return ScreenBody(children: [
            _Preview(camera: camera, snap: snap, running: running),
            const SizedBox(height: 16),
            HeroButton(
              label: running ? 'Stop camera' : (cam.status == CameraStatus.starting ? 'Starting…' : 'Start camera'),
              icon: running ? Icons.videocam_off_rounded : Icons.videocam_rounded,
              busy: cam.status == CameraStatus.starting,
              onPressed: () => running ? lifecycle.stop() : lifecycle.start(),
            ),
            const SizedBox(height: 12),
            HeroButton(
              label: "What's around me?",
              icon: Icons.record_voice_over_rounded,
              hint: 'Speaks the objects the detector has confirmed',
              onPressed: running ? () => _speakSummary(ref, confirmed) : null,
            ),
            if (cam.status == CameraStatus.noPermission ||
                cam.status == CameraStatus.error ||
                cam.status == CameraStatus.unavailable) ...[
              const SizedBox(height: 16),
              StatusCard(
                  icon: Icons.no_photography_rounded,
                  title: 'Camera unavailable',
                  body: cam.message,
                  tone: Tone.danger),
            ],
            const SizedBox(height: 16),
            const DetectorCard(compact: true),
            SectionHeader(confirmed.isEmpty ? 'Objects' : 'Objects (${confirmed.length})'),
            if (!running)
              Text('Start the camera to hear what is around you.', style: Theme.of(context).textTheme.bodyLarge)
            else if (confirmed.isEmpty)
              Text('Nothing confirmed yet. Hold the phone steady at chest height.',
                  style: Theme.of(context).textTheme.bodyLarge)
            else
              for (final o in confirmed) _ObjectTile(o),
            if (running && snap.detector.available) ...[
              const SizedBox(height: 16),
              _Stats(snap),
            ],
          ]);
        },
      ),
    );
  }

  void _speakSummary(WidgetRef ref, List<ObjectView> confirmed) {
    final det = summarizeObjects(confirmed);
    final text = det.isEmpty ? "I don't see any objects I recognise right now." : 'I can see $det.';
    ref.read(audioProvider).sayAnswer(text);
  }
}

class _Preview extends StatelessWidget {
  final CameraService camera;
  final LiveSnapshot snap;
  final bool running;
  const _Preview({required this.camera, required this.snap, required this.running});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final confirmed = [
      for (final o in snap.objects)
        if (o.confirmed) o
    ];
    final label = !running
        ? 'Camera is off.'
        : confirmed.isEmpty
            ? 'Camera view. No objects confirmed.'
            : 'Camera view. ${summarizeObjects(confirmed, detailed: false)}.';
    final preview = camera.buildPreview();
    return Semantics(
      label: label,
      image: true,
      child: ExcludeSemantics(
        child: ClipRRect(
          borderRadius: BorderRadius.circular(22),
          child: AspectRatio(
            aspectRatio: 3 / 4,
            child: Container(
              color: cs.surfaceContainerHighest,
              child: preview == null
                  ? Center(child: Icon(Icons.videocam_off_rounded, size: 56, color: cs.onSurfaceVariant))
                  : Stack(fit: StackFit.expand, children: [
                      FittedBox(
                          fit: BoxFit.cover,
                          clipBehavior: Clip.hardEdge,
                          child: SizedBox(width: 300, height: 400, child: preview)),
                      CustomPaint(painter: _BoxPainter(snap.objects, cs.primary, cs.onPrimary)),
                    ]),
            ),
          ),
        ),
      ),
    );
  }
}

/// Confirmed objects get solid boxes; unverified ones thin outlines.
class _BoxPainter extends CustomPainter {
  final List<ObjectView> objects;
  final Color color, onColor;
  _BoxPainter(this.objects, this.color, this.onColor);

  @override
  void paint(Canvas canvas, Size size) {
    for (final o in objects) {
      final r =
          Rect.fromLTRB(o.box[0] * size.width, o.box[1] * size.height, o.box[2] * size.width, o.box[3] * size.height);
      final p = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = o.confirmed ? 3 : 1.5
        ..color = o.confirmed ? color : color.withValues(alpha: 0.6);
      canvas.drawRRect(RRect.fromRectAndRadius(r, const Radius.circular(8)), p);
      final tp = TextPainter(
        text: TextSpan(
            text: o.confirmed ? ' ${spokenLabel(o.label)} ' : ' ${spokenLabel(o.label)}? ',
            style: TextStyle(color: onColor, fontSize: 14, fontWeight: FontWeight.w700, backgroundColor: color)),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, r.topLeft.translate(0, -tp.height).clamp(Offset.zero, Offset(size.width, size.height)));
    }
  }

  @override
  bool shouldRepaint(_BoxPainter old) => old.objects != objects;
}

extension on Offset {
  Offset clamp(Offset lo, Offset hi) => Offset(dx.clamp(lo.dx, hi.dx), dy.clamp(lo.dy, hi.dy));
}

class _ObjectTile extends StatelessWidget {
  final ObjectView o;
  const _ObjectTile(this.o);

  @override
  Widget build(BuildContext context) {
    final name = spokenLabel(o.label);
    final cap = name[0].toUpperCase() + name.substring(1);
    return Card(
      margin: const EdgeInsets.only(bottom: 10),
      child: ListTile(
        leading: ExcludeSemantics(child: Icon(_icon(o.spatial.horizontal), size: 30)),
        title: Text(cap),
        subtitle: Text('${o.spatial.phrase()} � ${(o.confidence * 100).round()}% sure'),
        trailing: o.approaching ? const Icon(Icons.warning_amber_rounded, semanticLabel: 'approaching') : null,
      ),
    );
  }

  static IconData _icon(String h) => switch (h) {
        'left' => Icons.west_rounded,
        'right' => Icons.east_rounded,
        _ => Icons.north_rounded,
      };
}

class _Stats extends StatelessWidget {
  final LiveSnapshot snap;
  const _Stats(this.snap);

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context).textTheme.bodyMedium!.copyWith(color: Theme.of(context).colorScheme.onSurfaceVariant);
    final throttle = snap.throttleReason.isEmpty ? '' : ' · slowed: ${snap.throttleReason}';
    return Text(
      'Detector ${snap.detector.backend.toUpperCase()} · ${snap.detectionFps.toStringAsFixed(1)} fps · '
      '${snap.inferenceMs.toStringAsFixed(0)} ms$throttle',
      style: t,
    );
  }
}
