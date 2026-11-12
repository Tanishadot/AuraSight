import 'package:flutter/material.dart';

import '../theme/aura_theme.dart';

/// Large primary action (72 dp high), full width by default.
class HeroButton extends StatelessWidget {
  final String label;
  final IconData icon;
  final VoidCallback? onPressed;
  final String? hint; // extra screen-reader hint, e.g. what happens next
  final bool busy;
  const HeroButton({super.key, required this.label, required this.icon, this.onPressed, this.hint, this.busy = false});

  @override
  Widget build(BuildContext context) {
    return Semantics(
      hint: hint,
      child: SizedBox(
        width: double.infinity,
        child: FilledButton.icon(
          style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(kHeroTarget)),
          onPressed: busy ? null : onPressed,
          icon: busy
              ? const SizedBox.square(dimension: 26, child: CircularProgressIndicator(strokeWidth: 3))
              : Icon(icon, size: 30),
          label: Text(label, textAlign: TextAlign.center),
        ),
      ),
    );
  }
}

class SecondaryButton extends StatelessWidget {
  final String label;
  final IconData icon;
  final VoidCallback? onPressed;
  final String? semanticLabel;
  const SecondaryButton({super.key, required this.label, required this.icon, this.onPressed, this.semanticLabel});

  @override
  Widget build(BuildContext context) => Semantics(
        label: semanticLabel,
        excludeSemantics: semanticLabel != null,
        button: true,
        enabled: onPressed != null,
        child: OutlinedButton.icon(onPressed: onPressed, icon: Icon(icon, size: 24), label: Text(label)),
      );
}

enum Tone { neutral, good, warn, danger }

/// A short status message with an icon. Not a live region by default:
/// AuraSight speaks important changes itself.
class StatusCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String? body;
  final Tone tone;
  final Widget? action;
  const StatusCard(
      {super.key, required this.icon, required this.title, this.body, this.tone = Tone.neutral, this.action});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final fg = switch (tone) {
      Tone.danger => cs.error,
      Tone.good || Tone.neutral || Tone.warn => cs.onSurface,
    };
    final bg = switch (tone) {
      Tone.good => cs.primaryContainer,
      Tone.danger => cs.error.withValues(alpha: 0.12),
      Tone.warn => cs.secondary.withValues(alpha: 0.12),
      Tone.neutral => cs.surfaceContainerHighest,
    };
    return Container(
      padding: const EdgeInsets.all(16),
      decoration:
          BoxDecoration(color: bg, borderRadius: BorderRadius.circular(18), border: Border.all(color: cs.outline)),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          ExcludeSemantics(child: Icon(icon, color: fg, size: 28)),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title, style: Theme.of(context).textTheme.titleMedium!.copyWith(color: fg)),
                if (body != null) ...[
                  const SizedBox(height: 4),
                  Text(body!, style: Theme.of(context).textTheme.bodyMedium!.copyWith(color: cs.onSurfaceVariant)),
                ],
                if (action != null) ...[const SizedBox(height: 10), action!],
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// Section heading announced as a heading by TalkBack / VoiceOver.
class SectionHeader extends StatelessWidget {
  final String text;
  const SectionHeader(this.text, {super.key});

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(top: 24, bottom: 10),
        child: Semantics(header: true, child: Text(text, style: Theme.of(context).textTheme.titleLarge)),
      );
}

/// Standard screen body: scrollable, padded, readable width.
class ScreenBody extends StatelessWidget {
  final List<Widget> children;
  const ScreenBody({super.key, required this.children});

  @override
  Widget build(BuildContext context) => SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 640),
            child: ListView(padding: const EdgeInsets.fromLTRB(20, 8, 20, 32), children: children),
          ),
        ),
      );
}

/// Animated "speaking" bars (static when animations are disabled).
class SpeakingIndicator extends StatefulWidget {
  final String label;
  const SpeakingIndicator({super.key, this.label = 'Speaking'});
  @override
  State<SpeakingIndicator> createState() => _SpeakingIndicatorState();
}

class _SpeakingIndicatorState extends State<SpeakingIndicator> with SingleTickerProviderStateMixin {
  late final AnimationController _c = AnimationController(vsync: this, duration: const Duration(milliseconds: 900));

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (MediaQuery.disableAnimationsOf(context)) {
      _c.stop();
    } else if (!_c.isAnimating) {
      _c.repeat(reverse: true);
    }
  }

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Semantics(
      liveRegion: true,
      label: widget.label,
      child: ExcludeSemantics(
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          AnimatedBuilder(
            animation: _c,
            builder: (_, __) => Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                for (var i = 0; i < 3; i++)
                  Container(
                    width: 4,
                    height: 8 + 12 * ((_c.value + i * 0.25) % 1.0),
                    margin: const EdgeInsets.only(right: 3),
                    decoration: BoxDecoration(color: cs.primary, borderRadius: BorderRadius.circular(2)),
                  ),
              ],
            ),
          ),
          const SizedBox(width: 8),
          Text(widget.label, style: Theme.of(context).textTheme.titleMedium!.copyWith(color: cs.primary)),
        ]),
      ),
    );
  }
}
