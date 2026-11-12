import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../data/database.dart';
import '../state/providers.dart';
import '../widgets/common.dart';

final _alertsOnly = StateProvider<bool>((ref) => false);

final _activityProvider = FutureProvider<List<ActivityEntry>>((ref) {
  ref.watch(activityTickProvider);
  return ref.read(databaseProvider).recentActivity(alertsOnly: ref.watch(_alertsOnly));
});

class ActivityScreen extends ConsumerWidget {
  const ActivityScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final alertsOnly = ref.watch(_alertsOnly);
    final items = ref.watch(_activityProvider);
    return ScreenBody(children: [
      Row(children: [
        Expanded(
          child: SegmentedButton<bool>(
            segments: const [
              ButtonSegment(value: false, label: Text('Everything'), icon: Icon(Icons.list_rounded)),
              ButtonSegment(value: true, label: Text('Alerts'), icon: Icon(Icons.warning_amber_rounded)),
            ],
            selected: {alertsOnly},
            onSelectionChanged: (s) => ref.read(_alertsOnly.notifier).state = s.first,
          ),
        ),
      ]),
      const SizedBox(height: 12),
      items.when(
        loading: () => const Padding(padding: EdgeInsets.symmetric(vertical: 24), child: Text('Loading…')),
        error: (e, _) => StatusCard(icon: Icons.error_outline_rounded, title: "Couldn't load activity", body: '$e'),
        data: (list) => list.isEmpty
            ? Padding(
                padding: const EdgeInsets.symmetric(vertical: 24),
                child: Text(alertsOnly ? 'No alerts yet.' : 'Nothing has happened yet.',
                    style: Theme.of(context).textTheme.bodyLarge),
              )
            : Column(children: [for (final e in list) _Entry(e)]),
      ),
      const SizedBox(height: 20),
      SecondaryButton(
        label: 'Clear history',
        icon: Icons.delete_outline_rounded,
        onPressed: () async {
          await ref.read(databaseProvider).clearActivity();
          ref.read(activityTickProvider.notifier).bump();
        },
      ),
    ]);
  }
}

class _Entry extends StatelessWidget {
  final ActivityEntry e;
  const _Entry(this.e);

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final t = TimeOfDay.fromDateTime(e.time).format(context);
    return Card(
      margin: const EdgeInsets.only(bottom: 10),
      child: ListTile(
        leading: ExcludeSemantics(
          child: Icon(
            e.isAlert ? Icons.warning_amber_rounded : (e.priority == 'error' ? Icons.error_outline : Icons.circle_outlined),
            color: e.isAlert || e.priority == 'error' ? cs.error : cs.onSurfaceVariant,
          ),
        ),
        title: Text(e.message),
        subtitle: Text(e.isAlert ? 'Alert · $t' : t),
      ),
    );
  }
}
