import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'screens/activity_screen.dart';
import 'screens/ask_screen.dart';
import 'screens/describe_screen.dart';
import 'screens/find_screen.dart';
import 'screens/live_screen.dart';
import 'screens/settings_screen.dart';
import 'state/providers.dart';
import 'theme/aura_theme.dart';

class AuraApp extends StatelessWidget {
  const AuraApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'AuraSight',
        debugShowCheckedModeBanner: false,
        theme: auraTheme(Brightness.light),
        darkTheme: auraTheme(Brightness.dark),
        themeMode: ThemeMode.system,
        home: const HomeShell(),
      );
}

class _Tab {
  final String label;
  final String title;
  final IconData icon, selectedIcon;
  final Widget screen;
  const _Tab(this.label, this.title, this.icon, this.selectedIcon, this.screen);
}

const _tabs = [
  _Tab('Live', 'Live vision', Icons.visibility_outlined, Icons.visibility_rounded, LiveScreen()),
  _Tab('Describe', 'Scene description', Icons.auto_awesome_outlined, Icons.auto_awesome_rounded, DescribeScreen()),
  _Tab('Ask', 'Ask Aura', Icons.chat_bubble_outline_rounded, Icons.chat_bubble_rounded, AskScreen()),
  _Tab('Find', 'Find something', Icons.search_rounded, Icons.manage_search_rounded, FindScreen()),
  _Tab('Activity', 'Activity and alerts', Icons.history_rounded, Icons.history_toggle_off_rounded, ActivityScreen()),
];

/// Five tabs plus Settings. Focus order: title, settings, screen content,
/// then the tab bar (the platform default for bottom navigation).
class HomeShell extends ConsumerStatefulWidget {
  const HomeShell({super.key});
  @override
  ConsumerState<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends ConsumerState<HomeShell> with WidgetsBindingObserver {
  int _index = 0;

  @override
  void initState() {
    super.initState();
    ref.read(audioProvider); // start the speech engine early
    ref.read(detectorControllerProvider).init(); // loads the detector if it is already downloaded
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) => ref.read(cameraLifecycleProvider).onAppLifecycle(state);

  @override
  Widget build(BuildContext context) {
    final tab = _tabs[_index];
    return Scaffold(
      appBar: AppBar(
        title: Semantics(header: true, child: Text(tab.title)),
        actions: [
          IconButton(
            iconSize: 30,
            icon: const Icon(Icons.settings_outlined, semanticLabel: 'Settings and models'),
            onPressed: () =>
                Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => const SettingsScreen())),
          ),
          IconButton(
            iconSize: 30,
            icon: const Icon(Icons.volume_off_outlined, semanticLabel: 'Stop speaking'),
            onPressed: () => ref.read(sceneControllerProvider).stop(),
          ),
        ],
      ),
      // IndexedStack keeps each screen's state (e.g. an active search) when switching tabs.
      body: IndexedStack(index: _index, children: [for (final t in _tabs) t.screen]),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
        onDestinationSelected: (i) => setState(() => _index = i),
        destinations: [
          for (final t in _tabs)
            // tooltip '' : the label is always visible and read by screen readers
            NavigationDestination(icon: Icon(t.icon), selectedIcon: Icon(t.selectedIcon), label: t.label, tooltip: ''),
        ],
      ),
    );
  }
}
