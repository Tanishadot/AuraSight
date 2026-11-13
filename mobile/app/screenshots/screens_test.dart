// Renders the real app screens to PNG (not part of the normal test run).
// Used for documentation; these are renders of the app's own widgets in a
// test environment, not photos of a device.
//
//   FLUTTER_ROOT=<flutter sdk> flutter test --update-goldens screenshots/
import 'dart:io';

import 'package:aurasight/app.dart';
import 'package:aurasight/data/database.dart';
import 'package:aurasight/services/detector_model.dart';
import 'package:aurasight/services/model_manager.dart';
import 'package:aurasight/state/providers.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

Future<void> _loadFonts() async {
  final root = Platform.environment['FLUTTER_ROOT'];
  if (root == null) throw StateError('set FLUTTER_ROOT to the Flutter SDK');
  final dir = Directory('$root/bin/cache/artifacts/material_fonts');
  final roboto = FontLoader('Roboto');
  for (final f in dir.listSync().whereType<File>()) {
    final name = f.uri.pathSegments.last;
    if (name.startsWith('Roboto') && name.endsWith('.ttf') && !name.contains('Condensed')) {
      roboto.addFont(Future.value(ByteData.sublistView(f.readAsBytesSync())));
    }
  }
  await roboto.load();
  final icons = FontLoader('MaterialIcons')
    ..addFont(Future.value(ByteData.sublistView(File('${dir.path}/MaterialIcons-Regular.otf').readAsBytesSync())));
  await icons.load();
}

void main() {
  sqfliteFfiInit();

  setUpAll(_loadFonts);

  Future<void> pumpApp(WidgetTester tester, {Brightness brightness = Brightness.light}) async {
    tester.view.physicalSize = const Size(1080, 2400); // iQOO Z7s resolution
    tester.view.devicePixelRatio = 3;
    tester.platformDispatcher.platformBrightnessTestValue = brightness;
    addTearDown(tester.view.reset);
    addTearDown(tester.platformDispatcher.clearPlatformBrightnessTestValue);
    final dir = await tester.runAsync(() => Directory.systemTemp.createTemp('shots'));
    final db = await tester.runAsync(
        () => AuraDatabase.open(path: inMemoryDatabasePath, factory: databaseFactoryFfiNoIsolate));
    final manifest = await tester.runAsync(() => DetectorManifest.load());
    await tester.pumpWidget(ProviderScope(
      overrides: [
        databaseProvider.overrideWithValue(db!),
        modelManagerProvider.overrideWithValue(ModelManager(dir!)),
        detectorManifestProvider.overrideWithValue(manifest),
      ],
      child: const AuraApp(),
    ));
    // let the detector controller check local storage (real file IO)
    await tester.runAsync(() => Future<void>.delayed(const Duration(milliseconds: 500)));
    await tester.pump();
  }

  Future<void> shot(WidgetTester tester, String name) async {
    await tester.pump(const Duration(milliseconds: 400));
    await expectLater(find.byType(AuraApp), matchesGoldenFile('out/$name.png'));
  }

  testWidgets('live', (tester) async {
    await pumpApp(tester);
    await shot(tester, '1_live');
  });

  testWidgets('describe', (tester) async {
    await pumpApp(tester);
    await tester.tap(find.text('Describe'));
    await shot(tester, '2_describe');
  });

  testWidgets('find', (tester) async {
    await pumpApp(tester);
    await tester.tap(find.text('Find'));
    await shot(tester, '3_find');
  });

  testWidgets('settings models', (tester) async {
    await pumpApp(tester);
    await tester.tap(find.bySemanticsLabel('Settings and models'));
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(find.text('Detector benchmark'), 300, scrollable: find.byType(Scrollable).last);
    await tester.drag(find.byType(Scrollable).last, const Offset(0, 250));
    await shot(tester, '4_settings_models');
  });

  testWidgets('live dark', (tester) async {
    await pumpApp(tester, brightness: Brightness.dark);
    await shot(tester, '5_live_dark');
  });
}
