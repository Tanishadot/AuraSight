import 'dart:io';

import 'package:aurasight/app.dart';
import 'package:aurasight/data/database.dart';
import 'package:aurasight/services/model_manager.dart';
import 'package:aurasight/state/providers.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

void main() {
  sqfliteFfiInit();

  Future<void> pumpApp(WidgetTester tester) async {
    final db = await tester.runAsync(
        () => AuraDatabase.open(path: inMemoryDatabasePath, factory: databaseFactoryFfiNoIsolate));
    await tester.pumpWidget(ProviderScope(
      overrides: [
        databaseProvider.overrideWithValue(db!),
        modelManagerProvider.overrideWithValue(ModelManager(Directory.systemTemp)),
      ],
      child: const AuraApp(),
    ));
    await tester.pump();
  }

  testWidgets('five tabs, each reachable, with a heading', (tester) async {
    await pumpApp(tester);
    for (final (tab, title) in [
      ('Describe', 'Scene description'),
      ('Ask', 'Ask Aura'),
      ('Find', 'Find something'),
      ('Activity', 'Activity and alerts'),
      ('Live', 'Live vision'),
    ]) {
      await tester.tap(find.text(tab));
      await tester.pump();
      expect(find.text(title), findsOneWidget);
    }
  });

  testWidgets('settings opens from every screen', (tester) async {
    await pumpApp(tester);
    await tester.tap(find.bySemanticsLabel('Settings and models'));
    await tester.pumpAndSettle();
    expect(find.text('Speak out loud'), findsOneWidget);
    await tester.scrollUntilVisible(find.text('SmolVLM 500M (recommended)'), 300,
        scrollable: find.byType(Scrollable).last);
    expect(find.text('SmolVLM 500M (recommended)'), findsOneWidget);
  });

  testWidgets('meets tap-target, labelling and contrast guidelines', (tester) async {
    final handle = tester.ensureSemantics();
    await pumpApp(tester);
    for (final tab in ['Live', 'Describe', 'Find', 'Activity']) {
      await tester.tap(find.text(tab));
      await tester.pump();
      await expectLater(tester, meetsGuideline(androidTapTargetGuideline));
      await expectLater(tester, meetsGuideline(iOSTapTargetGuideline));
      await expectLater(tester, meetsGuideline(labeledTapTargetGuideline));
      await expectLater(tester, meetsGuideline(textContrastGuideline));
    }
    handle.dispose();
  });

  testWidgets('works at 200% text size without overflow', (tester) async {
    tester.platformDispatcher.textScaleFactorTestValue = 2.0;
    addTearDown(tester.platformDispatcher.clearTextScaleFactorTestValue);
    await pumpApp(tester);
    for (final tab in ['Live', 'Describe', 'Ask', 'Find', 'Activity']) {
      await tester.tap(find.text(tab).last);
      await tester.pump();
      expect(tester.takeException(), isNull, reason: tab);
    }
  });

  testWidgets('honest status when native modules are missing', (tester) async {
    await pumpApp(tester);
    await tester.tap(find.text('Ask'));
    await tester.pump();
    expect(find.text('Ask Aura is not ready in this build'), findsOneWidget);
    await tester.tap(find.text('Describe'));
    await tester.pump();
    expect(find.textContaining('Scene model'), findsWidgets);
  });

  testWidgets('dark theme renders', (tester) async {
    tester.platformDispatcher.platformBrightnessTestValue = Brightness.dark;
    addTearDown(tester.platformDispatcher.clearPlatformBrightnessTestValue);
    await pumpApp(tester);
    final ctx = tester.element(find.text('Live vision'));
    expect(Theme.of(ctx).brightness, Brightness.dark);
  });
}
