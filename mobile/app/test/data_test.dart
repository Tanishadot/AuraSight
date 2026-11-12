import 'dart:convert';
import 'dart:io';

import 'package:aura_core/aura_core.dart';
import 'package:aurasight/data/database.dart';
import 'package:aurasight/services/model_manager.dart';
import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

void main() {
  sqfliteFfiInit();

  group('database', () {
    late AuraDatabase db;
    setUp(() async => db = await AuraDatabase.open(path: inMemoryDatabasePath, factory: databaseFactoryFfiNoIsolate));
    tearDown(() => db.close());

    test('settings round-trip and defaults', () async {
      expect((await db.loadSettings()).speechRate, 1.0);
      await db.saveSettings(const AuraSettings().merge({'speechRate': 1.4, 'verbosity': 'detailed'}));
      final s = await db.loadSettings();
      expect(s.speechRate, 1.4);
      expect(s.verbosity, Verbosity.detailed);
    });

    test('corrupt or out-of-range settings fall back to defaults', () async {
      await db.db.insert('settings', {'key': 'settings', 'value': '{not json'});
      expect((await db.loadSettings()).speechEnabled, isTrue);
      final s = const AuraSettings().merge({'speechRate': 9.0, 'minTrackFrames': 'four', 'unknown': 1});
      expect(s.speechRate, 1.0);
      expect(s.minTrackFrames, 4);
    });

    test('activity log, newest first, alerts filter', () async {
      final t = DateTime(2026, 11, 12, 9);
      await db.addActivity(ActivityEntry(time: t, kind: 'entered', message: 'Chair on your left.'));
      await db.addActivity(
          ActivityEntry(time: t, kind: 'obstacle', message: 'Caution: person close, ahead.', priority: 'warning'));
      final all = await db.recentActivity();
      expect(all.map((e) => e.kind), ['obstacle', 'entered']);
      final alerts = await db.recentActivity(alertsOnly: true);
      expect(alerts.single.isAlert, isTrue);
      await db.clearActivity();
      expect(await db.recentActivity(), isEmpty);
    });
  });

  group('model manager', () {
    late Directory dir;
    setUp(() async => dir = await Directory.systemTemp.createTemp('aura_models'));
    tearDown(() => dir.delete(recursive: true));

    final payload = utf8.encode('pretend model weights ' * 1000);
    ModelSpec spec(String sha) => ModelSpec('test', 'Test', 'testing', 'MIT', [
          ModelFile('weights.gguf', 'https://example.invalid/weights.gguf', payload.length, sha),
        ]);

    test('downloads, verifies and reports installed', () async {
      final mm = ModelManager(dir, client: () => MockClient((_) async => http.Response.bytes(payload, 200)));
      final m = spec(sha256.convert(payload).toString());
      final events = await mm.install(m).toList();
      expect(events.last.status, ModelStatus.installed);
      expect(events.any((e) => e.status == ModelStatus.verifying), isTrue);
      expect(await mm.isInstalled(m), isTrue);
      await mm.delete(m);
      expect(await mm.isInstalled(m), isFalse);
    });

    test('a corrupted download is rejected and deleted', () async {
      final bad = List<int>.of(payload)..[10] ^= 0xff;
      final mm = ModelManager(dir, client: () => MockClient((_) async => http.Response.bytes(bad, 200)));
      final m = spec(sha256.convert(payload).toString());
      final last = await mm.install(m).last;
      expect(last.status, ModelStatus.failed);
      expect(last.error, contains('integrity'));
      expect(await mm.isInstalled(m), isFalse);
      expect(File('${mm.fileFor(m, m.files.first).path}.part').existsSync(), isFalse);
    });

    test('resumes a partial download with a Range request', () async {
      final m = spec(sha256.convert(payload).toString());
      final mm0 = ModelManager(dir);
      await mm0.dirFor(m).create(recursive: true);
      await File('${mm0.fileFor(m, m.files.first).path}.part').writeAsBytes(payload.sublist(0, 5000));
      String? range;
      final mm = ModelManager(dir,
          client: () => MockClient((req) async {
                range = req.headers['Range'];
                return http.Response.bytes(payload.sublist(5000), 206);
              }));
      final last = await mm.install(m).last;
      expect(range, 'bytes=5000-');
      expect(last.status, ModelStatus.installed);
    });

    test('HTTP errors are reported', () async {
      final mm = ModelManager(dir, client: () => MockClient((_) async => http.Response('nope', 404)));
      final last = await mm.install(spec('0' * 64)).last;
      expect(last.status, ModelStatus.failed);
      expect(last.error, contains('404'));
    });
  });
}
