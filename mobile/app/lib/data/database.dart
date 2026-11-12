import 'dart:convert';

import 'package:aura_core/aura_core.dart';
import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';

/// Local SQLite store: settings and the activity / alert log.
/// Nothing in here leaves the phone.
class AuraDatabase {
  final Database db;
  AuraDatabase._(this.db);

  static const _version = 1;

  static Future<AuraDatabase> open({String? path, DatabaseFactory? factory}) async {
    final f = factory ?? databaseFactory;
    final dbPath = path ?? p.join(await f.getDatabasesPath(), 'aurasight.db');
    final db = await f.openDatabase(dbPath, options: OpenDatabaseOptions(version: _version, onCreate: _create));
    return AuraDatabase._(db);
  }

  static Future<void> _create(Database db, int version) async {
    await db.execute('CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)');
    await db.execute('''
      CREATE TABLE activity (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER NOT NULL,
        kind TEXT NOT NULL,
        message TEXT NOT NULL,
        priority TEXT NOT NULL
      )''');
    await db.execute('CREATE INDEX activity_ts ON activity (ts DESC)');
  }

  // -- settings ---------------------------------------------------------------
  Future<AuraSettings> loadSettings() async {
    final rows = await db.query('settings', where: 'key = ?', whereArgs: ['settings']);
    if (rows.isEmpty) return const AuraSettings();
    try {
      final map = jsonDecode(rows.first['value']! as String) as Map<String, Object?>;
      return AuraSettings.fromJson(map);
    } on FormatException {
      return const AuraSettings(); // corrupt -> defaults
    }
  }

  Future<void> saveSettings(AuraSettings s) =>
      db.insert('settings', {'key': 'settings', 'value': jsonEncode(s.toJson())},
          conflictAlgorithm: ConflictAlgorithm.replace);

  // -- activity log -------------------------------------------------------------
  static const maxActivity = 2000;

  Future<int> addActivity(ActivityEntry e) async {
    final id = await db.insert('activity', e.toRow());
    if (id % 200 == 0) {
      // keep the log bounded
      await db.delete('activity', where: 'id <= ?', whereArgs: [id - maxActivity]);
    }
    return id;
  }

  Future<List<ActivityEntry>> recentActivity({int limit = 200, bool alertsOnly = false}) async {
    final rows = await db.query('activity',
        where: alertsOnly ? 'priority = ?' : null,
        whereArgs: alertsOnly ? ['warning'] : null,
        orderBy: 'id DESC',
        limit: limit);
    return rows.map(ActivityEntry.fromRow).toList();
  }

  Future<void> clearActivity() => db.delete('activity');

  Future<void> close() => db.close();
}

class ActivityEntry {
  final int? id;
  final DateTime time;
  final String kind; // entered | left | obstacle | approaching | scene_change | description | answer | error ...
  final String message;
  final String priority; // info | warning | error

  const ActivityEntry({this.id, required this.time, required this.kind, required this.message, this.priority = 'info'});

  bool get isAlert => priority == 'warning';

  Map<String, Object?> toRow() =>
      {'ts': time.millisecondsSinceEpoch, 'kind': kind, 'message': message, 'priority': priority};

  static ActivityEntry fromRow(Map<String, Object?> r) => ActivityEntry(
        id: r['id'] as int?,
        time: DateTime.fromMillisecondsSinceEpoch(r['ts']! as int),
        kind: r['kind']! as String,
        message: r['message']! as String,
        priority: r['priority']! as String,
      );
}
