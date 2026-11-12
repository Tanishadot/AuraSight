import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import 'app.dart';
import 'data/database.dart';
import 'services/model_manager.dart';
import 'state/providers.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // Positions ("on your left") assume the phone is held upright.
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  final db = await AuraDatabase.open();
  final settings = await db.loadSettings();
  final support = await getApplicationSupportDirectory();
  final models = ModelManager(Directory(p.join(support.path, 'models')));
  runApp(ProviderScope(
    overrides: [
      databaseProvider.overrideWithValue(db),
      initialSettingsProvider.overrideWithValue(settings),
      modelManagerProvider.overrideWithValue(models),
    ],
    child: const AuraApp(),
  ));
}
