import 'dart:async';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;

/// One downloadable file with a pinned size and SHA-256.
class ModelFile {
  final String name;
  final String url;
  final int bytes;
  final String sha256;
  const ModelFile(this.name, this.url, this.bytes, this.sha256);
}

class ModelSpec {
  final String id;
  final String title;
  final String purpose;
  final String license;
  final List<ModelFile> files;
  const ModelSpec(this.id, this.title, this.purpose, this.license, this.files);

  int get totalBytes => files.fold(0, (a, f) => a + f.bytes);
}

/// Models AuraSight can download. The object detector ships inside the app;
/// the scene model is downloaded once (Wi-Fi recommended) and then works offline.
const modelCatalog = [
  ModelSpec(
    'smolvlm-500m-q8',
    'SmolVLM 500M (recommended)',
    'Scene descriptions and answers to questions about the camera view.',
    'Apache-2.0, Hugging Face',
    [
      ModelFile(
          'SmolVLM-500M-Instruct-Q8_0.gguf',
          'https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct-Q8_0.gguf',
          436806912,
          '9d4612de6a42214499e301494a3ecc2be0abdd9de44e663bda63f1152fad1bf4'),
      ModelFile(
          'mmproj-SmolVLM-500M-Instruct-Q8_0.gguf',
          'https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf',
          108783360,
          'd1eb8b6b23979205fdf63703ed10f788131a3f812c7b1f72e0119d5d81295150'),
    ],
  ),
];

enum ModelStatus { notInstalled, downloading, verifying, installed, failed }

class ModelProgress {
  final ModelStatus status;
  final int receivedBytes;
  final int totalBytes;
  final String? error;
  const ModelProgress(this.status, {this.receivedBytes = 0, this.totalBytes = 0, this.error});

  double get fraction => totalBytes == 0 ? 0 : receivedBytes / totalBytes;
}

class ModelIntegrityError implements Exception {
  final String message;
  const ModelIntegrityError(this.message);
  @override
  String toString() => message;
}

/// Downloads, verifies, locates and deletes local models.
/// Downloads resume from a `.part` file; a file only counts as installed
/// after its SHA-256 matches the pinned value.
class ModelManager {
  final Directory root;
  final http.Client Function() _client;
  final _cancelled = <String>{};

  ModelManager(this.root, {http.Client Function()? client}) : _client = client ?? http.Client.new;

  Directory dirFor(ModelSpec m) => Directory(p.join(root.path, m.id));
  File fileFor(ModelSpec m, ModelFile f) => File(p.join(dirFor(m).path, f.name));
  File _marker(ModelSpec m) => File(p.join(dirFor(m).path, '.verified'));

  /// Installed = every file present with the right size and a verification marker.
  Future<bool> isInstalled(ModelSpec m) async {
    if (!await _marker(m).exists()) return false;
    for (final f in m.files) {
      final file = fileFor(m, f);
      if (!await file.exists() || await file.length() != f.bytes) return false;
    }
    return true;
  }

  Future<void> delete(ModelSpec m) async {
    final d = dirFor(m);
    if (await d.exists()) await d.delete(recursive: true);
  }

  void cancel(ModelSpec m) => _cancelled.add(m.id);

  Stream<ModelProgress> install(ModelSpec m) async* {
    _cancelled.remove(m.id);
    await dirFor(m).create(recursive: true);
    final total = m.totalBytes;
    var done = 0;
    try {
      for (final f in m.files) {
        final dest = fileFor(m, f);
        if (await dest.exists() && await dest.length() == f.bytes) {
          done += f.bytes;
          continue;
        }
        final part = File('${dest.path}.part');
        var have = await part.exists() ? await part.length() : 0;
        if (have > f.bytes) {
          await part.delete();
          have = 0;
        }
        final client = _client();
        try {
          final req = http.Request('GET', Uri.parse(f.url));
          if (have > 0) req.headers['Range'] = 'bytes=$have-';
          final res = await client.send(req);
          if (res.statusCode == 200 && have > 0) {
            have = 0; // server ignored the range: start over
          } else if (res.statusCode != 200 && res.statusCode != 206) {
            throw HttpException('Download failed (HTTP ${res.statusCode}) for ${f.name}');
          }
          final sink = part.openWrite(mode: have > 0 ? FileMode.append : FileMode.write);
          var got = have;
          try {
            await for (final chunk in res.stream) {
              if (_cancelled.contains(m.id)) throw const ModelIntegrityError('Download cancelled.');
              sink.add(chunk);
              got += chunk.length;
              yield ModelProgress(ModelStatus.downloading, receivedBytes: done + got, totalBytes: total);
            }
          } finally {
            await sink.close();
          }
        } finally {
          client.close();
        }
        yield ModelProgress(ModelStatus.verifying, receivedBytes: done + f.bytes, totalBytes: total);
        final digest = await sha256.bind(part.openRead()).first;
        if (digest.toString() != f.sha256) {
          await part.delete();
          throw ModelIntegrityError('${f.name} failed its integrity check and was deleted. Try again.');
        }
        await part.rename(dest.path);
        done += f.bytes;
      }
      await _marker(m).writeAsString(DateTime.now().toIso8601String());
      yield ModelProgress(ModelStatus.installed, receivedBytes: total, totalBytes: total);
    } catch (e) {
      yield ModelProgress(ModelStatus.failed, receivedBytes: done, totalBytes: total, error: '$e');
    }
  }
}
