import 'dart:convert';

import 'package:flutter/services.dart';

import 'model_manager.dart';

/// The detector model's manifest (`assets/models/detector.json`), written by
/// mobile/tools/export_detector.py when the model was exported and checked
/// in CI against the published file. It pins the download URL, size and
/// SHA-256 and describes the tensors and class labels.
class DetectorManifest {
  final String id, title, purpose, license;
  final ModelFile file;
  final int inputSize;
  final List<int> outputShape;
  final bool normalizedBoxes;
  final double iouThreshold;
  final List<String> labels;

  const DetectorManifest({
    required this.id,
    required this.title,
    required this.purpose,
    required this.license,
    required this.file,
    required this.inputSize,
    required this.outputShape,
    required this.normalizedBoxes,
    required this.iouThreshold,
    required this.labels,
  });

  factory DetectorManifest.fromJson(Map<String, Object?> j) {
    final f = j['file']! as Map<String, Object?>;
    return DetectorManifest(
      id: j['id']! as String,
      title: j['title']! as String,
      purpose: j['purpose']! as String,
      license: j['license']! as String,
      file: ModelFile(f['name']! as String, f['url']! as String, f['bytes']! as int, f['sha256']! as String),
      inputSize: j['inputSize']! as int,
      outputShape: (j['outputShape']! as List).cast<int>(),
      normalizedBoxes: j['normalizedBoxes']! as bool,
      iouThreshold: (j['iouThreshold'] as num?)?.toDouble() ?? 0.5,
      labels: (j['labels']! as List).cast<String>(),
    );
  }

  static const assetPath = 'assets/models/detector.json';

  static Future<DetectorManifest> load([AssetBundle? bundle]) async =>
      DetectorManifest.fromJson(jsonDecode(await (bundle ?? rootBundle).loadString(assetPath)) as Map<String, Object?>);

  ModelSpec get spec => ModelSpec(id, title, purpose, license, [file]);
}
