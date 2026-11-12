/// User settings shared by the core logic and persisted by the app.
/// Defaults match backend/config/settings.py; desktop-only options
/// (llama-server URL, webcam index, MJPEG preview) are not carried over.
library;

enum Verbosity { brief, detailed }

class AuraSettings {
  // Detection
  final int inferenceSize;
  final double detectionFps; // upper bound; the scheduler adapts below it
  final double confidenceThreshold;
  final double confusableThreshold;
  final double strongConfidence;

  // Scene model
  final bool vlmEnabled;
  final int vlmImageSize;
  final int vlmMaxTokens;
  final bool autoDescribe;
  final double autoDescribeIntervalS;

  // Speech
  final bool speechEnabled;
  final double speechRate; // multiplier, 1.0 = platform default
  final double speechVolume;
  final String? voiceId;

  // Announcements & alerts
  final bool announcementsMuted;
  final bool announceNewObjects;
  final bool announceDepartures;
  final bool announceSceneChanges;
  final bool obstacleAlerts;
  final double announceCooldownS;
  final int minTrackFrames;
  final double leaveAfterS;
  final Verbosity verbosity;

  // Interaction
  final bool hapticsEnabled;

  const AuraSettings({
    this.inferenceSize = 320,
    this.detectionFps = 6, // upper bound; a mid-range phone stays cool and responsive at this rate
    this.confidenceThreshold = 0.45,
    this.confusableThreshold = 0.5,
    this.strongConfidence = 0.7,
    this.vlmEnabled = true,
    this.vlmImageSize = 512,
    this.vlmMaxTokens = 80,
    this.autoDescribe = false,
    this.autoDescribeIntervalS = 45,
    this.speechEnabled = true,
    this.speechRate = 1.0,
    this.speechVolume = 0.9,
    this.voiceId,
    this.announcementsMuted = false,
    this.announceNewObjects = true,
    this.announceDepartures = false,
    this.announceSceneChanges = true,
    this.obstacleAlerts = true,
    this.announceCooldownS = 20,
    this.minTrackFrames = 4,
    this.leaveAfterS = 2.0,
    this.verbosity = Verbosity.brief,
    this.hapticsEnabled = true,
  });

  static const _ranges = <String, (num, num)>{
    'inferenceSize': (256, 640),
    'detectionFps': (0.5, 30),
    'confidenceThreshold': (0.1, 0.95),
    'confusableThreshold': (0.1, 0.95),
    'strongConfidence': (0.3, 0.99),
    'vlmImageSize': (224, 1024),
    'vlmMaxTokens': (16, 512),
    'autoDescribeIntervalS': (10, 600),
    'speechRate': (0.5, 2.0),
    'speechVolume': (0.0, 1.0),
    'announceCooldownS': (0, 600),
    'minTrackFrames': (1, 30),
    'leaveAfterS': (0.5, 30),
  };

  Map<String, Object?> toJson() => {
        'inferenceSize': inferenceSize,
        'detectionFps': detectionFps,
        'confidenceThreshold': confidenceThreshold,
        'confusableThreshold': confusableThreshold,
        'strongConfidence': strongConfidence,
        'vlmEnabled': vlmEnabled,
        'vlmImageSize': vlmImageSize,
        'vlmMaxTokens': vlmMaxTokens,
        'autoDescribe': autoDescribe,
        'autoDescribeIntervalS': autoDescribeIntervalS,
        'speechEnabled': speechEnabled,
        'speechRate': speechRate,
        'speechVolume': speechVolume,
        'voiceId': voiceId,
        'announcementsMuted': announcementsMuted,
        'announceNewObjects': announceNewObjects,
        'announceDepartures': announceDepartures,
        'announceSceneChanges': announceSceneChanges,
        'obstacleAlerts': obstacleAlerts,
        'announceCooldownS': announceCooldownS,
        'minTrackFrames': minTrackFrames,
        'leaveAfterS': leaveAfterS,
        'verbosity': verbosity.name,
        'hapticsEnabled': hapticsEnabled,
      };

  /// Unknown keys are ignored and out-of-range or mistyped values fall back to
  /// the default, so a corrupt or older settings file never stops the app.
  factory AuraSettings.fromJson(Map<String, Object?> j) {
    const d = AuraSettings();
    T pick<T>(String k, T def) {
      final v = j[k];
      if (v is num && def is double) {
        final r = _ranges[k];
        final x = v.toDouble();
        return (r == null || (x >= r.$1 && x <= r.$2) ? x : def) as T;
      }
      if (v is int && def is int) {
        final r = _ranges[k];
        return (r == null || (v >= r.$1 && v <= r.$2) ? v : def) as T;
      }
      return v is T ? v : def;
    }

    return AuraSettings(
      inferenceSize: pick('inferenceSize', d.inferenceSize),
      detectionFps: pick('detectionFps', d.detectionFps),
      confidenceThreshold: pick('confidenceThreshold', d.confidenceThreshold),
      confusableThreshold: pick('confusableThreshold', d.confusableThreshold),
      strongConfidence: pick('strongConfidence', d.strongConfidence),
      vlmEnabled: pick('vlmEnabled', d.vlmEnabled),
      vlmImageSize: pick('vlmImageSize', d.vlmImageSize),
      vlmMaxTokens: pick('vlmMaxTokens', d.vlmMaxTokens),
      autoDescribe: pick('autoDescribe', d.autoDescribe),
      autoDescribeIntervalS: pick('autoDescribeIntervalS', d.autoDescribeIntervalS),
      speechEnabled: pick('speechEnabled', d.speechEnabled),
      speechRate: pick('speechRate', d.speechRate),
      speechVolume: pick('speechVolume', d.speechVolume),
      voiceId: j['voiceId'] is String ? j['voiceId'] as String : null,
      announcementsMuted: pick('announcementsMuted', d.announcementsMuted),
      announceNewObjects: pick('announceNewObjects', d.announceNewObjects),
      announceDepartures: pick('announceDepartures', d.announceDepartures),
      announceSceneChanges: pick('announceSceneChanges', d.announceSceneChanges),
      obstacleAlerts: pick('obstacleAlerts', d.obstacleAlerts),
      announceCooldownS: pick('announceCooldownS', d.announceCooldownS),
      minTrackFrames: pick('minTrackFrames', d.minTrackFrames),
      leaveAfterS: pick('leaveAfterS', d.leaveAfterS),
      verbosity: Verbosity.values.asNameMap()[j['verbosity']] ?? d.verbosity,
      hapticsEnabled: pick('hapticsEnabled', d.hapticsEnabled),
    );
  }

  /// Returns a copy with [patch] applied (same validation as [fromJson]).
  AuraSettings merge(Map<String, Object?> patch) => AuraSettings.fromJson({...toJson(), ...patch});
}
