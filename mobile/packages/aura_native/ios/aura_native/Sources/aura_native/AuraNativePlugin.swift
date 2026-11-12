import AVFoundation
import Flutter
import UIKit
import os

/// Registers AuraSight's native channels (see app/lib/platform/channels.dart).
public class AuraNativePlugin: NSObject, FlutterPlugin {
  private let tts = TtsHandler()

  public static func register(with registrar: FlutterPluginRegistrar) {
    let instance = AuraNativePlugin()
    let speech = FlutterMethodChannel(name: "aurasight/tts", binaryMessenger: registrar.messenger())
    speech.setMethodCallHandler { call, result in instance.tts.handle(call, result: result) }
    let system = FlutterMethodChannel(name: "aurasight/system", binaryMessenger: registrar.messenger())
    system.setMethodCallHandler { call, result in SystemHandler.handle(call, result: result) }
  }
}

/// AVSpeechSynthesizer behind the `aurasight/tts` channel. System voices run
/// on the device, so speech works offline. `speak` completes with true when
/// the utterance finished and false when it was stopped.
final class TtsHandler: NSObject, AVSpeechSynthesizerDelegate {
  private let synth = AVSpeechSynthesizer()
  private var pending: [ObjectIdentifier: FlutterResult] = [:]

  override init() {
    super.init()
    synth.delegate = self
  }

  func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    switch call.method {
    case "init":
      configureAudioSession()
      result(["engine": "AVSpeechSynthesizer", "offlineVoices": AVSpeechSynthesisVoice.speechVoices().count])
    case "voices":
      result(AVSpeechSynthesisVoice.speechVoices().map {
        ["id": $0.identifier, "name": $0.name, "language": $0.language, "offline": true]
      })
    case "speak":
      guard let args = call.arguments as? [String: Any], let text = args["text"] as? String else {
        result(FlutterError(code: "bad_args", message: "text is required", details: nil))
        return
      }
      let u = AVSpeechUtterance(string: text)
      // AuraSight's rate is a multiplier where 1.0 is normal speech; iOS's normal rate is 0.5.
      let multiplier = (args["rate"] as? Double) ?? 1.0
      let rate = Float(multiplier) * AVSpeechUtteranceDefaultSpeechRate
      u.rate = min(max(rate, AVSpeechUtteranceMinimumSpeechRate), AVSpeechUtteranceMaximumSpeechRate)
      u.volume = Float((args["volume"] as? Double) ?? 1.0)
      if let id = args["voiceId"] as? String, let voice = AVSpeechSynthesisVoice(identifier: id) {
        u.voice = voice
      } else {
        u.voice = AVSpeechSynthesisVoice(language: AVSpeechSynthesisVoice.currentLanguageCode())
      }
      if synth.isSpeaking { synth.stopSpeaking(at: .immediate) }
      pending[ObjectIdentifier(u)] = result
      synth.speak(u)
    case "stop":
      synth.stopSpeaking(at: .immediate)
      result(nil)
    default:
      result(FlutterMethodNotImplemented)
    }
  }

  func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
    pending.removeValue(forKey: ObjectIdentifier(utterance))?(true)
  }

  func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
    pending.removeValue(forKey: ObjectIdentifier(utterance))?(false)
  }

  /// Speak even with the ringer switch on silent, and lower other audio while talking.
  private func configureAudioSession() {
    let session = AVAudioSession.sharedInstance()
    try? session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
    try? session.setActive(true)
  }
}

/// Thermal and memory state for the adaptive detection rate (`aurasight/system`).
enum SystemHandler {
  static func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard call.method == "snapshot" else {
      result(FlutterMethodNotImplemented)
      return
    }
    let thermal: String
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: thermal = "nominal"
    case .fair: thermal = "fair"
    case .serious: thermal = "serious"
    case .critical: thermal = "critical"
    @unknown default: thermal = "nominal"
    }
    let available = os_proc_available_memory()
    result([
      "thermal": thermal,
      "availMemMb": Int(available / (1024 * 1024)),
      "lowMemory": available < 200 * 1024 * 1024,
    ])
  }
}
