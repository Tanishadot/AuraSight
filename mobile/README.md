# AuraSight Mobile (Android and iOS)

AuraSight for phones: live object detection, spatial awareness, spoken alerts and scene descriptions, all running on the phone. The internet is only used to download models you choose to install.

The desktop app in the repository root is unchanged. This folder is a separate Flutter app that reuses the desktop's proven logic, ported to Dart and checked against it.

## Status

A feature is marked **verified** only after it has run on a physical phone for that platform.

| Area | Android | iOS | Notes |
|---|---|---|---|
| Shared logic (tracking, verification, spatial, events, announcements, speech queue, find) | Tested in CI | Tested in CI | Matches the desktop's outputs on every golden scenario |
| App shell, six screens, settings, activity log, model downloads | Builds; unit and widget tests pass | Builds for the simulator | Not yet run on a phone |
| Camera | Code in place (`camera` plugin) | Code in place | Not yet run on a phone |
| Object detection (YOLO11n) | Not built yet | Not built yet | LiteRT on Android, Core ML on iOS |
| Scene model (SmolVLM via llama.cpp) | Not built yet | Not built yet | Downloadable from Settings, but there is no runtime yet |
| Speech (system TTS, offline voices) | Code in place (TextToSpeech) | Code in place (AVSpeechSynthesizer) | Not yet run on a phone |
| Thermal and memory monitoring | Code in place | Code in place | Feeds the adaptive detection rate |
| Voice input | Not started | Not started | Optional; sherpa-onnx or whisper.cpp |
| Benchmarks | Not started | Not started | Target phone: iQOO Z7s (Snapdragon 695) |

Until a native module exists, the app shows a clear message for it, such as "This build doesn't include the on-device detector yet". It never shows made-up results.

## Layout

```
mobile/
  app/                 Flutter app (Dart UI, Riverpod state, services)
    lib/platform/      channel contracts shared by the Kotlin and Swift code
    lib/services/      camera, detection, tracking, scene model, speech, audio, models
    lib/screens/       Live, Describe, Ask, Find, Activity, Settings
  packages/
    aura_core/         pure-Dart logic ported from the desktop backend
    aura_native/       native inference plugin (Kotlin + Swift)
  tools/gen_golden.py  generates parity test data from the desktop Python
  docs/                feasibility audit and, later, benchmark and offline reports
```

## How the logic stays in sync with the desktop

`tools/gen_golden.py` runs the desktop Python implementations on fixed scenarios and saves their outputs in `packages/aura_core/test/golden/`. The Dart tests must reproduce them exactly. To regenerate after changing the desktop logic, run this from the repository root:

```
.venv/Scripts/python mobile/tools/gen_golden.py
```

## Building

Everything is built in GitHub Actions (`.github/workflows/mobile.yml`); nothing has to be installed locally:

| Job | What it does |
|---|---|
| `aura_core` | Analyses and tests the shared Dart logic |
| `app` | Runs `flutter analyze` and `flutter test`, and builds a debug APK (artifact `aurasight-debug-apk`) |
| `ios` | Builds for the iOS simulator, unsigned |

Building locally needs Flutter (stable), JDK 17 and, for iOS, a Mac with Xcode:

```
cd mobile/app
flutter pub get
flutter test
flutter run            # with a phone connected
```

## Installing on an Android phone

1. On the phone, turn on **Developer options** (tap *Build number* seven times in *About phone*), then turn on **USB debugging**. On iQOO/Funtouch, also turn on **Install via USB**.
2. Download the `aurasight-debug-apk` artifact from the latest successful **mobile** run on GitHub Actions, and unzip it.
3. Install it with either method:
   - **With adb** (Android platform-tools): `adb install -r app-debug.apk`
   - **Without a computer**: copy `app-debug.apk` to the phone, open it, and allow installing from that app when asked.

Debug builds are unsigned for release and meant for testing only.

## Privacy

- No network calls except model downloads from the URLs listed in `lib/services/model_manager.dart`.
- Every download is verified against a pinned SHA-256 before use.
- Settings and the activity log are stored in a local SQLite database.

## Licences

- The YOLO11n weights are AGPL-3.0 (Ultralytics). The app's source must stay public under AGPL-compatible terms. App Store distribution is a known conflict; see `docs/feasibility-audit.md`.
- SmolVLM is Apache-2.0, llama.cpp is MIT, and Flutter is BSD-3.
