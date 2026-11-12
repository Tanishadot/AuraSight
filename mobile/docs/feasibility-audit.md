# AuraSight Mobile: Phase 1 feasibility audit

Status: audit only. No mobile code has been built or run yet, so nothing below is claimed to work on a phone.
Date: 12 November 2026. Project owner: Tanisha Jogi.

## Summary

The product plan is technically feasible. Flutter, LiteRT, Core ML, platform text-to-speech and llama.cpp's multimodal library (mtmd) all have a credible on-device path. On a phone, though, the Python desktop code has to be ported to Dart, not reused.

Four blockers stop development from starting:

| # | Blocker | Severity | What clears it |
|---|---|---|---|
| B1 | Only 3.7 GB of disk is free. The Flutter, Android SDK, NDK and Gradle toolchain needs roughly 15 GB or more. | Blocks all local builds | Free at least 20 GB, or build in CI (see B2) |
| B2 | There is no Mac. iOS apps can only be compiled, signed and run with Xcode on macOS. | Blocks all iOS work and testing | A Mac with Xcode, or GitHub Actions macOS runners for builds plus a physical iPhone for testing |
| B3 | There is no Android phone attached. The emulator can't give realistic camera, NPU, GPU, thermal or battery numbers, and BIOS virtualisation reports as off. | Blocks Android testing and benchmarks | A physical Android phone with USB debugging enabled |
| B4 | YOLO11 weights are AGPL-3.0. App Store distribution and AGPL are widely considered incompatible, and an AGPL app must publish its full source. | Blocks store release, not development | Keep the app open source and off the App Store, buy an Ultralytics licence, or switch to an Apache-licensed detector |

A minor fix: the installed JDK is 11, and current Android Gradle Plugin builds need JDK 17.

## 1. Existing desktop repository

The desktop app has a FastAPI backend, YOLO11n through ONNX Runtime, a NumPy ByteTrack, SmolVLM-500M through llama-server, SAPI speech and a React dashboard. 137 tests pass. It stays as it is; the mobile app goes in `mobile/` and has no runtime dependency on FastAPI.

What carries over to mobile:

| Desktop component | Mobile plan | Reuse |
|---|---|---|
| `tracking/bytetrack.py` | Port to Dart (pure math, about 300 lines) | Algorithm and test cases |
| `detection/verify.py` (false-positive verifier) | Port to Dart | Thresholds, class lists, test cases |
| `spatial/positioning.py` | Port to Dart | Rules and test cases |
| `assistant/announcer.py`, `scene_monitor.py` | Port into `AudioOrchestrator` | Cooldowns, priorities, de-duplication |
| `speech/service.py` priority queue | Port to Dart over native TTS | Queue rules (pre-emption, re-reading interrupted sentences, cancelling a description) |
| `vision/analyzer.py` prompts and text clean-up | Port to Dart | Prompts, generation settings, clean-up regexes |
| `assistant/qa.py` | Port to Dart | Intents, grounding rules |
| ONNX and llama-server runtimes | Replaced by LiteRT, Core ML and an embedded llama.cpp | None (runtime layer only) |

To check that the Dart ports behave the same, a script will run the Python implementations on fixed inputs (the 16 scene fixtures, synthetic tracks) and save golden JSON files. The Dart tests must then reproduce those outputs exactly. That keeps one source of truth for behaviour instead of two drifting copies.

## 2. Object detection: YOLO11n

| Item | Android (LiteRT) | iOS (Core ML) |
|---|---|---|
| Export path | Ultralytics `export format=tflite` (goes through onnx2tf and TensorFlow) | Ultralytics `export format=coreml` (coremltools) |
| Where export can run | Any OS, but TensorFlow needs about 2 GB of disk, so it can't run here (B1). It can run in a Linux CI job. | coremltools has Linux wheels for conversion. Compiling to `.mlmodelc` happens in the Xcode build. |
| Precision | FP16 for the GPU delegate, INT8 for CPU and NPU (needs calibration images) | FP16 by default; the Neural Engine prefers FP16 |
| Acceleration | GPU delegate on most devices. NPU only through vendor-specific delegates (Qualcomm QNN, MediaTek NeuroPilot), so support varies by phone. | Neural Engine on A12 and later, with GPU and CPU fallback set by `computeUnits` |
| Known risks | Some exported ops may fall back to CPU on the GPU delegate. Op coverage has to be checked per export. | Built-in NMS is only available through `nms=True` export; otherwise NMS runs in Dart |

Decision rule (still to be measured): keep YOLO11n if it beats 15 fps at 320 px on a mid-range phone at FP16. Otherwise compare 256 px input and INT8. The false-positive verifier ports unchanged, but its crop re-detection doubles inference cost, so on mobile it only runs on newly appeared, easily confused classes.

If a non-AGPL detector is needed (B4), the candidates are YOLOX-Nano (Apache-2.0) and EfficientDet-Lite0 (Apache-2.0, ships as a ready-made LiteRT model). Both are less accurate than YOLO11n; that would be measured, not assumed.

## 3. Vision-language model: SmolVLM-500M

Desktop files: `SmolVLM-500M-Instruct-Q8_0.gguf` (417 MB) and `mmproj-SmolVLM-500M-Instruct-Q8_0.gguf` (104 MB), 521 MB in total.

- **The files themselves are portable.** GGUF is architecture-independent, so the same weights load on ARM. The open question is the runtime build, not the weights.
- **Runtime:** llama.cpp's `libmtmd` is the multimodal layer and lists SmolVLM as supported. The iOS XCFramework includes mtmd. Android builds need `-DLLAMA_BUILD_MTMD=ON`. Upstream describes mtmd as under heavy development with breaking changes expected, so the llama.cpp version must be pinned.
- **Flutter integration options**, still to be evaluated before choosing:
  - Existing plugins that claim mtmd vision on Android and iOS: `llama_cpp_dart`, `llamadart`, `llamafu`, `mt_llmkit`.
  - A thin plugin of our own over a pinned llama.cpp, with the same API on both platforms.

  Selection criteria: the pinned llama.cpp version, whether the build is reproducible, Metal support on iOS, licence, and whether it is maintained.
- **Performance risk (the biggest unknown):** on the desktop i5-8350U, encoding one 512 px image takes about 10 s on the CPU and generation runs at about 23 tokens/s.
  - Android: phones are likely to be similar or slower on CPU, and llama.cpp's Android GPU path (Vulkan or OpenCL) varies by device.
  - iOS: Metal acceleration in llama.cpp is mature, so iOS is likely to be much faster.

  Both must be measured. The design already fits a slow model: it runs only on request or on selected frames, streams sentences as they finish, and runs off the UI thread.
- **Memory:** about 0.55 GB of weights plus context and image buffers, roughly 0.8 to 1 GB in use. It should work on 4 GB phones, but only if the model is unloaded under memory pressure, which is planned.
- **Fallback:** if a phone is too slow or short of memory, the Scene screen falls back to detector-only descriptions and says so, as the desktop app does today. There will be no fake or mock descriptions.

## 4. Voice

| Function | Android | iOS | Offline check |
|---|---|---|---|
| Text-to-speech | `android.speech.tts.TextToSpeech` | `AVSpeechSynthesizer` | Android: pick a `Voice` with `isNetworkConnectionRequired()==false` and the language installed. iOS: system voices run on-device. |
| Interrupt and stop | `stop()` | `stopSpeaking(at: .immediate)` | none needed |
| Voice input (optional) | sherpa-onnx (Apache-2.0, Flutter package available) or whisper.cpp (MIT) | same | Model downloaded once and run on-device |

The TTS bridge will be a small platform channel of our own, with `speak`, `stop`, `voices` and completion events, because the priority queue needs exact completion and interruption callbacks.

## 5. Licences

| Component | Licence | Issue |
|---|---|---|
| Flutter, Dart, Riverpod | BSD-3 / MIT | none |
| LiteRT (TensorFlow Lite) | Apache-2.0 | none |
| Core ML / coremltools | Apple SDK / BSD-3 | none |
| llama.cpp / ggml | MIT | none |
| SmolVLM-500M-Instruct | Apache-2.0 | none (attribution required) |
| sherpa-onnx / whisper.cpp | Apache-2.0 / MIT | none; the speech-model licences need checking separately |
| YOLO11n weights and Ultralytics export | AGPL-3.0 | **B4.** The app's whole source must stay AGPL-compatible and public, and App Store distribution is a known conflict. |
| sqflite / drift (SQLite) | BSD / MIT | none |

## 6. Hardware compatibility (targets still to be validated)

| Tier | Android | iOS | Expected mode |
|---|---|---|---|
| Minimum | Android 9+, arm64, 4 GB RAM | iOS 16+, A12 or later, 3 GB RAM | Detection on CPU or GPU; VLM on request only, may be slow |
| Recommended | Snapdragon 8 Gen 1 or equivalent, 8 GB | A15 or later | Detection on GPU or ANE at 15 fps or more; VLM in seconds |

Not supported: 32-bit Android, because llama.cpp performance and memory are impractical.

## 7. Development machine

| Need | Available | Gap |
|---|---|---|
| Flutter SDK | not installed | install (about 3 GB) |
| Android SDK, NDK r27, build tools | not installed | install (about 6 to 8 GB) |
| JDK 17 | JDK 11 | install JDK 17 |
| Gradle and pub caches | none | 2 to 4 GB |
| Free disk | 3.7 GB | **B1** |
| RAM | 7.8 GB | Enough for builds; tight when running an emulator at the same time |
| Mac with Xcode | none | **B2** |
| Test phones | none attached | **B3** |

Proposed CI on the public GitHub repository, so builds don't depend on this machine's disk:

- Linux runner: `flutter analyze`, `flutter test`, the Dart parity tests, the YOLO11n TFLite export and an Android APK build.
- macOS runner: the Core ML export, an iOS simulator build and `flutter test` on iOS.

Free for public repositories. CI builds prove the code compiles; they don't prove camera, accelerator or speech behaviour. That still needs real phones.

## 8. Plan once the blockers are cleared

1. Toolchain, CI and an empty Flutter app with Riverpod and the six screens, with navigation and accessibility checked by TalkBack and VoiceOver tests on device.
2. Dart ports of the tracker, verifier, spatial rules and audio orchestrator, plus golden parity tests against Python. These can start as soon as the Dart SDK is available, in CI or locally.
3. Camera on both platforms, then detection (export, a benchmark of LiteRT CPU, GPU and NNAPI against Core ML CPU, GPU and ANE, then the choice).
4. VLM plugin evaluation, a benchmark on each phone, then integration.
5. Speech, Ask Aura and Find Something; then accessibility refinement, benchmarks, installation builds and reports.

Each feature is marked as verified only after it has been run on a physical phone for that platform, with the device model recorded.

## Sources

- llama.cpp multimodal (mtmd) README: https://github.com/ggml-org/llama.cpp/tree/master/tools/mtmd
- Liquid AI, llama.cpp multimodal on-device (Android `-DLLAMA_BUILD_MTMD=ON`; mtmd in the iOS XCFramework): https://docs.liquid.ai/deployment/on-device/llama-cpp/multimodal
- llama_cpp_dart: https://pub.dev/documentation/llama_cpp_dart/0.9.0-dev.6/
- llamadart: https://github.com/leehack/llamadart
- llamafu: https://pub.dev/documentation/llamafu/latest/
- mt_llmkit: https://pub.dev/packages/mt_llmkit
- Flutter Windows and Android setup (disk requirements): https://docs.flutter.dev/get-started/install/windows/mobile
