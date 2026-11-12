# aura_native

Native modules for the AuraSight app: Android (Kotlin) and iOS (Swift).

| Channel | Android | iOS |
|---|---|---|
| `aurasight/tts` | TextToSpeech, offline voices only | AVSpeechSynthesizer |
| `aurasight/system` | Thermal status, memory info | Thermal state, available memory |

The Dart side of each channel is defined in `mobile/app/lib/platform/channels.dart`.
