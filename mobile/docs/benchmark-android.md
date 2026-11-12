# Android detector benchmark: iQOO Z7s (Snapdragon 695)

**Status: not yet measured on the phone.** No numbers below come from a physical device yet. Fill this in from the report the app produces (Settings → Models → Detector benchmark → Copy report). The app saves the same report to its documents folder and prints it to the system log with the prefix `AuraBench:`.

## How to run it

1. Install the latest debug APK (see "Testing on your phone" in `mobile/README.md`).
2. Open AuraSight, then **Settings → Models**, and wait for **Object detector ready**. Download it first if asked; it's 10.2 MB.
3. Unplug the charger, so the battery readings mean something.
4. On the **Live** tab, start the camera and point it at an ordinary room. Then go to **Settings → Models → Detector benchmark**, choose **5 min** and press **Run benchmark**. Keep the phone still and the screen on.
5. Press **Copy report** and paste it below. Run it twice: once as is (GPU), and once after a cold start for the load time.

Optional, with adb on a computer:

```
adb logcat -s flutter | findstr AuraBench        # Windows
adb logcat -s flutter | grep AuraBench           # macOS / Linux
adb shell dumpsys battery                          # temperature in tenths of a degree C
adb shell dumpsys thermalservice | findstr Status
```

## What the report records

| Measure | How it is measured |
|---|---|
| Model load time | From opening the file to the native self-check passing, which includes GPU delegate compilation |
| Latency: average, median, p95, min, max | 100 back-to-back detections after 5 warm-up runs, split into preprocessing (YUV→RGB, rotation, letterbox), model, and decoding with NMS |
| Detection FPS | Sustained run paced at the configured rate (default 6 per second) for 30 s, 1 min or 5 min |
| Delegate | GPU (LiteRT GPU delegate) or CPU (XNNPACK), with the reason if the GPU was not used |
| Memory | App PSS and free system memory, before and after |
| Temperature | Battery temperature before and after (the closest reading Android gives apps), plus the thermal status |
| Battery | Level and instantaneous current before and after; only meaningful when unplugged and over several minutes |

## Results (to be filled in from the phone)

| | GPU run | CPU run |
|---|---|---|
| Delegate reported | | |
| Model load time (ms) | | |
| Latency median / p95 (ms) | | |
| Model-only median (ms) | | |
| Preprocessing median (ms) | | |
| Sustained detections per second | | |
| Battery temperature before → after (°C) | | |
| Thermal status before → after | | |
| Battery level before → after (%) | | |
| App memory (MB) | | |

## Reference: conversion parity (CI, not the phone)

`mobile/docs/detector-parity-tflite.md`: the exported TFLite model matched all 21 desktop detections scoring 0.4 or higher on the 16 fixture photos (mean IoU 0.964). The same comparison runs through the app's Kotlin code on an Android emulator in CI (job "Android emulator integration test"). Emulator timings are not phone timings.
