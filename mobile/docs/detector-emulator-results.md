# Detector integration test on an Android emulator (CI)

This test runs the app's real code (Dart, Kotlin and LiteRT 1.4.0) inside an Android 13 x86_64 emulator on GitHub Actions: CI run for commit `eb51827`, job "Android emulator integration test (detector)". It is **not** the iQOO Z7s. The emulator has no usable GPU delegate, so everything ran on the CPU, and its timings say nothing about the phone.

| Test | Result |
|---|---|
| Download the published model from the GitHub release and verify its SHA-256 | Passed |
| Load with LiteRT and pass the self-test, with networking disabled for the model manager | Passed: CPU (fallback reason "GPU delegate not supported on this device"), load 124 ms, self-test person 0.84 |
| Detections match the desktop detector on 16 fixture photos | Passed: 21 of 21 desktop detections scoring 0.40 or higher matched; mean IoU 0.954; mean score difference −0.022 |
| A rotated camera YUV frame gives the same detections as the photo | Passed (preprocessing 43.7 ms, model 26.2 ms on the emulator) |
| A damaged model file is refused before loading and deleted | Passed |

## Expected differences from the desktop

- **Letterbox shape:** the desktop pads to a multiple of 32 (for example 320×256 for a 4:3 photo), while the mobile model has a fixed square 320×320 input. Some objects therefore see slightly different context.
- **Score changes in crowded scenes:** the largest score change was 0.27, in `tv_control_room_monitors.jpg`, a cluttered control room with many small monitors and chairs. A chair scored 0.74 on the desktop and 0.47 on the phone path, yet was still matched by position. Simple scenes changed by about 0.02.
- **Resampling:** Kotlin bilinear resampling versus OpenCV `INTER_LINEAR`, and JPEG decoding by Android versus OpenCV, cause sub-pixel differences.
- **Runtime:** TFLite kernels versus ONNX Runtime. On the phone the GPU delegate computes in FP16, which may shift scores slightly more; that has not been measured yet.

The desktop-only detections below 0.40 (for example `cup 0.25`) are not part of the comparison. They fall under the confidence the app needs before announcing anything.
