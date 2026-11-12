# Android detector crash on start-up (iQOO Z7s, Snapdragon 695)

**Report:** after "Loading the object detector… running a self-test" appears, the whole app closes. Every later launch does the same.

**Status:** fixes are in this branch. **Nobody has checked them on the phone yet**, and **no log from the phone has been collected yet**. The cause below comes from reading the code and the CI evidence, not from a stack trace. Section 4 explains how to get the trace that confirms or rules it out.

## 1. What the code and CI show

| Finding | Evidence |
|---|---|
| The model is fine | The published `yolo11n_320_float32.tflite` (SHA-256 `2de8b9aa…4d70`, as in the manifest) has input `[1,320,320,3]` float32 and output `[1,84,2100]` float32. It is not quantised and has no tensors with more than 4 dimensions. It uses only standard ops (CONV_2D, MUL, LOGISTIC, CONCATENATION, RESHAPE, TRANSPOSE, STRIDED_SLICE, SOFTMAX, RESIZE_NEAREST_NEIGHBOR, …). LiteRT's CPU interpreter loads and runs it. The emulator parity test matched all 21 reference detections. |
| The self-test image is small | 640×510 JPEG. Decoding it takes about 1.3 MB on the Java heap, so it cannot cause an OutOfMemoryError. |
| **The GPU path had never run anywhere** | The old `YoloDetector.load` used `backend = "auto"`, which tries the LiteRT GPU delegate first. The only on-device test was CI's x86_64 emulator, which logged "GPU delegate not supported on this device" and ran on the CPU. On the Snapdragon 695, `CompatibilityList` reports the Adreno 619 as supported, and the plugin declares `libOpenCL.so`. So the phone was the first device ever to create the OpenCL GPU delegate and run the warm-up inference on it. That warm-up runs inside `load`, while the card says "running a self-test". |
| **A native crash cannot be caught** | Every detector call was wrapped in `catch (e: Throwable)`. That only catches Java exceptions. If the GPU delegate or the vendor OpenCL driver crashes with SIGSEGV or SIGABRT, the process dies immediately. That matches "the entire app closes" with no error on screen. |
| **The crash repeated on every launch** | `HomeShell.initState` called `DetectorController.init()`, and that loaded the model and ran the self-test automatically whenever the file was on the phone. A crashing load therefore came back on every start. That is the "repeated self-test". |
| Other defects found (fixed) | Verification crops ran outside the one-request-in-flight gate (`unawaited(_verify)`), so the native queue could grow, with each queued request holding a whole camera frame. `CompatibilityList` was never closed. During a reload, Dart still reported the old model as available, so frames could race the load. Failed live detections went only to the activity log and were retried forever. |

**Most likely crash point:** creating the GPU delegate or running the first GPU inference (journal steps `gpu_delegate_create`, `gpu_interpreter_create` or `gpu_warmup`). This is the hypothesis that fits all the evidence. A tombstone from the phone is needed to confirm it (section 4). These were ruled out by inspection or test: the model file, tensor shapes and types, input preprocessing, output decoding (JVM unit tests), Java OutOfMemoryError on the test image, and camera concurrency (the crash happens before the camera is used).

## 2. What changed

**Native (Kotlin, `aura_native`)**
- **CPU (XNNPACK) is the default backend.** The GPU is only tried from *Settings → Detector diagnostics → Try the GPU*.
- **Crash journal (`DetectorGuard`).** Before each native step (`map_model`, `gpu_compat_check`, `gpu_delegate_create`, `gpu_interpreter_create`, `gpu_warmup`, `cpu_interpreter_create`, `cpu_warmup`, `first_inference`), the step's name is written to a file. It is removed when the step returns. If the file is still there at the next start, the app knows exactly which step killed it and says so. If that step was on the GPU, the GPU is switched off for the detector until the user allows it again. This doesn't hide the crash: the crash is reported, the step that caused it isn't repeated, and the cause is recorded.
- **Crash report without adb (`CrashReports`).** Android's `ApplicationExitInfo` gives the reason the last processes ended (native crash, Java crash, low-memory kill, ANR). For a native crash on Android 12+ it also gives the full tombstone, with signal, fault address and backtrace. The tombstone is saved to `files/crash/` and shown under *Detector diagnostics → Crash report*, with a *Copy* button. Uncaught Java exceptions on any thread are also written there.
- **One request at a time.** A single native thread serves all requests. Inference requests that arrive while one is pending are refused with `busy` instead of being queued.
- **No repeated loading.** A `load` with the same file, settings and backend reuses the loaded interpreter.
- **Errors are reported, not fatal.** An `OutOfMemoryError` comes back as error `oom` and the app keeps running. Every reply includes native-heap and Java-heap readings. `CompatibilityList` is closed. The interpreter is closed before its delegate.

**Dart (app)**
- **No automatic load** unless the previous load and self-test completed *and* the previous process didn't die in a detector step. After a crash, the Live screen shows what happened and a **Retry** button (which uses the CPU).
- **Downloading no longer loads the model.** The card shows *Load detector*.
- **One load at a time.** Concurrent `load` and `retry` calls share one native load, and the self-test runs once per load.
- **Live detection:** verification crops run inside the single in-flight slot, at most one per frame. Skipped checks are released without being counted as misses (`ObjectVerifier.cancel`). After 3 errors in a row, detection stops and shows *Object detection stopped* with **Retry**.
- **Conservative first run:** at most 2 detections per second, whatever the setting says (`LivePipeline.conservativeFps`). Camera frames are at `ResolutionPreset.low` (about 320×240; the model input is 320×320 anyway), which cuts the per-frame copy from about 520 KB to about 115 KB. The preview is softer as a result.
- **Staged hardware check** (*Settings → Detector diagnostics*): the six stages below, in order. Each stage records latency and memory, and logs a line starting `AuraDiag:`.

## 3. Testing on the phone, in stages

Install the debug APK. Before installing it, collect the old crash if you can (section 4). Then go through the stages in order, and stop at the first one that fails or closes the app.

1. **Launch.** Open AuraSight. It must stay open. The model is *not* loaded automatically, and the Live tab shows either "Object detector downloaded – Load detector" or the crash message from the old build's run. Open *Settings → Detector diagnostics* and run **stage 1**.
2. **Stage 2: load the model on the CPU, camera off.** Expect "CPU, load … ms, warm-up … ms".
3. **Stage 3: one inference on the bundled photo.** Expect "person 0.8x".
4. **Stage 4: one camera frame.** Allow camera access when asked.
5. **Stage 5: continuous, 1 per second for 20 s.** Expect about 20 inferences and 0 errors.
6. **Stage 6: full start-up with self-test.** Expect "ready on CPU".
7. Press **Copy hardware check report** and send it. It contains latency and memory (app PSS and free RAM before and after each stage, plus native and Java heap).
8. Only then use the Live tab normally. The GPU is optional: **Try the GPU (may close the app)**. If it closes the app, reopen it. The detector card will say which GPU step crashed, the GPU will stay off, and the crash report will contain the tombstone. Send that report.

If the app closes at any stage, reopen it, go to *Settings → Detector diagnostics → Copy crash report*, and send that, together with the adb output below if you have a computer.

## 4. Getting the log with adb

Install the platform tools (`adb`) on a computer. On the phone, enable *Developer options → USB debugging* (iQOO: *Settings → About phone → Software version*, tap 7 times), connect it by USB and accept the prompt. Then:

```sh
adb devices                                   # the phone must be listed as "device"

# 1. Before reinstalling: what killed the old build (reason + description)
adb shell dumpsys activity exit-info com.tanishadot.aurasight > exit-info.txt

# 2. Native crash tombstones (full backtrace of a SIGSEGV/SIGABRT)
adb bugreport bugreport.zip                    # contains FS/data/tomb_stones/; works without root

# 3. Reproduce with a clean, full log
adb logcat -c
adb logcat -v threadtime -b main,system,crash > aurasight-log.txt
#    ...now open the app and reproduce; then press Ctrl+C

# 4. Narrow views of the same log
adb logcat -d -v threadtime -s AndroidRuntime DEBUG libc tflite AuraDetector AuraCrash flutter
adb logcat -d | grep -E "AuraDetector|AuraDiag|Fatal signal|SIGSEGV|SIGABRT|backtrace|OutOfMemory|lowmemorykiller|am_crash"

# Memory at any moment while the app runs
adb shell dumpsys meminfo com.tanishadot.aurasight
```

On Windows, replace `grep -E "…"` with `findstr /R "AuraDetector AuraDiag Fatal SIGSEGV SIGABRT backtrace OutOfMemory"`.

The previously suggested `adb logcat -s flutter AndroidRuntime tflite` would **miss** a native crash. Native crashes are logged under the `DEBUG` and `libc` tags, and in the `crash` buffer.

**What to send:** `exit-info.txt`, `aurasight-log.txt` (or at least the lines from `Fatal signal` / `*** *** ***` through the end of the `backtrace:` block), and the *Copy crash report* / *Copy hardware check report* text from the app. From the backtrace, the deciding frames are the ones in `libtensorflowlite_gpu_jni.so`, `libOpenCL.so` / `libCB.so` / `libllvm-qcom.so` (GPU), `libtensorflowlite_jni.so` (CPU/XNNPACK), or `libflutter.so` / camera libraries (elsewhere). The last `AuraDetector: begin <step>` line before the crash names the step.

## 5. Results from the phone

*Not yet run. Fill this in from the hardware check report.*

| Stage | Result | Latency (median / p95) | App PSS before → after | Notes |
|---|---|---|---|---|
| 1 Launch | | – | | |
| 2 Load (CPU) | | | | |
| 3 Test photo | | | | |
| 4 One camera frame | | | | |
| 5 Continuous 20 s | | | | |
| 6 Full self-test | | | | |
| GPU attempt | | | | |
