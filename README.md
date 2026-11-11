# AuraSight

**An offline, private vision assistant for blind and low-vision people.**
AuraSight uses your computer's webcam to recognise and track objects, describe the scene in plain language, answer questions about what's in view, and speak it all aloud. Everything runs on your own machine. No cloud services, no API keys, no telemetry, and no recording.

> **Safety:** AuraSight is an assistive aid, **not** a mobility aid. A webcam cannot reliably detect every obstacle, step, drop-off or hazard, and it sees nothing outside its field of view. Keep using your white cane, guide dog or other mobility aid. Distances are rough estimates taken from the image, not measurements.

---

## What it does

| Feature | How |
|---|---|
| Live camera with start / pause / resume / stop, and camera selection | OpenCV capture thread with a single-frame buffer, plus a low-rate MJPEG preview to the local UI |
| Real-time object detection (80 COCO classes) | **YOLO11n** exported to ONNX, running on **ONNX Runtime** (CPU) |
| Persistent tracking IDs; enter and leave events | **ByteTrack** (NumPy implementation of the two-stage association, with per-class matching) |
| Approximate positions | left / slightly left / ahead / slightly right / right, plus near / middle / far from box geometry and per-class size priors |
| Scene descriptions and Q&A | **SmolVLM-500M-Instruct** (GGUF) on **llama.cpp `llama-server`**, or any **Ollama** vision model |
| Offline speech | **Windows SAPI 5** (built in), with **pyttsx3** as a fallback and **Piper** as an option. Speech never overlaps, is ordered by priority, and can be interrupted. |
| Alerts | new objects, approaching objects, near objects in your path, big scene changes, and (optionally) departures |
| Accessible UI | React with full keyboard control, screen-reader live regions, large targets, light and dark themes |

### Design decisions

- **The detector runs continuously; the VLM runs only on demand.** YOLO11n takes about 30–46 ms per frame on a 2018 laptop CPU. A VLM takes seconds, so it runs only when you press **Describe**, ask a question, or turn on the slow automatic mode. At most one VLM request runs at a time.
- **Answers come from the detector first.** "Where is my water bottle?" is answered from tracked detections, with a position. If the detector doesn't see the object, AuraSight says so. It may then ask the VLM for a second opinion, which it labels as *less reliable*. Small VLMs hallucinate, so they never get the final word on whether something is present. "What is on my left/right?" is also answered from measured box positions, because SmolVLM often mixes up left and right.
- **Objects must be confirmed before they're announced.** YOLO11n confuses some architecture with appliance classes. A bright window in a dark room looks like a lit screen (`tv` at 0.59), a glass panel scores `tv` at 0.52, and a plank door scores `refrigerator` at 0.63. A single threshold can't separate these from genuine but difficult TVs, so [backend/detection/verify.py](backend/detection/verify.py) combines three checks:
  1. **Temporal stability.** The object must be tracked for at least 4 frames (about 0.8 s) and be present in most of them.
  2. **Class-aware confidence.** The *mean* recent score must reach 0.45, or 0.5 for confusable classes (`tv`, `refrigerator`, `oven`, `microwave`, `toaster`, `sink`, `toilet`).
  3. **Secondary verification.** A confusable object scoring below 0.7 is re-detected on two zoomed crops. Over jittered frames, real TVs re-detect at 0.45–0.9; windows and doors at 0.1–0.4.

  Confirmed confusable objects are demoted if their confidence sinks again. Unconfirmed detections still appear in the UI, dashed and labelled "possible TV?", but they're never announced, never alert, and never answer questions. The TV class itself isn't suppressed: the TVs in the test photos are confirmed.
- **Scene descriptions combine the VLM's narrative with detector facts.** SmolVLM supplies the environment and things YOLO can't see (windows, doors, walls, lighting). Its camera-frame phrasing is rewritten into your frame of reference: "on the left side of the image" becomes "on your left", and "in the background" becomes "further away". Identifications are hedged ("This looks like…"). The description then ends with what the detector actually confirmed, for example "The object detector confirms a person ahead, near; a laptop slightly right, near."
- **Choosing the VLM.** All candidates were evaluated against this target machine (8 GB RAM, no GPU):

  | Model | Size | Verdict |
  |---|---|---|
  | Llama 3.2 Vision 11B | 7.9 GB, about 9 GB RAM | Rejected: doesn't fit in 8 GB RAM, and CPU latency would be minutes |
  | Moondream 2 (Ollama) | 1.7 GB, about 2.6 GB RAM | Supported through the Ollama backend; good captions |
  | **SmolVLM-500M Q8** (llama.cpp) | 546 MB, about 1.1 GB RAM (estimate) | **Default**: smallest that still gives useful descriptions; no extra runtime to install |
  | SmolVLM-256M Q8 | 280 MB | `--small` option for machines with 4 GB of RAM |

- **TTS choice.** SAPI 5 ships with Windows, so it needs no downloads, starts instantly, and supports real asynchronous speech with immediate purge (measured interrupt latency is about 70 ms). Piper sounds more natural but needs an extra voice model and a player. It is supported when you set `AURASIGHT_PIPER_EXE` and `AURASIGHT_PIPER_VOICE`.

## Architecture

```
 Browser UI (React, served locally)            Python backend (FastAPI, 127.0.0.1:8000)
 ┌──────────────────────────┐   REST    ┌───────────────────────────────────────────────┐
 │ Camera preview + boxes   │◀─────────▶│ /api/*  settings, camera, describe, ask, speech│
 │ Scene / Ask / Objects    │  WebSocket│ /ws     detections, activity, status, speech   │
 │ Activity / Voice / Status│◀──────────│ /api/stream.mjpg  low-rate preview (local only)│
 └──────────────────────────┘           │                                               │
                                        │ camera thread ─▶ latest frame (1 slot)        │
                                        │ detection thread @ N fps (frame skipping):    │
                                        │   YOLO11n ONNX ─▶ ByteTrack ─▶ spatial ─▶      │
                                        │   SceneMonitor ─▶ Announcer ─▶ speech queue    │
                                        │ VLM (single-flight) ──HTTP──▶ llama-server     │
                                        │                               127.0.0.1:8090   │
                                        └───────────────────────────────────────────────┘
```

```
backend/
  main.py              FastAPI app (REST, WebSocket, MJPEG, serves frontend/dist)
  pipeline.py          orchestrator and detection loop
  camera/service.py    capture thread, pause/resume, failure detection, enumeration
  detection/           YOLO11n ONNX detector (lazy load, dynamic input size)
  tracking/            ByteTrack + Kalman filter
  spatial/             relative position and distance buckets, obstacle classes
  assistant/           scene_monitor (events), announcer (speech policy), qa (grounded Q&A)
  vision/              VLM backends (llama.cpp, Ollama), llama-server supervisor, analyzer
  speech/              TTS engines (SAPI, pyttsx3, Piper) and priority speech queue
  config/settings.py   validated, persisted settings
  hardware.py          hardware compatibility check
frontend/src/          components/, hooks/, pages/, services/, styles/
scripts/               setup_models, export_yolo, benchmark, offline_check, start/dev
tests/                 84 automated tests
```

## Prerequisites

- Windows 10/11 (primary target). macOS and Linux should work but are untested; they use pyttsx3 or eSpeak for speech.
- Python 3.10–3.12, Node.js 18+ (only to build the UI), and a webcam
- About 1 GB of disk for models and tools

## Installation

```powershell
cd edge_ai
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python scripts\setup_models.py      # one-time downloads, ~560 MB
cd frontend; npm install; npm run build; cd ..
```

**The YOLO ONNX export is a one-time step.** It needs PyTorch, which AuraSight itself doesn't use at runtime, so do it in a throwaway environment:

```powershell
py -3.11 -m venv .export-env
.\.export-env\Scripts\pip install ultralytics onnx
.\.export-env\Scripts\python scripts\export_yolo.py   # writes models\yolo11n.onnx
```

See [models/README.md](models/README.md) for every file, its size, source and license.

## Running

**One command (Windows):**

```powershell
.\scripts\start.ps1          # opens http://127.0.0.1:8000
```

**Backend only:** `.\.venv\Scripts\python -m backend.main`. It serves the built UI at http://127.0.0.1:8000.

**Development with hot reload:** `.\scripts\dev.ps1` (or `scripts/dev.sh`) runs the backend on :8000 and Vite on http://127.0.0.1:5173.

By default the backend starts `tools/llama.cpp/llama-server` itself, bound to 127.0.0.1 with `--offline`. If you'd rather run it yourself:

```powershell
tools\llama.cpp\llama-server.exe -m models\vlm\SmolVLM-500M-Instruct-Q8_0.gguf `
  --mmproj models\vlm\mmproj-SmolVLM-500M-Instruct-Q8_0.gguf --host 127.0.0.1 --port 8090 -c 2048 -np 1 --offline
```

Then set `"vlm_autostart": false` in `data/settings.json`. If the scene model isn't available, AuraSight keeps working: descriptions and answers fall back to the object detector, and say so.

## Offline operation and privacy

- No cloud inference, no API keys, no analytics. The backend binds to `127.0.0.1` only.
- Video frames stay in memory, and only the latest frame is kept. Nothing is written to disk or recorded. The only copy of the video that leaves the capture thread is the preview stream, which goes to your own browser over localhost.
- The UI shows when the camera is on, both visually and in text, and the app says "Camera on" and "Camera off" aloud.
- The UI font (Atkinson Hyperlegible) is bundled. No CDN is used.
- **Verify it yourself:** `python scripts/offline_check.py` blocks every non-loopback connection and DNS lookup inside the process, then runs detection, tracking, a webcam frame, TTS and (if running) the VLM. It fails if anything tries to reach the network.

## Hardware requirements

Measured on the development machine (Intel i5-8350U, 4 cores/8 threads, 7.8 GB RAM, integrated graphics, no NVIDIA GPU), using `python scripts/benchmark.py`:

| Inference size | YOLO11n median | Max detector FPS |
|---|---|---|
| 256 px | 16 ms | ~64 |
| 320 px | 30 ms | ~33 |
| **416 px (default)** | **46 ms** | ~22 |
| 512 px | 60 ms | ~17 |
| 640 px | 101 ms | ~10 |

- ByteTrack: under 1 ms per frame. Backend process memory: about 150 MB with the detector loaded.
- Default detection rate is 5 FPS, which leaves CPU free for the VLM and the rest of the system.
- **Scene model (SmolVLM-500M Q8, llama.cpp CPU).** Measured on never-seen images (3 runs each):

  | Condition | Total (median) | First sentence spoken (median) |
  |---|---|---|
  | VLM alone | 13.1 s | 11.6 s |
  | With detection at 5 fps (old behaviour) | 16.2 s | 14.3 s |
  | With detection throttled to 1 fps during analysis (current) | 12.1 s | 10.8 s |

  - **Where the time goes:** about 10 s is the vision encoder processing the image. llama.cpp always encodes at SmolVLM's native 512 px, so shrinking the image client-side doesn't help. Text generation runs at about 23 tokens/s, so roughly 3 s for a three-sentence answer.
  - **Streaming:** sentences are spoken as soon as they're complete.
  - **Repeats are fast:** asking about the same frame again takes 1–4 s, because llama-server caches the image.
  - **Cold start:** the first request after starting the server is slower; AuraSight sends a tiny warm-up request when the model comes up.

| RAM | Recommendation |
|---|---|
| < 4 GB | Detector only (`vlm_backend: "none"`), 320 px |
| 4–8 GB | SmolVLM-500M (default) or 256M; close heavy apps |
| 8–16 GB | SmolVLM-500M comfortably; Moondream via Ollama is possible |
| 16 GB+ | Larger Ollama models are possible, but still slow on CPU |

`python -m backend.hardware` (or `GET /api/hardware`) prints a compatibility report for your machine.

## Configuration

Settings are editable in the UI (**Settings**, or press `,`) and are stored in `data/settings.json`. Key options:

| Setting | Default | Meaning |
|---|---|---|
| `inference_size` | 416 | Detector resolution: lower is faster, higher finds smaller objects |
| `detection_fps` | 5 | Detection passes per second; extra camera frames are skipped |
| `confidence_threshold` | 0.45 | Minimum *mean* score before an object is confirmed |
| `confusable_threshold` | 0.5 | Higher bar for classes YOLO confuses with windows and doors (`tv`, `refrigerator`, …) |
| `strong_confidence` | 0.7 | At or above this, a confusable object is confirmed without the zoomed-crop check |
| `min_track_frames` | 4 | Frames an object must be tracked before it can be confirmed |
| `vlm_max_tokens` | 80 | Upper bound on description length (about three sentences) |
| `vlm_backend` | `llamacpp` | `llamacpp`, `ollama` or `none` |
| `vlm_autostart` | true | Launch the bundled llama-server automatically |
| `auto_describe` | false | Periodic descriptions, and after big scene changes |
| `speech_rate` / `speech_volume` | 1.0 / 0.9 | Voice settings |
| `announce_cooldown_s` | 20 | Don't re-announce the same kind of object within this time |
| `obstacle_alerts` | true | Warnings for near objects in your path and fast approaches |

Environment variables: `AURASIGHT_PORT`, `AURASIGHT_DATA_DIR`, `AURASIGHT_MODELS_DIR`, `AURASIGHT_PIPER_EXE`, `AURASIGHT_PIPER_VOICE`.

**Testing without a webcam:** set `AURASIGHT_CAMERA_SOURCE` to an image, a folder of images (cycling every `AURASIGHT_SOURCE_SECONDS`, default 8) or a video file. The whole pipeline then runs on that input instead of the camera.

**Reading descriptions aloud:**
- Each sentence is spoken as soon as the model finishes it, using the voice, rate and volume set under Audio. What is spoken is exactly the text shown, sentence for sentence.
- The detector adds a closing sentence only for objects the model didn't mention, so nothing is said twice. "New: …" announcements are held while a description is being generated or read.
- Obstacle warnings still interrupt. The sentence they cut off is read again afterwards.
- A new description replaces one that is still playing. `POST /api/scene/replay` (the Replay button, or `R`) reads the last one again, and `POST /api/speech/stop` silences it, including sentences the model hasn't written yet.
- With **Speak out loud** off, descriptions are shown but not spoken. A TTS failure is shown in the Scene panel; the text stays complete.

**Diagnosing a description:** `GET /api/vlm/debug` returns the exact prompt, the model's raw output and the size of the image sent for the last request.

If you start llama-server yourself, these flags match what AuraSight uses when it launches the server: `-t 4 -tb 8 --cache-ram 256`. They set physical-core generation threads, all-core prompt threads, and cap the image cache, whose default is 8 GB.

## Keyboard shortcuts

These work anywhere except while typing in a text box. **Esc** always stops speech.

| Key | Action |
|---|---|
| `D` | Describe the scene (VLM) |
| `R` | Read the last description aloud again |
| `W` | What's around me? (instant, from the detector) |
| `A` or `/` | Ask a question |
| `S` or `Esc` | Stop speaking |
| `C` | Camera on/off |
| `P` | Pause/resume |
| `M` | Mute/unmute automatic announcements |
| `T` | Light/dark theme |
| `,` | Settings |
| `?` | Shortcut help |

Accessibility details:

- A "Skip to main controls" link.
- Native `<dialog>` modals, which trap focus and close on Esc.
- A polite live region for events and an assertive one for warnings. You can turn these off if your screen reader would double up with the app's voice.
- Every state is conveyed in text, not just colour.
- A 3 px focus ring and controls at least 44 px tall.
- The font is Atkinson Hyperlegible, designed by the Braille Institute for low-vision readers.
- `prefers-reduced-motion` is respected.

## Testing

```powershell
.\.venv\Scripts\python -m pytest -q          # 125 passed, 1 skipped (real-SmolVLM tests run when llama-server is up on :8090)
$env:AURASIGHT_SKIP_AUDIO="1"                # skip the test that plays a short phrase through the speakers
$env:AURASIGHT_SKIP_HARDWARE="1"             # skip the real-webcam test
```

| Area | What is tested |
|---|---|
| Detection | Real YOLO11n inference on real photos at 320/416/640 (bus and people, tie), box validity, blank image, missing-model error |
| Tracking | Persistent IDs while moving, confirmation, occlusion recovery, low-confidence second stage, class isolation, IDs on real detections |
| Spatial | Zones, class-aware distance, in-path, plurals |
| Camera | Start, frames, pause, resume, stop, open failure, driver exception, stream stall, bounded buffer, **real webcam** |
| Speech | Ordering, urgent interrupts, de-duplication, stale-drop, bounded queue, stop-all, engine failure, **real SAPI WAV synthesis and playback interrupt** |
| Scene events | Enter once, leave, ID-switch suppression, obstacle transition and cooldown, approaching, scene change, announcer grouping and mute |
| Object confirmation | **Real photos through YOLO and ByteTrack as jittered video.** Window→TV, glass panel→TV and door→fridge are never confirmed; three real TVs and person/laptop/chair are confirmed. Also covers temporal, flicker, crop-check and demotion rules |
| VLM | llama.cpp and Ollama streaming and non-streaming wire formats, warm-up, single-flight, the **"Left." regression** (degenerate answer is not spoken and is retried), sentence streaming, 10 real-phrasing rewrites. **Real SmolVLM** on 5 scene photos (describe) plus a question |
| Q&A | Target extraction and synonyms, intents, grounded answers, side questions from geometry, hedged VLM second opinion, terse-answer wording, unknown objects, fallbacks |
| Scenes (app level) | Real app on replayed photos: window-as-TV never announced and "Is there a TV?" says no; real TV announced and located; Describe streams sentences to speech and appends detector facts |
| API / E2E | Real app with real pipeline threads, real detector and tracker: WebSocket events, stable track IDs, announcements, MJPEG, describe fallback, ask, camera failure, settings validation |

## Troubleshooting

- **"Could not open camera"**: another app (Teams, Zoom, Camera) may be using it, or camera access is blocked in Windows Settings → Privacy & security → Camera → *Let desktop apps access your camera*.
- **The preview is black and nothing is detected**: check the privacy shutter or lens cover, and the lighting.
- **"Scene model: Unavailable"**: run `scripts/setup_models.py`, then check `data/llama-server.log`. Close memory-hungry apps; SmolVLM-500M needs about 1 GB free. Try the 256M model with `--small`.
- **No speech**: check that Windows speech voices exist (Settings → Time & language → Speech), and check the volume slider and *Speak out loud*.
- **Hearing everything twice with a screen reader**: turn off either *Speak out loud* or *Screen reader announcements*.
- **Slow or laggy**: lower `inference_size` to 320 and `detection_fps` to 3.

## Current limitations

- **Scene descriptions take about 11 s to the first spoken sentence on a 2018 laptop CPU**, and the vision encoder is most of that. Faster options are a GPU or a different model; SmolVLM-256M uses the same vision encoder, so it wouldn't be faster.
- **SmolVLM-500M still makes mistakes.** In testing it said a man sitting in an armchair was "sitting on the floor", called a furnished room "empty", and invented a person in a chair. The detector-confirmed sentence at the end of each description is the reliable part, and the descriptions avoid stating identifications as certain.
- **Some look-alike false positives can survive.** The stained-glass panel above a door is rejected only because its confidence stays under 0.5; it passes the zoomed-crop check. Small or distant real monitors that score under 0.5 stay "possible" and aren't announced.
- **Live detection on real objects has been verified only on photos.** During development the webcam image was almost black (brightness 8–9 out of 255, likely a closed shutter). Capture, detection, preview and WebSocket all ran on live frames, but detection quality was checked on real photos replayed through the full app.
- **The detector knows only the 80 COCO classes.** No doors, stairs, kerbs, keys or wallets. Questions about those go to the VLM, with a caveat.
- **Distances are heuristic buckets**, not measurements. There is no depth sensing.
- **Obstacle detection is limited.** A webcam has a narrow field of view and cannot see drop-offs, steps, glass or overhead hazards.
- **SmolVLM-500M is small.** Expect occasional vague or wrong descriptions; answers say "I'm not sure" when the model is uncertain.
- **A laptop webcam facing you will detect *you*.** For navigation, use an outward-facing or external camera.
- **Licensing:** YOLO11 weights are AGPL-3.0 (Ultralytics). That's fine for personal and open-source use, but check before distributing commercially.
