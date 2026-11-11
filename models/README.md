# Models

Run `python scripts/setup_models.py` to download everything (SHA-256 verified where pinned).
Nothing here is committed to git. After download, AuraSight needs no network access.

| File | Size | Source | License | Used for |
|---|---|---|---|---|
| `yolo11n.pt` | 5.4 MB | github.com/ultralytics/assets v8.3.0 | AGPL-3.0 | source weights |
| `yolo11n.onnx` | 10.1 MB | exported locally via `scripts/export_yolo.py` | AGPL-3.0 | live object detection |
| `vlm/SmolVLM-500M-Instruct-Q8_0.gguf` | 437 MB | huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF | Apache-2.0 | scene descriptions / Q&A |
| `vlm/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf` | 109 MB | same | Apache-2.0 | image encoder for the VLM |
| `../tools/llama.cpp/` | ~60 MB | github.com/ggml-org/llama.cpp release b12273 | MIT | local VLM inference server |

Alternatives: `--small` downloads SmolVLM-256M (~280 MB, ~650 MB RAM, weaker answers).
With Ollama installed you can instead use `ollama pull moondream` (1.7 GB) and pick
"Ollama" as the scene-model runtime in Settings.
