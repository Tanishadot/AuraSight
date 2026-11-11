"""Hardware compatibility check: ``python -m backend.hardware``."""
from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from pathlib import Path

from .config.settings import MODELS_DIR, TOOLS_DIR

# Measured / documented peak resident memory (approximate) for each component.
FOOTPRINT_MB = {
    "backend + detector (YOLO11n ONNX @416)": 350,
    "SmolVLM-500M Q8 via llama-server (ctx 2048)": 1100,
    "SmolVLM-256M Q8 via llama-server (ctx 2048)": 650,
    "moondream (Ollama)": 2600,
    "llama3.2-vision 11B (Ollama)": 9000,
}


def check() -> dict:
    info: dict = {"os": f"{platform.system()} {platform.release()}", "python": sys.version.split()[0],
                  "cpu": platform.processor() or platform.machine(), "logical_cpus": os.cpu_count()}
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 2**30, 1)
        info["ram_available_gb"] = round(vm.available / 2**30, 1)
        info["physical_cores"] = psutil.cpu_count(logical=False)
    except ImportError:
        info["ram_total_gb"] = info["ram_available_gb"] = None
    du = shutil.disk_usage(MODELS_DIR if MODELS_DIR.exists() else Path.cwd())
    info["disk_free_gb"] = round(du.free / 2**30, 1)
    info["nvidia_gpu"] = shutil.which("nvidia-smi") is not None
    try:
        import onnxruntime as ort

        info["onnx_providers"] = ort.get_available_providers()
    except ImportError:
        info["onnx_providers"] = []

    exe = TOOLS_DIR / "llama.cpp" / ("llama-server.exe" if os.name == "nt" else "llama-server")
    info["files"] = {
        "detector": (MODELS_DIR / "yolo11n.onnx").exists(),
        "vlm_model": (MODELS_DIR / "vlm" / "SmolVLM-500M-Instruct-Q8_0.gguf").exists(),
        "vlm_mmproj": (MODELS_DIR / "vlm" / "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf").exists(),
        "llama_server": exe.exists(),
    }
    ram = info.get("ram_total_gb") or 0
    rec = []
    if ram and ram < 4:
        rec.append("Under 4 GB RAM: disable the scene model (vlm_backend='none') and use inference_size 320.")
    elif ram and ram < 8:
        rec.append("4-8 GB RAM: use SmolVLM-500M (default) or SmolVLM-256M; close other heavy apps.")
    elif ram and ram < 16:
        rec.append("8-16 GB RAM: SmolVLM-500M is comfortable; moondream via Ollama is possible.")
    else:
        rec.append("16 GB+ RAM: larger VLMs (e.g. llama3.2-vision via Ollama) are possible but slow on CPU.")
    if not info["nvidia_gpu"]:
        rec.append("No NVIDIA GPU detected: everything runs on CPU (supported). "
                   "Keep detection_fps at 3-6 and inference_size 320-416.")
    if (info.get("ram_available_gb") or 99) < 1.5:
        rec.append("Less than 1.5 GB RAM currently free: the scene model may load slowly or swap.")
    info["recommendations"] = rec
    info["footprint_mb"] = FOOTPRINT_MB
    return info


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
