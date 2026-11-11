"""Download every model/runtime file AuraSight needs, once. After this, no
network access is required.

    python scripts/setup_models.py            # detector + VLM + llama.cpp
    python scripts/setup_models.py --small    # SmolVLM-256M instead of 500M
    python scripts/setup_models.py --no-vlm   # detector only

Files and where they go:
    models/yolo11n.pt                      5.4 MB  Ultralytics YOLO11n weights (AGPL-3.0)
    models/yolo11n.onnx                   10.1 MB  exported from the .pt (scripts/export_yolo.py)
    models/vlm/SmolVLM-500M-Instruct-Q8_0.gguf       437 MB  (Apache-2.0)
    models/vlm/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf 109 MB
    tools/llama.cpp/                       ~60 MB  llama.cpp server binaries (MIT), pinned build
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
TOOLS = ROOT / "tools"

LLAMA_BUILD = "b12273"
HF = "https://huggingface.co"
FILES = {
    "yolo": [(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        MODELS / "yolo11n.pt",
        "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    )],
    "vlm500": [
        (f"{HF}/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct-Q8_0.gguf",
         MODELS / "vlm" / "SmolVLM-500M-Instruct-Q8_0.gguf",
         "9d4612de6a42214499e301494a3ecc2be0abdd9de44e663bda63f1152fad1bf4"),
        (f"{HF}/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-500M-Instruct-Q8_0.gguf",
         MODELS / "vlm" / "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf",
         "d1eb8b6b23979205fdf63703ed10f788131a3f812c7b1f72e0119d5d81295150"),
    ],
    "vlm256": [
        (f"{HF}/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf",
         MODELS / "vlm" / "SmolVLM-256M-Instruct-Q8_0.gguf", None),
        (f"{HF}/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
         MODELS / "vlm" / "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf", None),
    ],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, expected: str | None) -> None:
    if dest.exists() and (expected is None or sha256(dest) == expected):
        print(f"  ok      {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  fetch   {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "aurasight-setup"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r          {done / 2**20:7.1f} / {total / 2**20:.1f} MB", end="", flush=True)
    print()
    if expected and sha256(tmp) != expected:
        tmp.unlink()
        raise SystemExit(f"Checksum mismatch for {dest.name} - download corrupted or changed upstream.")
    tmp.replace(dest)


def llama_asset() -> str:
    sysname = platform.system()
    if sysname == "Windows":
        return f"llama-{LLAMA_BUILD}-bin-win-cpu-x64.zip"
    if sysname == "Darwin":
        return f"llama-{LLAMA_BUILD}-bin-macos-{'arm64' if platform.machine() == 'arm64' else 'x64'}.zip"
    return f"llama-{LLAMA_BUILD}-bin-ubuntu-x64.zip"


def install_llama() -> None:
    exe = TOOLS / "llama.cpp" / ("llama-server.exe" if platform.system() == "Windows" else "llama-server")
    if exe.exists():
        print(f"  ok      {exe.relative_to(ROOT)}")
        return
    asset = llama_asset()
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/{asset}"
    with tempfile.TemporaryDirectory() as td:
        z = Path(td) / asset
        download(url, z, None)
        out = TOOLS / "llama.cpp"
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            for m in zf.infolist():
                target = (out / Path(m.filename).name)
                if m.is_dir() or not Path(m.filename).name:
                    continue
                with zf.open(m) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    if platform.system() != "Windows":
        exe.chmod(0o755)
    print(f"  ok      {exe.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="use SmolVLM-256M (less RAM, lower quality)")
    ap.add_argument("--no-vlm", action="store_true", help="skip the scene model and llama.cpp")
    a = ap.parse_args()

    print("Detector (YOLO11n)")
    for f in FILES["yolo"]:
        download(*f)
    if not (MODELS / "yolo11n.onnx").exists():
        print("  -> models/yolo11n.onnx missing. Export it once with:\n"
              "     python -m venv .export-env && .export-env\\Scripts\\pip install ultralytics onnx\n"
              "     .export-env\\Scripts\\python scripts/export_yolo.py")
    if not a.no_vlm:
        print("Scene model (SmolVLM)")
        for f in FILES["vlm256" if a.small else "vlm500"]:
            download(*f)
        print("llama.cpp server")
        install_llama()
        if a.small:
            print('  -> set "vlm_model"/"vlm_mmproj" to the 256M files in Settings or data/settings.json')
    print("Done. AuraSight can now run fully offline.")


if __name__ == "__main__":
    sys.exit(main())
