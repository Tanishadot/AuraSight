"""Local vision-language model (VLM) backends.

Different runtimes expose different APIs, so each gets its own adapter:

* ``LlamaCppBackend`` – llama.cpp ``llama-server`` (OpenAI-compatible
  ``/v1/chat/completions`` with base64 ``image_url`` parts). Default, used with
  SmolVLM-500M-Instruct (GGUF). Lightest option that runs well on CPU.
* ``OllamaBackend`` – Ollama ``/api/generate`` with an ``images`` array. Use
  with ``moondream`` (1.7 GB) or ``llama3.2-vision`` (7.9 GB, needs >=12 GB RAM).

Both talk only to 127.0.0.1 by default. Neither sends anything off-machine.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Protocol

import httpx

log = logging.getLogger("aurasight.vlm")


class VLMUnavailable(RuntimeError):
    pass


class VLMBackend(Protocol):
    name: str

    def health(self) -> tuple[bool, str]: ...

    def generate(self, image_jpeg: bytes, prompt: str, max_tokens: int, timeout: float,
                 params: dict | None = None, on_delta: Callable[[str], None] | None = None) -> str: ...


def _b64(image_jpeg: bytes) -> str:
    return base64.b64encode(image_jpeg).decode("ascii")


class LlamaCppBackend:
    name = "llamacpp"

    def __init__(self, base_url: str, model: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def health(self) -> tuple[bool, str]:
        try:
            r = httpx.get(f"{self.base_url}/health", timeout=2.0)
        except httpx.HTTPError:
            return False, f"llama-server is not running on {self.base_url.split('//')[-1]}"
        if r.status_code == 200:
            return True, "ready"
        if r.status_code == 503:
            return False, "model loading"
        return False, f"llama-server returned HTTP {r.status_code}"

    def generate(self, image_jpeg: bytes, prompt: str, max_tokens: int, timeout: float,
                 params: dict | None = None, on_delta: Callable[[str], None] | None = None) -> str:
        """Image goes first: SmolVLM's chat template expects ``User:<image>text``."""
        body = {
            "model": self.model or "local",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(image_jpeg)}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
            **(params or {}),
            "stream": on_delta is not None,
        }
        url = f"{self.base_url}/v1/chat/completions"
        deadline = time.monotonic() + timeout
        try:
            if on_delta is None:
                r = httpx.post(url, json=body, timeout=timeout)
                if r.status_code != 200:
                    raise VLMUnavailable(f"Scene model error HTTP {r.status_code}: {r.text[:200]}")
                return (r.json()["choices"][0]["message"].get("content") or "").strip()
            parts: list[str] = []
            with httpx.stream("POST", url, json=body, timeout=httpx.Timeout(timeout, connect=5)) as r:
                if r.status_code != 200:
                    r.read()
                    raise VLMUnavailable(f"Scene model error HTTP {r.status_code}: {r.text[:200]}")
                for line in r.iter_lines():
                    if time.monotonic() > deadline:
                        raise VLMUnavailable(f"Scene model timed out after {timeout:.0f}s")
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        delta = json.loads(payload)["choices"][0].get("delta", {}).get("content") or ""
                    except (ValueError, KeyError, IndexError):
                        continue
                    if delta:
                        parts.append(delta)
                        on_delta(delta)
            return "".join(parts).strip()
        except httpx.TimeoutException as e:
            raise VLMUnavailable(f"Scene model timed out after {timeout:.0f}s") from e
        except httpx.HTTPError as e:
            raise VLMUnavailable(f"Scene model not reachable: {type(e).__name__}") from e


class OllamaBackend:
    name = "ollama"

    def __init__(self, base_url: str, model: str = "moondream"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def health(self) -> tuple[bool, str]:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            r.raise_for_status()
        except httpx.HTTPError as e:
            return False, f"Ollama not reachable at {self.base_url} ({type(e).__name__})"
        names = {m.get("name", "") for m in r.json().get("models", [])}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names and self.model not in names:
            return False, f"Ollama model '{self.model}' not pulled (run: ollama pull {self.model})"
        return True, "ready"

    def generate(self, image_jpeg: bytes, prompt: str, max_tokens: int, timeout: float,
                 params: dict | None = None, on_delta: Callable[[str], None] | None = None) -> str:
        opts = {"temperature": 0.1, "num_predict": max_tokens}
        for k in ("temperature", "top_p", "repeat_penalty", "frequency_penalty", "presence_penalty"):
            if params and k in params:
                opts[k] = params[k]
        body = {"model": self.model, "prompt": prompt, "images": [_b64(image_jpeg)],
                "stream": on_delta is not None, "options": opts}
        url = f"{self.base_url}/api/generate"
        deadline = time.monotonic() + timeout
        try:
            if on_delta is None:
                r = httpx.post(url, json=body, timeout=timeout)
                if r.status_code != 200:
                    raise VLMUnavailable(f"Ollama error HTTP {r.status_code}: {r.text[:200]}")
                return (r.json().get("response") or "").strip()
            parts: list[str] = []
            with httpx.stream("POST", url, json=body, timeout=httpx.Timeout(timeout, connect=5)) as r:
                if r.status_code != 200:
                    r.read()
                    raise VLMUnavailable(f"Ollama error HTTP {r.status_code}: {r.text[:200]}")
                for line in r.iter_lines():
                    if time.monotonic() > deadline:
                        raise VLMUnavailable(f"Scene model timed out after {timeout:.0f}s")
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except ValueError:
                        continue
                    delta = chunk.get("response") or ""
                    if delta:
                        parts.append(delta)
                        on_delta(delta)
                    if chunk.get("done"):
                        break
            return "".join(parts).strip()
        except httpx.TimeoutException as e:
            raise VLMUnavailable(f"Scene model timed out after {timeout:.0f}s") from e
        except httpx.HTTPError as e:
            raise VLMUnavailable(f"Ollama not reachable: {type(e).__name__}") from e


class LlamaServerProcess:
    """Launches and supervises the bundled llama.cpp server (optional)."""

    def __init__(self, exe: Path, model: Path, mmproj: Path, port: int, threads: int = 0,
                 ctx: int = 2048, log_path: Path | None = None):
        self.exe, self.model, self.mmproj = Path(exe), Path(model), Path(mmproj)
        self.port, self.threads, self.ctx = port, threads, ctx
        self.log_path = log_path
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.error: str | None = None

    def missing_files(self) -> list[str]:
        return [str(p) for p in (self.exe, self.model, self.mmproj) if not p.exists()]

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> bool:
        with self._lock:
            if self.running:
                return True
            missing = self.missing_files()
            if missing:
                self.error = "Missing VLM files: " + ", ".join(missing)
                return False
            threads = self.threads or max(1, (os.cpu_count() or 2) // 2)
            # Prompt/image processing is compute-bound and benefits from all logical cores;
            # token generation is memory-bound and runs best on physical cores.
            batch_threads = max(threads, os.cpu_count() or threads)
            cmd = [str(self.exe), "-m", str(self.model), "--mmproj", str(self.mmproj),
                   "--host", "127.0.0.1", "--port", str(self.port), "-c", str(self.ctx),
                   "-t", str(threads), "-tb", str(batch_threads), "-np", "1",
                   "--cache-ram", "256", "--offline"]
            out = open(self.log_path, "ab") if self.log_path else subprocess.DEVNULL
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            try:
                self._proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                                              creationflags=flags)
            except OSError as e:
                self.error = f"Could not start llama-server: {e}"
                return False
            self.error = None
            log.info("started llama-server pid=%s on port %s", self._proc.pid, self.port)
            return True

    def wait_exit(self, timeout: float) -> int | None:
        if not self._proc:
            return None
        try:
            return self._proc.wait(timeout)
        except subprocess.TimeoutExpired:
            return None

    def stop(self) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            self._proc = None


def wait_until_healthy(backend: VLMBackend, timeout: float, poll: float = 0.5,
                       proc: LlamaServerProcess | None = None) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout
    msg = "starting"
    while time.monotonic() < deadline:
        ok, msg = backend.health()
        if ok:
            return True, msg
        if proc is not None and proc._proc is not None and proc._proc.poll() is not None:
            return False, f"llama-server exited with code {proc._proc.returncode}"
        time.sleep(poll)
    return False, msg
