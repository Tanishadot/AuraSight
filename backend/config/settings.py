"""Runtime settings for AuraSight.

Settings are validated with pydantic and persisted as JSON in the data
directory (``AURASIGHT_DATA_DIR``, default ``<repo>/data``). Nothing in here
ever leaves the machine.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("AURASIGHT_MODELS_DIR", REPO_ROOT / "models"))
TOOLS_DIR = Path(os.environ.get("AURASIGHT_TOOLS_DIR", REPO_ROOT / "tools"))
DATA_DIR = Path(os.environ.get("AURASIGHT_DATA_DIR", REPO_ROOT / "data"))


class Settings(BaseModel):
    # --- Camera -----------------------------------------------------------
    camera_index: int = Field(0, ge=0, le=16)
    camera_width: int = Field(640, ge=160, le=1920)
    camera_height: int = Field(480, ge=120, le=1080)
    preview_fps: float = Field(12.0, ge=1, le=30)
    preview_quality: int = Field(70, ge=30, le=95)

    # --- Detection --------------------------------------------------------
    detector_model: str = "yolo11n.onnx"
    inference_size: Literal[256, 320, 416, 480, 512, 640] = 416
    detection_fps: float = Field(5.0, ge=0.5, le=30)
    confidence_threshold: float = Field(0.45, ge=0.1, le=0.95)
    # Higher bar for classes YOLO confuses with windows/doors (tv, refrigerator...)
    confusable_threshold: float = Field(0.5, ge=0.1, le=0.95)
    # Above this mean score a confusable class is confirmed without crop re-check
    strong_confidence: float = Field(0.7, ge=0.3, le=0.99)
    detector_threads: int = Field(0, ge=0, le=32)  # 0 = auto

    # --- Vision-language model -------------------------------------------
    vlm_backend: Literal["llamacpp", "ollama", "none"] = "llamacpp"
    vlm_url: str = "http://127.0.0.1:8090"
    vlm_model: str = "SmolVLM-500M-Instruct-Q8_0.gguf"
    vlm_mmproj: str = "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf"
    vlm_autostart: bool = True  # launch the bundled llama-server ourselves
    vlm_threads: int = Field(0, ge=0, le=32)  # 0 = auto
    vlm_image_size: int = Field(512, ge=224, le=1024)
    vlm_max_tokens: int = Field(80, ge=16, le=512)
    vlm_timeout_s: float = Field(120.0, ge=5, le=600)
    auto_describe: bool = False
    auto_describe_interval_s: float = Field(45.0, ge=10, le=600)

    # --- Speech -----------------------------------------------------------
    speech_enabled: bool = True
    speech_rate: float = Field(1.0, ge=0.5, le=2.0)  # multiplier
    speech_volume: float = Field(0.9, ge=0.0, le=1.0)
    voice_id: str | None = None

    # --- Announcements & alerts --------------------------------------------
    announcements_muted: bool = False
    announce_new_objects: bool = True
    announce_departures: bool = False
    announce_scene_changes: bool = True
    obstacle_alerts: bool = True
    announce_cooldown_s: float = Field(20.0, ge=0, le=600)
    min_track_frames: int = Field(4, ge=1, le=30)
    leave_after_s: float = Field(2.0, ge=0.5, le=30)
    verbosity: Literal["brief", "detailed"] = "brief"

    # --- UI -----------------------------------------------------------------
    theme: Literal["system", "light", "dark"] = "system"
    screen_reader_announcements: bool = True


class SettingsStore:
    """Thread-safe settings holder with JSON persistence and change listeners."""

    def __init__(self, path: Path | None = None, persist: bool = True):
        self.path = path or (DATA_DIR / "settings.json")
        self.persist = persist
        self._lock = threading.RLock()
        self._listeners: list[Callable[[Settings, set[str]], None]] = []
        self._settings = self._load()

    def _load(self) -> Settings:
        if self.persist and self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                # Ignore unknown/obsolete keys instead of failing.
                known = {k: v for k, v in data.items() if k in Settings.model_fields}
                return Settings(**known)
            except (OSError, ValueError, ValidationError):
                pass  # corrupt file -> defaults
        return Settings()

    @property
    def current(self) -> Settings:
        with self._lock:
            return self._settings

    def update(self, patch: dict[str, Any]) -> Settings:
        with self._lock:
            unknown = set(patch) - set(Settings.model_fields)
            if unknown:
                raise ValueError(f"Unknown setting(s): {', '.join(sorted(unknown))}")
            merged = {**self._settings.model_dump(), **patch}
            new = Settings(**merged)  # raises ValidationError on bad values
            changed = {k for k in patch if getattr(new, k) != getattr(self._settings, k)}
            self._settings = new
            if self.persist:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(new.model_dump_json(indent=2), encoding="utf-8")
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(new, changed)
            except Exception:  # a broken listener must not break settings
                pass
        return new

    def on_change(self, fn: Callable[[Settings, set[str]], None]) -> None:
        self._listeners.append(fn)
