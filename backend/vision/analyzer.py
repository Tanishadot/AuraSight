"""Scene understanding on demand (never per frame).

The VLM is far more expensive than the detector (seconds vs. tens of ms on a
laptop CPU), so it only runs on explicit requests, answers to questions, or an
optional slow periodic timer. Only one VLM request is in flight at a time.

Prompt design (measured on tests/fixtures/scenes with SmolVLM-500M, see README):
  * Short, plain instructions work; multi-clause instructions do not. The old
    prompt ended with "...left, right or ahead" and the 500M model answered it
    like a multiple-choice question with the single word "Left.".
  * Repetition penalties are required - without them the model loops
    ("a bottle, a bottle, a bottle ...").
  * llama.cpp always encodes the image at SmolVLM's native 512 px, so
    client-side resizing only affects upload size, not inference time.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .backends import (LlamaCppBackend, LlamaServerProcess, OllamaBackend, VLMBackend,
                       VLMUnavailable, wait_until_healthy)

log = logging.getLogger("aurasight.vlm")

DESCRIBE_PROMPT = "Describe this image for a blind person in two or three short sentences."
DESCRIBE_FALLBACK_PROMPT = "Describe this image in two or three sentences."
QUESTION_PROMPT = "{question} Explain to a blind person in one or two short sentences."
PRESENCE_PROMPT = "Is there a {thing} in this image? Answer yes or no, then say where it is."
HINT_PREFIX = "An object detector found: {labels}.\n"

GENERATION = {"temperature": 0.1, "top_p": 0.9, "repeat_penalty": 1.15,
              "frequency_penalty": 0.3, "presence_penalty": 0.3}


class AnalyzerBusy(RuntimeError):
    pass


@dataclass
class AnalysisResult:
    text: str
    latency_ms: float
    model: str
    raw: str
    first_sentence_ms: float | None = None
    retried: bool = False
    sentences: list[str] = field(default_factory=list)


def encode_frame(frame_bgr: np.ndarray, max_side: int, quality: int = 85) -> bytes:
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


# --------------------------------------------------------------------------- #
# Output cleaning: model phrasing -> what a blind user needs to hear
# --------------------------------------------------------------------------- #
_IMG = r"(?:image|picture|photo|photograph|frame)"
_REWRITES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"<[^>]+>"), " "),  # stray special tokens
    (re.compile(r"^\s*(?:assistant|answer)\s*:\s*", re.I), ""),
    (re.compile(rf"^in (?:the|this) {_IMG},?\s*(?:i|we) can see\s+", re.I), "I can see "),
    (re.compile(rf"^(?:the|this) {_IMG} (?:shows|depicts|features)\s+", re.I), "I can see "),
    (re.compile(rf"^in (?:the|this) {_IMG},?\s*", re.I), ""),
    (re.compile(rf"^(?:this|the) {_IMG} is taken (?:in|inside|at) ", re.I), "This is "),
    (re.compile(r"\bwe can see\b", re.I), "I can see"),
    (re.compile(rf"\b(?:on|at|to) (?:the )?(left|right)(?: hand)?(?: side| corner)?(?: side| corner)? of (?:the|this) (?:{_IMG}|room|frame)\b", re.I), r"on your \1"),
    # "on the left side" -> "on your left", but not "on the left side of the wall" (relational)
    (re.compile(r"\b(?:on|at) (?:the )?(left|right)(?: hand)? (?:side|corner)(?: corner)?\b(?! of\b)", re.I), r"on your \1"),
    (re.compile(rf"\b(?:in|at) the (?:center|centre|middle) of (?:the|this) {_IMG}\b", re.I), "straight ahead"),
    (re.compile(rf"\bin the foreground(?: of (?:the|this) {_IMG})?", re.I), "closest to you"),
    (re.compile(rf"\bin the background(?: of (?:the|this) {_IMG})?", re.I), "further away"),
    (re.compile(r"^this is (an?|the) ", re.I), r"This looks like \1 "),
]
_META = re.compile(
    rf"(the focus should|blind person|as an ai|^(?:this|the) {_IMG} is (?:a |an )?(?:black and white|colou?r|blurry))",
    re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _dedupe_list_items(s: str) -> str:
    """'a bottle, a bottle, a cup, a bottle' -> 'a bottle, a cup'."""
    end = s[-1] if s[-1:] in ".!?" else ""
    parts = [p.strip() for p in (s[:-1] if end else s).split(",")]
    if len(parts) < 3:
        return s
    seen, out = set(), []
    for p in parts:
        key = re.sub(r"^(and|or)\s+", "", p.lower()).strip(" .")
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return ", ".join(out) + end


def clean_sentence(s: str) -> str:
    s = " ".join(s.replace("*", " ").replace("#", " ").split())
    for pat, rep in _REWRITES:
        s = pat.sub(rep, s).strip()
    s = _dedupe_list_items(s)
    if not s or _META.search(s):
        return ""
    return s[:1].upper() + s[1:]


def clean_text(text: str, max_sentences: int = 3) -> str:
    """Normalise model output for speech: rewrite into the user's frame of
    reference, drop meta-commentary and loops, cap length, never cut mid-sentence."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split())
    sentences = [x for x in _SENT_SPLIT.split(text) if x.strip()]
    out, seen = [], set()
    for i, raw in enumerate(sentences):
        complete = raw.rstrip()[-1:] in ".!?"
        s = clean_sentence(raw)
        if not s:
            continue
        if not complete:
            if out:  # truncated tail after full sentences: drop it
                break
            s += "."
        key = s.lower().strip(" .")
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= max_sentences:
            break
    return " ".join(out)


def is_degenerate(text: str, min_words: int = 5) -> bool:
    """Too short to be a description (e.g. 'Left.', 'Room.')."""
    return len(re.findall(r"[A-Za-z']+", text)) < min_words


class SentenceStream:
    """Feeds streamed tokens; calls ``on_sentence`` with each cleaned, complete sentence."""

    def __init__(self, on_sentence: Callable[[str], None] | None, max_sentences: int, min_words: int = 0):
        self.on_sentence = on_sentence
        self.max_sentences = max_sentences
        self.min_words = min_words  # hold sentences back until the answer is clearly not degenerate
        self.buf = ""
        self.sentences: list[str] = []
        self._pending: list[str] = []
        self._seen: set[str] = set()
        self.first_sentence_at: float | None = None

    @property
    def full(self) -> bool:
        return len(self.sentences) >= self.max_sentences

    def feed(self, delta: str) -> None:
        self.buf += delta
        while not self.full:
            m = re.search(r"[.!?](?=\s)", self.buf)
            if not m:
                break
            raw, self.buf = self.buf[: m.end()], self.buf[m.end():]
            self._emit(raw)

    def finish(self, final: bool = True) -> None:
        """Flush the tail. ``final=False`` keeps short (degenerate) output unspoken
        so the caller can retry with another prompt."""
        rest = self.buf.strip()
        self.buf = ""
        if rest and not self.full:
            if rest[-1] in ".!?" or not self.sentences:
                self._emit(rest if rest[-1] in ".!?" else rest + ".")
        if final:
            self._flush()

    def _emit(self, raw: str) -> None:
        s = clean_sentence(raw)
        key = s.lower().strip(" .")
        if not s or key in self._seen:
            return
        self._seen.add(key)
        self.sentences.append(s)
        self._pending.append(s)
        if len(re.findall(r"[A-Za-z']+", self.text)) >= self.min_words:
            self._flush()

    def _flush(self) -> None:
        for s in self._pending:
            if self.first_sentence_at is None:
                self.first_sentence_at = time.perf_counter()
            if self.on_sentence:
                try:
                    self.on_sentence(s)
                except Exception:
                    log.exception("sentence callback failed")
        self._pending = []

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


# --------------------------------------------------------------------------- #
class SceneAnalyzer:
    def __init__(self, settings_getter: Callable[[], object], models_dir: Path | None, tools_dir: Path | None,
                 data_dir: Path | None, on_status: Callable[[dict], None] | None = None,
                 backend: VLMBackend | None = None):
        from ..config.settings import DATA_DIR, MODELS_DIR, TOOLS_DIR

        self._settings = settings_getter
        self.models_dir = Path(models_dir or MODELS_DIR)
        self.tools_dir = Path(tools_dir or TOOLS_DIR)
        self.data_dir = Path(data_dir or DATA_DIR)
        self._on_status = on_status or (lambda s: None)
        self._injected = backend is not None
        self.backend: VLMBackend | None = backend
        self.process: LlamaServerProcess | None = None
        self._busy = threading.Lock()
        self._start_lock = threading.Lock()
        self.state = "not_started"  # not_started|disabled|starting|ready|busy|unavailable|error
        self.detail = ""
        self.last_latency_ms: float | None = None
        self.last_first_sentence_ms: float | None = None
        self.last_raw: str | None = None
        self.last_prompt: str | None = None
        self.last_image_shape: tuple | None = None
        self.model_name = ""
        if not self._injected:
            self._build_backend()

    # -- setup ------------------------------------------------------------------
    def _build_backend(self) -> None:
        s = self._settings()
        if self.process:
            self.process.stop()
            self.process = None
        if s.vlm_backend == "none":
            self.backend = None
            self._set("disabled", "Scene model disabled in settings")
            return
        if s.vlm_backend == "ollama":
            model = s.vlm_model if not s.vlm_model.endswith(".gguf") else "moondream"
            url = s.vlm_url if "8090" not in s.vlm_url else "http://127.0.0.1:11434"
            self.backend = OllamaBackend(url, model)
            self.model_name = f"ollama:{model}"
        else:
            self.backend = LlamaCppBackend(s.vlm_url, s.vlm_model)
            self.model_name = Path(s.vlm_model).stem
            if s.vlm_autostart:
                exe = self.tools_dir / "llama.cpp" / ("llama-server.exe" if _is_win() else "llama-server")
                port = int(s.vlm_url.rsplit(":", 1)[-1].split("/")[0])
                self.data_dir.mkdir(parents=True, exist_ok=True)
                self.process = LlamaServerProcess(
                    exe, self.models_dir / "vlm" / s.vlm_model, self.models_dir / "vlm" / s.vlm_mmproj,
                    port, s.vlm_threads, log_path=self.data_dir / "llama-server.log")
        self._set("not_started", "")

    def reconfigure(self) -> None:
        if self._injected:
            return
        self._build_backend()
        self.start_background()

    def _set(self, state: str, detail: str) -> None:
        self.state, self.detail = state, detail
        self._on_status(self.status())

    def status(self) -> dict:
        return {"state": self.state, "detail": self.detail, "model": self.model_name,
                "backend": getattr(self.backend, "name", None),
                "last_latency_ms": None if self.last_latency_ms is None else round(self.last_latency_ms),
                "last_first_sentence_ms": None if self.last_first_sentence_ms is None
                else round(self.last_first_sentence_ms)}

    def debug(self) -> dict:
        """Exactly what was last sent to / received from the model (no image bytes)."""
        return {"prompt": self.last_prompt, "raw": self.last_raw, "image_shape": self.last_image_shape,
                **self.status()}

    def start_background(self) -> None:
        threading.Thread(target=self.ensure_ready, kwargs={"timeout": 240}, name="vlm-start",
                         daemon=True).start()

    def ensure_ready(self, timeout: float = 240) -> bool:
        if self.backend is None:
            return False
        if self.state in ("ready", "busy"):
            return True
        with self._start_lock:
            if self.state in ("ready", "busy"):
                return True
            ok, msg = self.backend.health()
            if not ok and self.process is not None:
                missing = self.process.missing_files()
                if missing:
                    self._set("unavailable", "Scene model files missing - run scripts/setup_models.py")
                    return False
                self._set("starting", "Loading scene model…")
                if not self.process.start():
                    self._set("error", self.process.error or "could not start")
                    return False
                ok, msg = wait_until_healthy(self.backend, timeout, proc=self.process)
            if not ok:
                self._set("unavailable", msg)
                return False
            self._warmup()
            self._set("ready", "")
            return True

    def _warmup(self) -> None:
        """One tiny request so weights are paged in before the user's first click."""
        try:
            self._set("starting", "Warming up scene model…")
            img = np.full((64, 64, 3), 128, np.uint8)
            self.backend.generate(encode_frame(img, 64), "Describe this image.", 1, 120, GENERATION)
        except Exception as e:  # warm-up is best effort
            log.info("VLM warm-up skipped: %s", e)

    def shutdown(self) -> None:
        if self.process:
            self.process.stop()

    # -- inference -----------------------------------------------------------
    def _run(self, frame_bgr: np.ndarray, prompts: list[str], max_tokens: int | None = None,
             max_sentences: int = 3, on_sentence: Callable[[str], None] | None = None,
             min_words: int = 5) -> AnalysisResult:
        """Run the first prompt; if the answer is degenerate, retry with the next.
        Retries are cheap because llama-server caches the image embedding."""
        s = self._settings()
        if self.backend is None:
            raise VLMUnavailable("Scene model is disabled")
        if not self._busy.acquire(blocking=False):
            raise AnalyzerBusy("A scene analysis is already running")
        try:
            if self.state != "ready" and not self.ensure_ready(timeout=30):
                raise VLMUnavailable(self.detail or "Scene model unavailable")
            self._set("busy", "Analysing scene…")
            jpeg = encode_frame(frame_bgr, s.vlm_image_size)
            self.last_image_shape = tuple(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape)
            t0 = time.perf_counter()
            result = None
            for attempt, prompt in enumerate(prompts):
                last_attempt = attempt == len(prompts) - 1
                stream = SentenceStream(on_sentence, max_sentences, min_words)
                self.last_prompt = prompt
                try:
                    raw = self.backend.generate(jpeg, prompt, max_tokens or s.vlm_max_tokens, s.vlm_timeout_s,
                                                GENERATION, on_delta=stream.feed)
                except VLMUnavailable as e:
                    ok, _ = self.backend.health()
                    self._set("ready" if ok else "unavailable", str(e))
                    raise
                stream.finish(final=last_attempt)
                self.last_raw = raw
                log.info("VLM raw (%s): %r", "retry" if attempt else "first", raw[:300])
                text = stream.text or clean_text(raw, max_sentences)
                degenerate = is_degenerate(text, min_words)
                if not degenerate and not last_attempt:
                    stream.finish(final=True)  # release anything still held back
                first = None if stream.first_sentence_at is None else (stream.first_sentence_at - t0) * 1000
                result = AnalysisResult(text, 0.0, self.model_name, raw, first, attempt > 0, stream.sentences)
                if not degenerate or last_attempt:
                    break
            result.latency_ms = (time.perf_counter() - t0) * 1000
            self.last_latency_ms = result.latency_ms
            self.last_first_sentence_ms = result.first_sentence_ms
            self._set("ready", "")
            return result
        finally:
            self._busy.release()

    def describe(self, frame_bgr: np.ndarray, hint_labels: list[str] | None = None,
                 on_sentence: Callable[[str], None] | None = None) -> AnalysisResult:
        hint = HINT_PREFIX.format(labels=", ".join(hint_labels)) if hint_labels else ""
        return self._run(frame_bgr, [hint + DESCRIBE_PROMPT, DESCRIBE_FALLBACK_PROMPT],
                         on_sentence=on_sentence)

    def ask(self, frame_bgr: np.ndarray, question: str, hint: str = "") -> AnalysisResult:
        prompt = QUESTION_PROMPT.format(question=question.strip())
        if hint:
            prompt = f"{hint}\n{prompt}"
        return self._run(frame_bgr, [prompt], max_tokens=60, max_sentences=2, min_words=1)

    def check_presence(self, frame_bgr: np.ndarray, thing: str) -> tuple[bool | None, AnalysisResult]:
        res = self._run(frame_bgr, [PRESENCE_PROMPT.format(thing=thing)], max_tokens=40, max_sentences=2,
                        min_words=1)
        first = res.text.strip().lower()
        verdict = True if first.startswith("yes") else False if first.startswith("no") else None
        return verdict, res

    @property
    def available(self) -> bool:
        return self.backend is not None and self.state in ("ready", "busy")

    @property
    def busy(self) -> bool:
        return self._busy.locked()


def _is_win() -> bool:
    import sys

    return sys.platform == "win32"
