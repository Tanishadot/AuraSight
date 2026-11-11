"""Prioritised, non-overlapping, interruptible speech queue.

Rules:
  * Only one utterance plays at a time (single worker thread).
  * A higher-priority request interrupts lower-priority speech in progress
    (e.g. an obstacle warning cuts off an object announcement).
  * The queue is bounded; when full the lowest-priority / oldest item is dropped.
  * Time-sensitive items carry ``max_age`` and are discarded if they could not
    be spoken in time - stale "chair on your left" is worse than silence.
  * Identical text already waiting in the queue is not queued twice.
"""
from __future__ import annotations

import heapq
import itertools
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable

from .engines import TTSEngine, create_engine

log = logging.getLogger("aurasight.speech")


class Priority(IntEnum):
    LOW = 0  # departures, ambient info
    NORMAL = 1  # new objects, automatic descriptions
    USER = 2  # answers to things the user explicitly asked for
    URGENT = 3  # obstacle / approaching-object warnings


@dataclass(order=True)
class _Item:
    sort_key: tuple = field(init=False, repr=False)
    priority: Priority
    seq: int
    text: str = field(compare=False)
    kind: str = field(compare=False, default="info")
    created: float = field(compare=False, default_factory=time.monotonic)
    max_age: float | None = field(compare=False, default=None)
    cancelled: bool = field(compare=False, default=False)

    def __post_init__(self):
        self.sort_key = (-int(self.priority), self.seq)


class SpeechService:
    def __init__(
        self,
        settings_getter: Callable[[], object],
        engine_factory: Callable[[], TTSEngine] = create_engine,
        on_event: Callable[[str, dict], None] | None = None,
        max_queue: int = 10,
    ):
        self._settings = settings_getter
        self._factory = engine_factory
        self._on_event = on_event or (lambda *_: None)
        self._max_queue = max_queue
        self._heap: list[_Item] = []
        self._cond = threading.Condition()
        self._seq = itertools.count()
        self._current: _Item | None = None
        self._engine: TTSEngine | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._tasks: list[tuple[Callable[[TTSEngine], object], dict]] = []
        self.state = "not_started"  # not_started | ready | speaking | error
        self.error: str | None = None
        self.engine_name: str | None = None
        self.spoken_count = 0

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="speech", daemon=True)
        self._thread.start()
        self._ready.wait(10)

    def shutdown(self) -> None:
        self._stop.set()
        self.stop_all()
        with self._cond:
            self._cond.notify_all()
        if self._thread:
            self._thread.join(3)

    def status(self) -> dict:
        return {
            "state": self.state,
            "error": self.error,
            "engine": self.engine_name,
            "speaking": self._current.text if self._current else None,
            "queued": len(self._heap),
        }

    @property
    def is_speaking(self) -> bool:
        return self._current is not None

    @property
    def current_kind(self) -> str | None:
        """Kind of the utterance playing now, or of the next queued one."""
        with self._cond:
            if self._current is not None:
                return self._current.kind
            return self._heap[0].kind if self._heap else None

    # -- public API -------------------------------------------------------------
    def say(self, text: str, priority: Priority = Priority.NORMAL, kind: str = "info",
            max_age: float | None = None, dedupe: bool = True) -> bool:
        """Queue text. Returns False if dropped (disabled, unavailable, duplicate, or full).
        ``dedupe=False`` is for explicit repeats such as replaying a description."""
        text = " ".join(text.split())
        if not text:
            return False
        s = self._settings()
        if not getattr(s, "speech_enabled", True) or self.state == "error":
            return False
        with self._cond:
            if dedupe and (any(i.text == text for i in self._heap) or (
                self._current is not None and self._current.text == text
            )):
                return False
            item = _Item(Priority(priority), next(self._seq), text, kind, max_age=max_age)
            heapq.heappush(self._heap, item)
            while len(self._heap) > self._max_queue:
                # drop the least important, oldest item
                worst = max(self._heap, key=lambda i: (-int(i.priority), -i.seq))
                self._heap.remove(worst)
                heapq.heapify(self._heap)
            if self._current is not None and priority > self._current.priority and self._engine:
                self._engine.interrupt()
            self._cond.notify_all()
        return True

    def stop_all(self) -> None:
        """Silence immediately and clear everything queued."""
        with self._cond:
            self._heap.clear()
            if self._current is not None:
                self._current.cancelled = True
            if self._engine:
                self._engine.interrupt()
        self._on_event("speech", {"state": "stopped"})

    def cancel(self, kind: str) -> int:
        """Drop queued items of one kind and cut off the current one if it is of
        that kind. Other speech (alerts, answers) is untouched. Returns the number
        of items removed."""
        with self._cond:
            before = len(self._heap)
            self._heap = [i for i in self._heap if i.kind != kind]
            heapq.heapify(self._heap)
            removed = before - len(self._heap)
            if self._current is not None and self._current.kind == kind and self._engine:
                # flag covers the gap between the worker taking the item and playback starting
                self._current.cancelled = True
                self._engine.interrupt()
                removed += 1
        return removed

    def skip(self) -> None:
        with self._cond:
            if self._current is not None:
                self._current.cancelled = True
            if self._engine:
                self._engine.interrupt()

    def run_on_engine(self, fn: Callable[[TTSEngine], object], timeout: float = 30.0):
        """Execute ``fn(engine)`` on the speech thread (COM affinity) and return its result."""
        box: dict = {"done": threading.Event()}
        with self._cond:
            self._tasks.append((fn, box))
            self._cond.notify_all()
        if not box["done"].wait(timeout):
            raise TimeoutError("speech engine busy")
        if "error" in box:
            raise box["error"]
        return box.get("result")

    # -- worker -------------------------------------------------------------
    def _run(self) -> None:
        try:
            self._engine = self._factory()
            self.engine_name = getattr(self._engine, "name", "unknown")
            self.state = "ready"
        except Exception as e:
            self.state, self.error = "error", f"No offline TTS engine available: {e}"
            log.exception("TTS init failed")
            self._ready.set()
            return
        self._ready.set()
        self._on_event("speech", {"state": "ready", "engine": self.engine_name})
        while not self._stop.is_set():
            with self._cond:
                while not self._heap and not self._tasks and not self._stop.is_set():
                    self._cond.wait(0.5)
                if self._stop.is_set():
                    break
                if self._tasks:
                    fn, box = self._tasks.pop(0)
                    item = None
                else:
                    item = heapq.heappop(self._heap)
                    self._current = item
            if item is None:
                try:
                    box["result"] = fn(self._engine)
                except Exception as e:
                    box["error"] = e
                finally:
                    box["done"].set()
                continue
            with self._cond:
                stale = item.max_age is not None and time.monotonic() - item.created > item.max_age
                if stale or item.cancelled:
                    self._current = None
                    continue
            s = self._settings()
            self.state = "speaking"
            self._on_event("speech", {"state": "speaking", "text": item.text, "kind": item.kind,
                                      "priority": item.priority.name.lower()})
            completed = True
            failed: str | None = None
            try:
                completed = self._engine.speak(
                    item.text, getattr(s, "speech_rate", 1.0), getattr(s, "speech_volume", 0.9),
                    getattr(s, "voice_id", None),
                )
                self.spoken_count += 1
            except Exception as e:
                log.exception("speech failed")
                self.error = failed = str(e) or e.__class__.__name__
            finally:
                with self._cond:
                    self._current = None
                self.state = "ready"
                if not completed and not failed and not item.cancelled and item.priority >= Priority.USER:
                    # Pre-empted by an urgent warning (not stopped by the user):
                    # read the sentence again afterwards so nothing asked for is lost.
                    with self._cond:
                        heapq.heappush(self._heap, _Item(item.priority, item.seq, item.text, item.kind))
                        self._cond.notify_all()
                if failed:
                    self._on_event("speech", {"state": "error", "error": failed, "kind": item.kind})
                self._on_event("speech", {"state": "idle", "interrupted": not completed, "kind": item.kind})
        try:
            self._engine.close()
        except Exception:
            pass
