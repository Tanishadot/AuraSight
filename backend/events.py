"""Thread-safe event bus bridging worker threads to asyncio WebSocket clients.

Worker threads (camera, detection, speech, VLM) call :meth:`EventBus.publish`.
Each WebSocket subscriber owns a bounded asyncio queue; when a slow client
falls behind, the *oldest* events are dropped so memory never grows.
"""
from __future__ import annotations

import asyncio
import itertools
import threading
import time
from collections import deque
from typing import Any

# Event types that are high-frequency and should not be kept in history.
_TRANSIENT = {"detections", "status", "vlm_status", "speech", "settings", "scene_progress"}


class EventBus:
    def __init__(self, history: int = 150, queue_size: int = 120):
        self._subs: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self._lock = threading.Lock()
        self._history: deque[dict[str, Any]] = deque(maxlen=history)
        self._queue_size = queue_size
        self._ids = itertools.count(1)

    # -- producers ----------------------------------------------------------
    def publish(self, type_: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        event = {"type": type_, "id": next(self._ids), "ts": time.time(), **(payload or {})}
        with self._lock:
            if type_ not in _TRANSIENT:
                self._history.append(event)
            subs = list(self._subs)
        for loop, q in subs:
            try:
                loop.call_soon_threadsafe(_put_drop_oldest, q, event)
            except RuntimeError:  # loop closed
                self.unsubscribe(q)
        return event

    def activity(self, kind: str, message: str, priority: str = "info", **extra: Any) -> dict[str, Any]:
        """Human-readable activity-feed entry."""
        return self.publish("activity", {"kind": kind, "message": message, "priority": priority, **extra})

    # -- consumers ----------------------------------------------------------
    def subscribe(self) -> asyncio.Queue:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        with self._lock:
            self._subs.append((loop, q))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subs = [(l, s) for (l, s) in self._subs if s is not q]

    def history(self, limit: int = 50, type_: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items = [e for e in self._history if type_ is None or e["type"] == type_]
        return items[-limit:]

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)


def _put_drop_oldest(q: asyncio.Queue, event: dict[str, Any]) -> None:
    if q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    q.put_nowait(event)
