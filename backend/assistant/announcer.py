"""Decides which scene events are worth *speaking*, and how.

Everything still goes to the activity feed (visual + screen-reader live
region); only the spoken channel is filtered, grouped and prioritised here.
"""
from __future__ import annotations

import time
from typing import Callable

from ..speech.service import Priority, SpeechService
from ..spatial.positioning import plural, spoken_label
from .scene_monitor import ObjectView, SceneEvent


class Announcer:
    def __init__(self, settings_getter: Callable[[], object], speech: SpeechService,
                 clock: Callable[[], float] = time.monotonic):
        self._settings = settings_getter
        self._speech = speech
        self._clock = clock
        self._label_last: dict[str, float] = {}

    def reset(self) -> None:
        self._label_last.clear()

    def handle(self, events: list[SceneEvent]) -> list[str]:
        """Speak what matters; returns the spoken phrases (for tests/logging)."""
        s = self._settings()
        if s.announcements_muted or not events:
            return []
        now = self._clock()
        spoken: list[str] = []

        warnings = [e for e in events if e.kind in ("obstacle", "approaching")]
        if s.obstacle_alerts and warnings:
            text = " ".join(e.message for e in warnings[:2])
            if self._speech.say(text, Priority.URGENT, kind="alert", max_age=2.5):
                spoken.append(text)
            for e in warnings:
                if e.label:
                    self._label_last[e.label] = now

        if s.announce_new_objects:
            fresh = []
            for e in events:
                if e.kind != "entered" or not e.label:
                    continue
                last = self._label_last.get(e.label, -1e9)
                if now - last < s.announce_cooldown_s:
                    continue
                self._label_last[e.label] = now
                fresh.append(e)
            if fresh:
                parts = []
                for e in fresh[:3]:
                    where = e.spatial.direction if e.spatial else ""
                    if s.verbosity == "detailed" and e.spatial:
                        where = e.spatial.phrase()
                    parts.append(f"{spoken_label(e.label)} {where}".strip())
                text = "New: " + ", ".join(parts)
                if len(fresh) > 3:
                    text += f", and {len(fresh) - 3} more"
                text += "."
                if self._speech.say(text, Priority.NORMAL, kind="new_objects", max_age=4.0):
                    spoken.append(text)

        if s.announce_departures:
            gone = [spoken_label(e.label) for e in events if e.kind == "left" and e.label]
            if gone:
                text = ", ".join(gone[:3]) + (" gone." if len(gone) > 1 else " gone.")
                text = text[:1].upper() + text[1:]
                if self._speech.say(text, Priority.LOW, kind="departure", max_age=4.0):
                    spoken.append(text)

        if s.announce_scene_changes and any(e.kind == "scene_change" for e in events):
            text = "The view has changed."
            if self._speech.say(text, Priority.NORMAL, kind="scene_change", max_age=3.0):
                spoken.append(text)
        return spoken


def summarize_objects(objects: list[ObjectView], max_groups: int = 6, detailed: bool = True) -> str:
    """Detector-only summary, e.g. 'a laptop ahead, near; 2 chairs on your right'."""
    if not objects:
        return ""
    # closest first, then most central
    order = {"near": 0, "middle": 1, "far": 2}
    objs = sorted(objects, key=lambda o: (order[o.spatial.distance], abs(o.spatial.cx - 0.5)))
    groups: dict[tuple[str, str], list[ObjectView]] = {}
    for o in objs:
        groups.setdefault((o.label, o.spatial.direction), []).append(o)
    parts = []
    for (label, direction), items in list(groups.items())[:max_groups]:
        name = spoken_label(label)
        n = len(items)
        head = f"{_article(name)} {name}" if n == 1 else f"{n} {plural(name, n)}"
        tail = f"{direction}, {items[0].spatial.distance_phrase}" if detailed else direction
        parts.append(f"{head} {tail}")
    extra = len(groups) - max_groups
    text = "; ".join(parts)
    if extra > 0:
        text += f"; and {extra} more"
    return text


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"
