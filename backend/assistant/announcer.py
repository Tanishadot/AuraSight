"""Decides which scene events are worth *speaking*, and how.

Everything still goes to the activity feed (visual + screen-reader live
region); only the spoken channel is filtered, grouped and prioritised here.
"""
from __future__ import annotations

import re
import time
from typing import Callable

from ..speech.service import Priority, SpeechService
from ..spatial.positioning import plural, spoken_label
from .scene_monitor import ObjectView, SceneEvent


class Announcer:
    def __init__(self, settings_getter: Callable[[], object], speech: SpeechService,
                 clock: Callable[[], float] = time.monotonic,
                 hold: Callable[[], bool] = lambda: False):
        self._settings = settings_getter
        self._speech = speech
        self._clock = clock
        # True while a scene description is being generated or read aloud:
        # routine announcements would otherwise land between its sentences.
        self._hold = hold
        self._label_last: dict[str, float] = {}

    def reset(self) -> None:
        self._label_last.clear()

    def note_described(self, labels) -> None:
        """A scene description is covering these classes: treat them as just
        announced so "New: laptop" doesn't repeat what the description says."""
        now = self._clock()
        for label in labels:
            self._label_last[label] = now

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

        if self._hold():
            return spoken  # safety warnings above still interrupt; the rest is in the activity feed

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


# Words a scene description may use for a detector class. Used only to avoid
# repeating an object the description already named.
_SYNONYMS = {
    "person": ("person", "people", "man", "men", "woman", "women", "boy", "girl", "child", "children",
               "kid", "kids", "someone", "lady", "guy", "baby"),
    "tv": ("tv", "tvs", "television", "monitor", "monitors", "screen", "screens"),
    "dining table": ("table", "tables", "desk", "desks"),
    "cell phone": ("phone", "phones", "smartphone", "mobile"),
    "couch": ("couch", "sofa", "sofas", "couches"),
    "potted plant": ("plant", "plants", "flower", "flowers", "pot"),
    "laptop": ("laptop", "laptops", "computer", "notebook"),
    "cup": ("cup", "cups", "mug", "mugs"),
    "wine glass": ("glass", "glasses", "wine"),
    "car": ("car", "cars", "vehicle", "vehicles"),
    "bicycle": ("bicycle", "bicycles", "bike", "bikes"),
    "motorcycle": ("motorcycle", "motorbike", "scooter"),
    "refrigerator": ("refrigerator", "fridge"),
    "handbag": ("handbag", "bag", "purse"),
    "backpack": ("backpack", "bag", "rucksack"),
    "keyboard": ("keyboard",),
    "mouse": ("mouse",),
    "book": ("book", "books"),
    "bottle": ("bottle", "bottles"),
    "chair": ("chair", "chairs", "stool", "seat"),
    "bed": ("bed", "beds"),
    "dog": ("dog", "dogs", "puppy"),
    "cat": ("cat", "cats", "kitten"),
}


def mentioned_in(label: str, text: str) -> bool:
    """True if a description already names this detector class (or a synonym)."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    names = _SYNONYMS.get(label) or tuple(label.lower().split()) + (plural(label, 2).lower(),)
    if label not in _SYNONYMS and " " in label:  # multi-word class: needs the whole phrase
        return label.lower() in text.lower() or plural(label, 2).lower() in text.lower()
    return any(n in words for n in names)


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"
