"""Grounded question answering about the current camera view.

Strategy (designed to avoid confident hallucinations from small VLMs):

1. "Where is / is there / find / how many <object>" for objects the detector
   knows -> answer from **tracked detections** (fast, grounded, positional).
   If the detector does not see it, optionally ask the VLM for a second opinion
   but clearly label that answer as less reliable.
2. Objects the detector cannot recognise (keys, wallet, door...) -> VLM, with
   an explicit caveat.
3. Open questions ("describe this room", "what's on the table") -> VLM with a
   short list of detector findings as context, falling back to a detector-only
   summary if the VLM is unavailable.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..detection.yolo_onnx import COCO_NAMES
from ..spatial.positioning import plural, spoken_label
from ..vision.analyzer import AnalyzerBusy, SceneAnalyzer
from ..vision.backends import VLMUnavailable
from .announcer import summarize_objects
from .scene_monitor import ObjectView

SYNONYMS: dict[str, str] = {
    "water bottle": "bottle", "bottle": "bottle", "flask": "bottle",
    "phone": "cell phone", "mobile": "cell phone", "mobile phone": "cell phone",
    "cellphone": "cell phone", "smartphone": "cell phone", "iphone": "cell phone",
    "mug": "cup", "coffee cup": "cup", "cup": "cup", "glass": "wine glass",
    "sofa": "couch", "couch": "couch", "table": "dining table", "desk": "dining table",
    "television": "tv", "tv": "tv", "monitor": "tv", "screen": "tv",
    "computer": "laptop", "notebook": "laptop", "laptop": "laptop",
    "bag": "backpack", "rucksack": "backpack", "purse": "handbag",
    "people": "person", "someone": "person", "somebody": "person", "anyone": "person",
    "anybody": "person", "man": "person", "woman": "person", "child": "person", "kid": "person",
    "plant": "potted plant", "bike": "bicycle", "motorbike": "motorcycle",
    "puppy": "dog", "kitten": "cat", "remote control": "remote", "tv remote": "remote",
    "fridge": "refrigerator", "teddy": "teddy bear", "hairdryer": "hair drier",
    "hair dryer": "hair drier", "seat": "chair", "stool": "chair", "armchair": "chair",
}
for _n in COCO_NAMES:
    SYNONYMS.setdefault(_n, _n)
    SYNONYMS.setdefault(_n + "s", _n)
SYNONYMS.update({"chairs": "chair", "persons": "person", "glasses": "wine glass",
                 "phones": "cell phone", "tables": "dining table"})

_VOCAB = sorted(SYNONYMS, key=len, reverse=True)
_LOCATE = re.compile(r"\b(where('s| is| are)?|find|locate|is there|are there|do you see|can you see"
                     r"|any|see (a|an|my|the))\b", re.I)
_COUNT = re.compile(r"\bhow many\b", re.I)
_DESCRIBE = re.compile(r"\b(describe|what do you see|what can you see|where am i|look around"
                       r"|what('s| is) around|surroundings)\b", re.I)
_FRONT = re.compile(r"\b(in front of me|ahead of me|ahead|in front)\b", re.I)
_TARGET = re.compile(
    r"(?:where(?:'s| is| are)?|find|locate|is there|are there|do you see|can you see|see)\s+"
    r"(?:a |an |the |my |any |some )?([a-z][a-z \-]{1,30}?)\s*"
    r"(?:\?|$| nearby| near| around| here| in (?:this|the|my) \w+| in front| on my| to my| at the)", re.I)
_YESNO = re.compile(r"^\s*(is|are|do|does|can|could|any)\b", re.I)


@dataclass
class Answer:
    question: str
    answer: str
    intent: str
    sources: list[str] = field(default_factory=list)
    grounded: bool = False  # True when backed by detector evidence
    latency_ms: float = 0.0
    vlm_used: bool = False

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("question", "answer", "intent", "sources", "grounded", "latency_ms", "vlm_used")}


def find_target(question: str) -> tuple[str | None, str | None]:
    """Return (coco_label, raw_phrase). coco_label is None for unknown objects."""
    q = question.lower()
    for phrase in _VOCAB:
        if re.search(rf"\b{re.escape(phrase)}\b", q):
            return SYNONYMS[phrase], phrase
    m = _TARGET.search(q)
    if m:
        raw = m.group(1).strip(" -")
        raw = re.sub(r"\b(please|right now|now)\b", "", raw).strip()
        return None, raw or None
    return None, None


_SIDE = re.compile(r"\b(?:on|to|at) (?:my|the) (left|right)\b", re.I)


def classify(question: str) -> str:
    if _COUNT.search(question):
        return "count"
    if _SIDE.search(question) and not find_target(question)[0]:
        return "side"
    if _DESCRIBE.search(question):
        return "describe"
    if _LOCATE.search(question):
        return "locate"
    if _FRONT.search(question):
        return "front"
    return "general"


def _where(objs: list[ObjectView]) -> str:
    return "; ".join(f"one {o.spatial.phrase()}" for o in objs[:4])


class Assistant:
    def __init__(self, analyzer: SceneAnalyzer, get_frame: Callable[[], np.ndarray | None],
                 get_objects: Callable[[], list[ObjectView]]):
        self.analyzer = analyzer
        self.get_frame = get_frame
        self.get_objects = get_objects

    def answer(self, question: str) -> Answer:
        t0 = time.perf_counter()
        question = question.strip()
        intent = classify(question)
        frame = self.get_frame()
        objects = self.get_objects() if frame is not None else []
        if frame is None:
            ans = Answer(question, "The camera is not running, so I can't see anything. "
                         "Start the camera and ask again.", intent)
        elif intent in ("locate", "count"):
            ans = self._locate(question, intent, frame, objects)
        elif intent == "side":
            ans = self._side(question, intent, frame, objects)
        elif intent == "describe":
            ans = self._open(question, intent, frame, objects, describe=True)
        else:
            ans = self._open(question, intent, frame, objects)
        ans.latency_ms = round((time.perf_counter() - t0) * 1000)
        return ans

    # -- intents -------------------------------------------------------------
    def _side(self, q: str, intent: str, frame, objects: list[ObjectView]) -> Answer:
        """'What is on my left?' - answered from measured box positions. Small VLMs
        frequently confuse left and right, so they only get a say when the detector
        has nothing on that side, and then with a caveat."""
        side = _SIDE.search(q).group(1).lower()
        there = [o for o in objects if o.spatial.horizontal == side]
        if there:
            return Answer(q, f"On your {side} I can see {summarize_objects(there, 4)}.", intent, ["detector"], True)
        base = f"The object detector doesn't see anything it recognises on your {side}."
        if self.analyzer.available and not self.analyzer.busy:
            try:
                res = self.analyzer.ask(frame, q)
                return Answer(q, f"{base} The scene model says: {naturalize(res.text)} It often mixes up left "
                              "and right, so please double-check.", intent, ["detector", "vlm"], False, vlm_used=True)
            except (VLMUnavailable, AnalyzerBusy):
                pass
        return Answer(q, base, intent, ["detector"], True)

    def _locate(self, q: str, intent: str, frame, objects: list[ObjectView]) -> Answer:
        label, raw = find_target(q)
        if label is None and raw is None:
            return self._open(q, intent, frame, objects)
        if label is None:  # not a detector class -> VLM only, with caveat
            return self._vlm_only(q, intent, frame, raw or "that")
        name = spoken_label(label)
        matches = sorted([o for o in objects if o.label == label],
                         key=lambda o: {"near": 0, "middle": 1, "far": 2}[o.spatial.distance])
        n = len(matches)
        if n:
            if intent == "count":
                text = f"I can see {n} {plural(name, n)}: {_where(matches)}."
            elif n == 1:
                o = matches[0]
                lead = "Yes. " if _YESNO.match(q) else ""
                text = f"{lead}I can see {_a(name)} {name} {o.spatial.phrase()}."
            else:
                text = f"I can see {n} {plural(name, n)}: {_where(matches)}."
            return Answer(q, text, intent, ["detector"], grounded=True)
        if intent == "count":
            return Answer(q, f"I don't see any {plural(name, 2)} right now.", intent, ["detector"], True)
        # Not detected: ask the VLM for a hedged second opinion if available.
        base = f"I don't see {_a(name)} {name} right now."
        if self.analyzer.available and not self.analyzer.busy:
            try:
                verdict, res = self.analyzer.check_presence(frame, raw or name)
                if verdict:
                    where = _strip_yes(res.text)
                    where = f" {where}" if where else ", but it couldn't say where."
                    return Answer(q, f"{base} The scene model thinks there may be one{where} "
                                  "It is less reliable, so please double-check.", intent,
                                  ["detector", "vlm"], False, vlm_used=True)
            except (VLMUnavailable, AnalyzerBusy):
                pass
        return Answer(q, base + " Try turning the camera slowly.", intent, ["detector"], grounded=True)

    def _vlm_only(self, q: str, intent: str, frame, thing: str) -> Answer:
        try:
            res = self.analyzer.ask(frame, q)
        except (VLMUnavailable, AnalyzerBusy) as e:
            return Answer(q, f"I can't recognise {thing} with the object detector, and the scene model "
                          f"is unavailable ({e}).", intent, [])
        return Answer(q, f"The object detector can't recognise that, but the scene model says: "
                      f"{naturalize(res.text) or 'I am not sure.'}", intent, ["vlm"], False, vlm_used=True)

    def _open(self, q: str, intent: str, frame, objects: list[ObjectView], describe: bool = False) -> Answer:
        if intent == "front":
            ahead = [o for o in objects if o.spatial.horizontal == "centre"]
            det = summarize_objects(ahead, 4) if ahead else ""
        else:
            det = summarize_objects(objects, 6)
        hint = f"An object detector also sees: {det}." if det else ""
        try:
            if describe:
                res = self.analyzer.describe(frame, hint_labels=sorted({spoken_label(o.label) for o in objects}))
            else:
                res = self.analyzer.ask(frame, q, hint)
            text = naturalize(res.text) or "I'm not sure."
            if intent == "front" and det:
                text = f"Directly ahead the detector sees {det}. {text}"
            return Answer(q, text, intent, ["vlm"] + (["detector"] if det else []), bool(det), vlm_used=True)
        except (VLMUnavailable, AnalyzerBusy) as e:
            why = "busy with another request" if isinstance(e, AnalyzerBusy) else "unavailable"
            if det:
                return Answer(q, f"The scene model is {why}. The object detector sees {det}.",
                              intent, ["detector"], True)
            return Answer(q, f"The scene model is {why}, and the object detector doesn't see any "
                          "objects it recognises right now.", intent, ["detector"], True)


def _a(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def naturalize(answer: str) -> str:
    """SmolVLM was trained on terse VQA answers ('Table.', 'Yes.'). Turn those
    into a sentence without changing their content."""
    t = answer.strip()
    words = re.findall(r"[A-Za-z']+", t)
    if not words or len(words) > 8:
        return t
    # Already a sentence, or an expression of uncertainty: keep it verbatim.
    if words[0].lower() in {"i", "i'm", "it", "it's", "there", "this", "that", "they", "he", "she", "the"} or \
            re.search(r"\b(not sure|don't know|unsure|unclear|cannot|can't)\b", t, re.I):
        return t
    if len(words) > 3:
        # Noun-phrase answers without a verb, e.g. 'Book and plate with fruits.'
        verbs = {"is", "are", "was", "were", "has", "have", "can", "see", "looks", "appears", "there", "it's"}
        if any(w.lower() in verbs for w in words):
            return t
    core = t.rstrip(".!? ").strip()
    if core.lower() in ("yes", "no"):
        return f"{core.capitalize()}, according to the scene model."
    if len(words) == 1 and not core.lower().endswith("s"):
        return f"It looks like {_a(core)} {core.lower()}."
    return f"It looks like {core[:1].lower() + core[1:]}."


def _strip_yes(text: str) -> str:
    """'Yes, on the left of the desk.' -> '- on the left of the desk.'; bare 'Yes.' -> ''."""
    t = re.sub(r"^\s*yes[,.!]?\s*", "", text, flags=re.I).strip()
    return f"- {t[:1].lower() + t[1:]}" if t else ""
