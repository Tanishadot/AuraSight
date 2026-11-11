import numpy as np
import pytest

from backend.assistant.qa import Assistant, classify, find_target
from backend.assistant.scene_monitor import ObjectView
from backend.spatial.positioning import analyze
from backend.vision.analyzer import AnalysisResult, AnalyzerBusy
from backend.vision.backends import VLMUnavailable

W, H = 640, 480
FRAME = np.zeros((H, W, 3), np.uint8)


def obj(tid, label, box):
    box = np.array(box, float)
    return ObjectView(tid, label, 0.8, [box[0] / W, box[1] / H, box[2] / W, box[3] / H],
                      analyze(box, W, H, label), 5, 0.0)


class FakeAnalyzer:
    """Records calls; behaves like an available or unavailable VLM."""

    def __init__(self, available=True, text="A wooden desk with a laptop.", presence=None):
        self.available = available
        self.busy = False
        self.text = text
        self.presence = presence
        self.calls = []

    def _res(self, text):
        return AnalysisResult(text, 1000, "fake", text)

    def describe(self, frame, hint_labels=None, on_sentence=None):
        self.calls.append(("describe",))
        if not self.available:
            raise VLMUnavailable("off")
        return self._res(self.text)

    def ask(self, frame, q, hint=""):
        self.calls.append(("ask", q, hint))
        if not self.available:
            raise VLMUnavailable("off")
        return self._res(self.text)

    def check_presence(self, frame, thing):
        self.calls.append(("presence", thing))
        if not self.available:
            raise VLMUnavailable("off")
        return self.presence, self._res("Yes, on the left of the desk.")


@pytest.mark.parametrize("q,label", [
    ("Where is my water bottle?", "bottle"),
    ("Is there a chair nearby?", "chair"),
    ("Can you see my phone", "cell phone"),
    ("is anyone here?", "person"),
    ("where is the sofa", "couch"),
])
def test_find_target_known(q, label):
    assert find_target(q)[0] == label


def test_find_target_unknown_object():
    assert find_target("Where are my keys?") == (None, "keys")
    assert find_target("Is there a door in this room?") == (None, "door")
    assert find_target("Is there a window in front of me?") == (None, "window")


@pytest.mark.parametrize("q,intent", [
    ("What is in front of me?", "front"),
    ("Describe this room.", "describe"),
    ("Where is my water bottle?", "locate"),
    ("Is there a chair nearby?", "locate"),
    ("How many people are here?", "count"),
    ("What objects are on the table?", "general"),
])
def test_classify(q, intent):
    assert classify(q) == intent


def make(objects, analyzer=None, frame=FRAME):
    return Assistant(analyzer or FakeAnalyzer(), lambda: frame, lambda: objects)


def test_locate_grounded_in_detector():
    an = FakeAnalyzer()
    a = make([obj(1, "bottle", [280, 300, 330, 420])], an).answer("Where is my water bottle?")
    assert a.grounded and a.sources == ["detector"]
    assert a.answer == "I can see a bottle ahead, near."  # no "Yes." for a where-question
    assert an.calls == []  # VLM not needed
    a = make([obj(1, "bottle", [280, 300, 330, 420])], an).answer("Is there a bottle?")
    assert a.answer == "Yes. I can see a bottle ahead, near."


def test_locate_multiple_and_count():
    objs = [obj(1, "chair", [0, 200, 100, 330]), obj(2, "chair", [520, 150, 640, 400])]
    a = make(objs).answer("Is there a chair nearby?")
    assert a.answer.startswith("I can see 2 chairs:")
    a = make(objs).answer("How many chairs are there?")
    assert a.answer.startswith("I can see 2 chairs")
    a = make([]).answer("How many people are here?")
    assert a.answer == "I don't see any people right now."


def test_not_detected_does_not_invent():
    an = FakeAnalyzer(presence=False)
    a = make([], an).answer("Where is my water bottle?")
    assert a.answer.startswith("I don't see a bottle right now.")
    assert "Try turning the camera" in a.answer


def test_not_detected_vlm_second_opinion_is_hedged():
    an = FakeAnalyzer(presence=True)
    a = make([], an).answer("Is there a cup?")
    assert "less reliable" in a.answer and not a.grounded and a.vlm_used


def test_unknown_object_uses_vlm_with_caveat():
    an = FakeAnalyzer(text="I'm not sure.")
    a = make([], an).answer("Where are my keys?")
    assert a.answer.startswith("The object detector can't recognise that, but the scene model says:")
    assert "I'm not sure" in a.answer


def test_open_question_includes_detector_hint():
    an = FakeAnalyzer(text="There is a laptop on the table.")
    a = make([obj(1, "laptop", [250, 150, 420, 380])], an).answer("What objects are on the table?")
    assert a.answer == "There is a laptop on the table."
    assert "laptop ahead" in an.calls[0][2]


def test_front_combines_detector_and_vlm():
    an = FakeAnalyzer(text="A desk with a laptop.")
    a = make([obj(1, "laptop", [250, 150, 420, 380]), obj(2, "chair", [0, 200, 100, 330])],
             an).answer("What is in front of me?")
    assert a.answer.startswith("Directly ahead the detector sees a laptop ahead, near.")
    assert "chair" not in a.answer.split(".")[0]


def test_vlm_unavailable_falls_back_to_detector():
    a = make([obj(1, "laptop", [250, 150, 420, 380])], FakeAnalyzer(available=False)).answer("Describe this room")
    assert a.answer.startswith("The scene model is unavailable. The object detector sees a laptop")
    a = make([], FakeAnalyzer(available=False)).answer("Describe this room")
    assert "doesn't see any objects" in a.answer


def test_camera_off():
    a = make([], frame=None).answer("What is in front of me?")
    assert "camera is not running" in a.answer


def test_vlm_second_opinion_without_location():
    an = FakeAnalyzer(presence=True)
    an.check_presence = lambda frame, thing: (True, an._res("Yes."))
    a = make([], an).answer("Is there a cup?")
    assert "may be one, but it couldn't say where." in a.answer


@pytest.mark.parametrize("raw,expected", [
    ("Table.", "It looks like a table."),
    ("Yes.", "Yes, according to the scene model."),
    ("Book and plate with fruits.", "It looks like book and plate with fruits."),
    ("There are chairs and tables in front of me.", "There are chairs and tables in front of me."),
])
def test_naturalize_terse_vqa_answers(raw, expected):
    from backend.assistant.qa import naturalize

    assert naturalize(raw) == expected


def test_side_questions_use_measured_positions():
    """SmolVLM answered 'What is on my left?' with the laptop that was on the
    right; side questions must come from detector geometry."""
    an = FakeAnalyzer(text="A laptop.")
    objs = [obj(1, "laptop", [420, 150, 600, 380]), obj(2, "chair", [0, 200, 110, 330])]
    a = make(objs, an).answer("What is on my left?")
    assert a.intent == "side" and a.grounded and a.sources == ["detector"]
    assert a.answer.startswith("On your left I can see a chair") and "laptop" not in a.answer
    assert an.calls == []
    a = make([obj(1, "laptop", [420, 150, 600, 380])], an).answer("What is on my left?")
    assert "doesn't see anything it recognises on your left" in a.answer and "double-check" in a.answer
