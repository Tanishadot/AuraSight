import os
import sys
import threading
import time
import wave

import pytest

from backend.config.settings import Settings
from backend.speech.service import Priority, SpeechService
from tests.conftest import RecordingEngine, wait_for


def make(engine, **settings):
    s = Settings(**settings)
    svc = SpeechService(lambda: s, engine_factory=lambda: engine)
    svc.start()
    return svc, s


def test_speaks_in_order_without_overlap():
    eng = RecordingEngine(0.05)
    svc, _ = make(eng)
    for t in ("one", "two", "three"):
        svc.say(t)
    assert wait_for(lambda: len(eng.spoken) == 3, 3)
    assert eng.spoken == ["one", "two", "three"]
    svc.shutdown()


def test_priority_order_and_urgent_interrupts():
    eng = RecordingEngine(0.4)
    svc, _ = make(eng)
    svc.say("long low priority sentence", Priority.LOW)
    assert wait_for(lambda: svc.is_speaking, 2)
    svc.say("normal", Priority.NORMAL)
    svc.say("Caution: chair close, ahead.", Priority.URGENT)
    assert wait_for(lambda: len(eng.spoken) == 2, 4)
    assert eng.interrupted == ["long low priority sentence"]
    assert eng.spoken == ["Caution: chair close, ahead.", "normal"]
    svc.shutdown()


def test_duplicates_are_not_queued():
    eng = RecordingEngine(0.3)
    svc, _ = make(eng)
    assert svc.say("same")
    wait_for(lambda: svc.is_speaking, 2)
    assert not svc.say("same")
    assert svc.say("other")
    assert not svc.say("other")
    svc.shutdown()


def test_stale_items_are_dropped():
    eng = RecordingEngine(0.5)
    svc, _ = make(eng)
    svc.say("blocking", Priority.USER)
    wait_for(lambda: svc.is_speaking, 2)
    svc.say("stale info", Priority.NORMAL, max_age=0.1)
    assert wait_for(lambda: not svc.is_speaking and len(eng.spoken) >= 1, 3)
    time.sleep(0.2)
    assert "stale info" not in eng.spoken
    svc.shutdown()


def test_queue_is_bounded():
    eng = RecordingEngine(1.0)
    svc, _ = make(eng)
    svc.say("busy", Priority.USER)
    wait_for(lambda: svc.is_speaking, 2)
    for i in range(20):
        svc.say(f"item {i}", Priority.LOW)
    assert svc.status()["queued"] <= 10
    svc.stop_all()
    svc.shutdown()


def test_stop_all_silences_and_clears():
    eng = RecordingEngine(1.0)
    svc, _ = make(eng)
    svc.say("a long announcement")
    svc.say("next")
    wait_for(lambda: svc.is_speaking, 2)
    svc.stop_all()
    assert wait_for(lambda: not svc.is_speaking, 2)
    time.sleep(0.2)
    assert eng.spoken == [] and eng.interrupted == ["a long announcement"]
    svc.shutdown()


def test_cancel_kind_only_silences_that_kind():
    eng = RecordingEngine(0.6)
    svc, _ = make(eng)
    svc.say("Sentence one of the description.", Priority.USER, kind="description")
    svc.say("Sentence two of the description.", Priority.USER, kind="description")
    svc.say("Your answer.", Priority.USER, kind="answer")
    assert wait_for(lambda: svc.current_kind == "description" and svc.is_speaking, 2)
    assert svc.cancel("description") == 2  # the playing one + the queued one
    assert wait_for(lambda: eng.spoken == ["Your answer."], 3), eng.spoken
    assert eng.interrupted == ["Sentence one of the description."]
    svc.shutdown()


def test_description_cut_off_by_warning_is_read_again_afterwards():
    eng = RecordingEngine(0.4)
    svc, _ = make(eng)
    svc.say("A woman sits at a desk.", Priority.USER, kind="description")
    svc.say("A laptop is on your right.", Priority.USER, kind="description")
    assert wait_for(lambda: svc.is_speaking, 2)
    svc.say("Caution: chair close, ahead.", Priority.URGENT, kind="alert")
    assert wait_for(lambda: len(eng.spoken) == 3, 4), eng.spoken
    assert eng.spoken == ["Caution: chair close, ahead.", "A woman sits at a desk.", "A laptop is on your right."]
    svc.shutdown()


def test_replay_bypasses_duplicate_filter():
    eng = RecordingEngine(0.3)
    svc, _ = make(eng)
    svc.say("A laptop is on your right.", kind="description")
    wait_for(lambda: svc.is_speaking, 2)
    assert svc.say("A laptop is on your right.", kind="description", dedupe=False)
    assert wait_for(lambda: len(eng.spoken) == 2, 3)
    svc.shutdown()


def test_speak_error_is_reported_and_worker_survives():
    eng = RecordingEngine(0.01)
    events = []
    s = Settings()
    svc = SpeechService(lambda: s, engine_factory=lambda: eng, on_event=lambda t, d: events.append(d))
    svc.start()
    eng.fail = True
    svc.say("broken", kind="description")
    assert wait_for(lambda: any(e.get("state") == "error" for e in events), 2)
    eng.fail = False
    svc.say("works again")
    assert wait_for(lambda: eng.spoken == ["works again"], 2)
    svc.shutdown()


def test_disabled_speech_drops_everything():
    eng = RecordingEngine()
    svc, _ = make(eng, speech_enabled=False)
    assert svc.say("hello") is False
    svc.shutdown()


def test_engine_failure_reports_error():
    def broken():
        raise RuntimeError("no audio")

    s = Settings()
    svc = SpeechService(lambda: s, engine_factory=broken)
    svc.start()
    assert svc.state == "error" and "no audio" in svc.error


def test_mentioned_in_matches_synonyms_only():
    from backend.assistant.announcer import mentioned_in

    text = "A woman sits at a desk. A computer screen glows on your right. There is a sofa."
    assert mentioned_in("person", text)
    assert mentioned_in("tv", text)
    assert mentioned_in("laptop", text)
    assert mentioned_in("couch", text)
    assert mentioned_in("dining table", text)
    assert not mentioned_in("cup", text)
    assert not mentioned_in("chair", text)
    assert not mentioned_in("teddy bear", "A bear statue.")  # multi-word needs the phrase
    assert not mentioned_in("cat", "A category label.")  # whole words only


def test_announcer_holds_routine_speech_during_description_but_not_warnings():
    from backend.assistant.announcer import Announcer
    from backend.assistant.scene_monitor import SceneEvent

    said = []

    class Spy:
        def say(self, text, priority=Priority.NORMAL, kind="info", max_age=None, dedupe=True):
            said.append(kind)
            return True

    hold = {"on": True}
    a = Announcer(lambda: Settings(), Spy(), hold=lambda: hold["on"])
    new = SceneEvent(kind="entered", message="chair", priority="info", label="chair")
    warn = SceneEvent(kind="obstacle", message="Caution: chair close.", priority="warning", label="person")
    a.handle([new, warn])
    assert said == ["alert"]  # obstacle warnings still interrupt
    hold["on"] = False
    a.handle([SceneEvent(kind="entered", message="cup", priority="info", label="cup")])
    assert said == ["alert", "new_objects"]


# ---- real offline engine (Windows SAPI 5) --------------------------------------
@pytest.mark.skipif(sys.platform != "win32", reason="SAPI is Windows-only")
def test_real_sapi_synthesizes_wav(tmp_path):
    from backend.speech.engines import SapiEngine

    out = {}

    def work():
        eng = SapiEngine()
        p = tmp_path / "chair.wav"
        eng.synthesize_to_file("Chair on your right, middle distance.", p, 1.0, 0.9, None)
        with wave.open(str(p)) as w:
            out["dur"] = w.getnframes() / w.getframerate()
        out["voices"] = eng.voices()

    t = threading.Thread(target=work)
    t.start()
    t.join(30)
    assert out["dur"] > 0.8
    assert len(out["voices"]) >= 1


@pytest.mark.skipif(sys.platform != "win32" or os.environ.get("AURASIGHT_SKIP_AUDIO") == "1",
                    reason="plays audio; Windows only")
def test_real_sapi_playback_and_interrupt():
    """Plays a short phrase through the speakers (volume 30%) and interrupts it."""
    from backend.speech.engines import create_engine

    out = {}

    def work():
        eng = create_engine()
        out["name"] = eng.name
        threading.Timer(0.5, eng.interrupt).start()
        t0 = time.perf_counter()
        out["completed"] = eng.speak("AuraSight is testing an interruptible sentence that keeps going "
                                     "for quite a while longer than half a second.", 1.0, 0.3, None)
        out["elapsed"] = time.perf_counter() - t0

    t = threading.Thread(target=work)
    t.start()
    t.join(20)
    assert out["name"] in ("sapi5", "pyttsx3", "piper")
    assert out["completed"] is False
    assert out["elapsed"] < 1.5
