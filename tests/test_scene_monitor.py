import numpy as np

from backend.assistant.announcer import Announcer, summarize_objects
from backend.assistant.scene_monitor import ObjectView, SceneMonitor
from backend.config.settings import Settings
from backend.spatial.positioning import analyze

W, H = 640, 480


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def obj(tid, label, box):
    box = np.array(box, float)
    return ObjectView(tid, label, 0.9, [box[0] / W, box[1] / H, box[2] / W, box[3] / H],
                      analyze(box, W, H, label), 1, 0.0)


def run(mon, clock, frames, dt=0.2):
    events = []
    for objs in frames:
        clock.t += dt
        events += mon.update(objs)
    return events


def make(**kw):
    s = Settings(**kw)
    clock = Clock()
    return SceneMonitor(lambda: s, clock), clock, s


def test_enter_once_after_min_frames_and_not_repeated():
    mon, clock, _ = make(min_track_frames=3)
    chair = [10, 200, 120, 330]
    ev = run(mon, clock, [[obj(1, "chair", chair)]] * 2)
    assert ev == []
    ev = run(mon, clock, [[obj(1, "chair", chair)]] * 30)
    assert [e.kind for e in ev] == ["entered"]
    assert ev[0].message == "Chair on your left, middle distance."


def test_leave_after_timeout():
    mon, clock, _ = make(min_track_frames=1, leave_after_s=1.0)
    run(mon, clock, [[obj(1, "cup", [300, 300, 340, 360])]] * 2)
    ev = run(mon, clock, [[]] * 3, dt=0.2)
    assert ev == []
    ev = run(mon, clock, [[]] * 3, dt=0.3)
    assert [e.kind for e in ev] == ["left"]


def test_id_switch_is_not_announced_twice():
    mon, clock, _ = make(min_track_frames=2, leave_after_s=1.0)
    run(mon, clock, [[obj(1, "bottle", [300, 300, 330, 380])]] * 3)
    # tracker loses id 1 and re-acquires the same bottle as id 7
    ev = run(mon, clock, [[obj(7, "bottle", [305, 300, 335, 380])]] * 10)
    assert "entered" not in [e.kind for e in ev]


def test_obstacle_warning_on_transition_with_cooldown():
    mon, clock, _ = make(min_track_frames=2)
    far = [300, 200, 340, 260]
    near = [200, 60, 440, 479]
    ev = run(mon, clock, [[obj(1, "chair", far)]] * 3)
    assert [e.kind for e in ev] == ["entered"]
    ev = run(mon, clock, [[obj(1, "chair", near)]] * 5)
    kinds = [e.kind for e in ev]
    assert kinds.count("obstacle") + kinds.count("approaching") == 1
    assert ev[0].priority == "warning"
    # stays near: no repeated warnings
    ev = run(mon, clock, [[obj(1, "chair", near)]] * 20)
    assert ev == []


def test_approaching_object_detected():
    mon, clock, _ = make(min_track_frames=1)
    frames = []
    for i in range(12):  # person walking towards the camera in the centre
        hh = 80 + 25 * i
        frames.append([obj(1, "person", [320 - hh * 0.2, 240 - hh / 2, 320 + hh * 0.2, 240 + hh / 2])])
    ev = run(mon, clock, frames, dt=0.25)
    assert any(e.kind in ("approaching", "obstacle") for e in ev)


def test_scene_change_detection():
    mon, clock, _ = make()
    dark = np.zeros((H, W, 3), np.uint8)
    bright = np.full((H, W, 3), 200, np.uint8)
    for _ in range(5):
        clock.t += 0.2
        assert mon.update([], dark) == []
    kinds = []
    for _ in range(3):
        clock.t += 0.2
        kinds += [e.kind for e in mon.update([], bright)]
    assert kinds == ["scene_change"]


class FakeSpeech:
    def __init__(self):
        self.said = []

    def say(self, text, priority=1, kind="info", max_age=None):
        self.said.append((text, int(priority)))
        return True


def test_announcer_groups_and_respects_cooldown():
    s = Settings(announce_cooldown_s=20)
    clock = Clock()
    sp = FakeSpeech()
    ann = Announcer(lambda: s, sp, clock)
    mon, mclock, _ = make(min_track_frames=1)
    ev = run(mon, mclock, [[obj(1, "chair", [500, 200, 620, 330]), obj(2, "bottle", [300, 300, 330, 380])]])
    ann.handle(ev)
    assert sp.said[0][0] == "New: chair on your right, bottle ahead."
    # another chair 5 s later is within the cooldown -> silent
    clock.t += 5
    mon2, c2, _ = make(min_track_frames=1)
    ann.handle(run(mon2, c2, [[obj(9, "chair", [0, 200, 100, 330])]]))
    assert len(sp.said) == 1


def test_announcer_muted_and_urgent_priority():
    s = Settings(announcements_muted=True)
    sp = FakeSpeech()
    mon, clock, _ = make(min_track_frames=1)
    ev = run(mon, clock, [[obj(1, "chair", [200, 60, 440, 479])]])
    Announcer(lambda: s, sp).handle(ev)
    assert sp.said == []
    s2 = Settings()
    Announcer(lambda: s2, sp).handle(ev)
    assert sp.said[0][0].startswith("Caution: chair close") and sp.said[0][1] == 3


def test_summary_text():
    objs = [obj(1, "chair", [500, 200, 620, 330]), obj(2, "chair", [520, 210, 630, 330]),
            obj(3, "laptop", [250, 150, 420, 380])]
    text = summarize_objects(objs)
    assert "a laptop ahead, near" in text
    assert "2 chairs on your right" in text
