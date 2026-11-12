"""Generate golden test vectors for the Dart port (mobile/packages/aura_core).

Runs the *desktop* Python implementations - the tested source of truth - on
deterministic scenarios and writes their exact outputs as JSON. The Dart tests
must reproduce them, so the two implementations cannot drift apart silently.

Run from the repository root with the desktop virtualenv:
    .venv/Scripts/python mobile/tools/gen_golden.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.assistant.announcer import Announcer, mentioned_in, summarize_objects  # noqa: E402
from backend.assistant.scene_monitor import ObjectView, SceneEvent, SceneMonitor  # noqa: E402
from backend.config.settings import Settings  # noqa: E402
from backend.detection.verify import ObjectVerifier  # noqa: E402
from backend.spatial.positioning import analyze, plural, spoken_label  # noqa: E402
from backend.speech.service import Priority  # noqa: E402
from backend.tracking.bytetrack import ByteTracker, TrackOutput  # noqa: E402

OUT = ROOT / "mobile" / "packages" / "aura_core" / "test" / "golden"
W, H = 640, 480
NAMES = ["person", "chair", "tv", "cup", "laptop", "refrigerator", "dining table", "bottle"]


def f(x) -> float:
    return float(x)


def box_list(b) -> list[float]:
    return [f(v) for v in b]


# --------------------------------------------------------------------------- tracker
def tracker_scenarios() -> list[dict]:
    rng = np.random.RandomState(7)
    scen = []

    def linear(n_frames=30):
        frames = []
        for t in range(n_frames):
            frames.append({"boxes": [[10 + 3 * t, 50, 60 + 3 * t, 150], [400 - 2 * t, 60, 460 - 2 * t, 200]],
                           "scores": [0.9, 0.8], "classes": [0, 1]})
        return frames

    def low_score_dips():
        frames = []
        for t in range(25):
            s = 0.25 if t % 5 == 3 else 0.85
            frames.append({"boxes": [[100 + 2 * t, 100, 170 + 2 * t, 260]], "scores": [s], "classes": [0]})
        return frames

    def occlusion():
        frames = []
        for t in range(40):
            hidden = 8 <= t < 13  # short gap: same id
            gone = 20 <= t < 40 and t < 32  # long gap: new id
            b = [[120 + t, 100, 190 + t, 260]]
            if hidden or gone:
                frames.append({"boxes": [], "scores": [], "classes": []})
            else:
                frames.append({"boxes": b, "scores": [0.9], "classes": [1]})
        return frames

    def crossing():
        frames = []
        for t in range(30):
            frames.append({"boxes": [[50 + 15 * t, 200, 130 + 15 * t, 330], [500 - 15 * t, 210, 580 - 15 * t, 340]],
                           "scores": [0.9, 0.88], "classes": [0, 3]})
        return frames

    def clutter():
        objs = [{"x": rng.uniform(40, 560), "y": rng.uniform(40, 380), "w": rng.uniform(30, 120),
                 "h": rng.uniform(40, 160), "vx": rng.uniform(-4, 4), "vy": rng.uniform(-3, 3),
                 "cls": int(rng.randint(0, len(NAMES)))} for _ in range(6)]
        frames = []
        for t in range(60):
            boxes, scores, classes = [], [], []
            for i, o in enumerate(objs):
                o["x"] += o["vx"]
                o["y"] += o["vy"]
                if rng.rand() < 0.08:
                    continue  # missed detection
                j = rng.normal(0, 1.5, 4)
                boxes.append([o["x"] + j[0], o["y"] + j[1], o["x"] + o["w"] + j[2], o["y"] + o["h"] + j[3]])
                scores.append(float(np.clip(rng.normal(0.7, 0.18), 0.05, 0.99)))
                classes.append(o["cls"])
            if rng.rand() < 0.3:  # spurious detection
                x, y = rng.uniform(0, 600), rng.uniform(0, 440)
                boxes.append([x, y, x + 30, y + 30])
                scores.append(float(rng.uniform(0.1, 0.6)))
                classes.append(int(rng.randint(0, len(NAMES))))
            frames.append({"boxes": boxes, "scores": scores, "classes": classes})
        return frames

    for name, frames, kw in [("linear", linear(), {}), ("low_score_dips", low_score_dips(), {}),
                             ("occlusion", occlusion(), {"max_lost_frames": 10}), ("crossing", crossing(), {}),
                             ("clutter", clutter(), {"max_lost_frames": 15})]:
        tr = ByteTracker(**kw)
        out_frames = []
        for fr in frames:
            # the tracker stores boxes/scores as float32; give Dart those exact values
            boxes = np.asarray(fr["boxes"], np.float32).reshape(-1, 4)
            scores = np.asarray(fr["scores"], np.float32).reshape(-1)
            fr["boxes"] = [box_list(b) for b in boxes]
            fr["scores"] = [f(s) for s in scores]
            outs = tr.update(boxes, scores, np.asarray(fr["classes"]).reshape(-1))
            out_frames.append([{"id": o.track_id, "box": box_list(o.box), "score": f(o.score), "cls": o.cls,
                                "hits": o.hits, "age": o.age_frames} for o in outs])
        scen.append({"name": name, "params": kw, "frames": frames, "expected": out_frames})
    return scen


# --------------------------------------------------------------------------- spatial
def spatial_cases() -> list[dict]:
    rng = np.random.RandomState(3)
    cases = []
    labels = ["person", "chair", "cup", "tv", "dog", "bottle", "dining table", "laptop", "unknownthing"]
    for _ in range(300):
        x1, y1 = rng.uniform(-20, 600), rng.uniform(-20, 440)
        w, h = rng.uniform(5, 640), rng.uniform(5, 480)
        box = [x1, y1, x1 + w, y1 + h]
        label = labels[rng.randint(0, len(labels))]
        s = analyze(np.array(box), W, H, label)
        cases.append({"box": box, "label": label, "horizontal": s.horizontal, "direction": s.direction,
                      "distance": s.distance, "in_path": s.in_path, "cx": s.cx, "area": s.area,
                      "phrase": s.phrase()})
    # exact zone boundaries
    for cx in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        box = [cx * W - 10, 100, cx * W + 10, 200]
        s = analyze(np.array(box), W, H, "cup")
        cases.append({"box": box, "label": "cup", "horizontal": s.horizontal, "direction": s.direction,
                      "distance": s.distance, "in_path": s.in_path, "cx": s.cx, "area": s.area,
                      "phrase": s.phrase()})
    words = ["person", "mouse", "chair", "tv", "dining table", "cell phone", "glass", "bus", "cup"]
    return {"analyze": cases,
            "plural": [{"label": w, "n": n, "out": plural(w, n)} for w in words for n in (1, 2)],
            "spoken": [{"label": w, "out": spoken_label(w)} for w in words + ["couch", "potted plant"]]}


# --------------------------------------------------------------------------- verifier
class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


def verifier_scenarios() -> list[dict]:
    names = NAMES
    scen = []
    # (name, per-frame list of (track_id, cls, score)), crop scores per (track_id, attempt)
    def steady(tid, cls, score, n=20):
        return [[(tid, cls, score)] for _ in range(n)]

    rng = np.random.RandomState(11)
    defs = [
        ("plain_confirm", steady(1, 0, 0.8), {}),
        ("low_confidence_never", steady(1, 1, 0.4), {}),
        ("flicker", [[(1, 0, 0.9)] if t % 3 == 0 else [] for t in range(30)], {}),
        ("tv_strong", steady(1, 2, 0.75), {}),
        ("tv_verified", steady(1, 2, 0.6), {1: [0.7]}),
        ("tv_two_pass", steady(1, 2, 0.6, 30), {1: [0.5, 0.42]}),
        ("tv_rejected_fast", steady(1, 2, 0.6, 30), {1: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]}),
        ("tv_rejected_two", steady(1, 2, 0.58, 40), {1: [0.3, 0.35, 0.3, 0.3, 0.3, 0.3]}),
        ("fridge_demoted", [[(1, 5, 0.75)]] * 10 + [[(1, 5, 0.3)]] * 15, {}),
        ("mixed", [[(1, 0, float(rng.uniform(0.5, 0.9))), (2, 2, float(rng.uniform(0.45, 0.65))),
                    (3, 3, float(rng.uniform(0.2, 0.6)))] for _ in range(40)], {2: [0.3, 0.5, 0.6, 0.2, 0.7, 0.4]}),
    ]
    class TableVerifier(ObjectVerifier):
        """Crop re-detection replaced by a fixed table of scores per track,
        so Dart can be fed exactly the same values."""

        def __init__(self, table, *a, **kw):
            super().__init__(*a, **kw)
            self.table, self.calls, self.current = table, {}, None

        def _evaluate(self, t, st, s, frame, now):
            self.current = t.track_id
            return super()._evaluate(t, st, s, frame, now)

        def crop_verify(self, frame, box, cls):
            i = self.calls.get(self.current, 0)
            self.calls[self.current] = i + 1
            seq = self.table.get(self.current, [0.0])
            return seq[min(i, len(seq) - 1)]

    for name, frames, crop in defs:
        s = Settings()
        clock = Clock()
        ver = TableVerifier(crop, lambda: s, crop_detect=lambda img, size: (None, None, None),
                            names=names, clock=clock)
        exp = []
        frame_img = np.zeros((H, W, 3), np.uint8)
        for fr in frames:
            clock.t += 0.2
            tracks = [TrackOutput(tid, np.array([100, 100, 300, 260], np.float32), sc, cls, 5, 5)
                      for tid, cls, sc in fr]
            states = ver.update(tracks, frame_img)
            exp.append({str(tid): {"status": st.status, "reason": st.reason} for tid, st in states.items()})
        scen.append({"name": name, "frames": [[{"id": t, "cls": c, "score": sc} for t, c, sc in fr] for fr in frames],
                     "crop_scores": {str(k): v for k, v in crop.items()}, "expected": exp})
    return scen


# --------------------------------------------------------------------------- scene monitor
def view(tid, label, box, conf=0.9):
    box = np.array(box, float)
    return ObjectView(tid, label, conf, [box[0] / W, box[1] / H, box[2] / W, box[3] / H],
                      analyze(box, W, H, label), 1, 0.0)


def ev_dict(e: SceneEvent) -> dict:
    return {"kind": e.kind, "message": e.message, "priority": e.priority, "label": e.label, "track_id": e.track_id}


def monitor_scenarios() -> list[dict]:
    scen = []
    rng = np.random.RandomState(5)

    def mk(frames_spec, settings=None, thumbs=None, dt=0.2):
        s = Settings(**(settings or {}))
        clock = Clock()
        mon = SceneMonitor(lambda: s, clock)
        out = []
        for i, objs in enumerate(frames_spec):
            clock.t += dt
            views = [view(*o) for o in objs]
            frame = None
            if thumbs is not None:
                g = np.asarray(thumbs[i], np.uint8)
                frame = np.repeat(g[:, :, None], 3, axis=2)
            evs = mon.update(views, frame)
            out.append({"events": [ev_dict(e) for e in evs], "approaching": [v.approaching for v in views]})
        return out

    # 1. enter once, then leave
    f1 = [[(1, "chair", [40, 200, 140, 330])]] * 10 + [[]] * 15
    scen.append({"name": "enter_leave", "settings": {"min_track_frames": 3}, "frames": f1})
    # 2. obstacle on entry (near + in path)
    f2 = [[(1, "person", [220, 20, 420, 478])]] * 8
    scen.append({"name": "obstacle_entry", "settings": {}, "frames": f2})
    # 3. approaching person grows quickly in path
    f3 = []
    for t in range(25):
        k = 1 + 0.12 * t
        cx, cy, w, h = 320, 240, 40 * k, 90 * k
        f3.append([(1, "person", [cx - w / 2, cy - h / 2, cx + w / 2, min(cy + h / 2, 470)])])
    scen.append({"name": "approaching", "settings": {}, "frames": f3})
    # 4. re-identification: lost id 1, new id 2 same class nearby within window
    f4 = [[(1, "cup", [300, 300, 340, 350])]] * 6 + [[]] * 3 + [[(2, "cup", [305, 300, 345, 350])]] * 6
    scen.append({"name": "reid", "settings": {"min_track_frames": 3}, "frames": f4})
    # 5. moves into near path later -> obstacle transition
    f5 = [[(1, "chair", [20, 300, 120, 420])]] * 6 + [[(1, "chair", [200, 150, 450, 478])]] * 6
    scen.append({"name": "obstacle_transition", "settings": {}, "frames": f5})
    for sc in scen:
        sc["expected"] = mk(sc["frames"], sc["settings"])
        sc["frames"] = [[{"id": o[0], "label": o[1], "box": [f(v) for v in o[2]]} for o in fr] for fr in sc["frames"]]
    # 6. scene change from thumbnails (24x32 grey)
    base = rng.randint(60, 120, (24, 32))
    thumbs = [base + rng.randint(-3, 4, (24, 32)) for _ in range(5)]
    other = rng.randint(150, 250, (24, 32))
    thumbs += [other + rng.randint(-3, 4, (24, 32)) for _ in range(5)]
    thumbs = [np.clip(t, 0, 255).astype(int).tolist() for t in thumbs]
    sc = {"name": "scene_change", "settings": {}, "frames": [[] for _ in thumbs], "thumbs": thumbs}
    sc["expected"] = mk(sc["frames"], {}, thumbs)
    scen.append(sc)
    return scen


# --------------------------------------------------------------------------- announcer
def announcer_cases() -> dict:
    class Spy:
        def __init__(self):
            self.said = []

        def say(self, text, priority=Priority.NORMAL, kind="info", max_age=None, dedupe=True):
            self.said.append({"text": text, "priority": int(priority), "kind": kind, "max_age": max_age})
            return True

    def sp(label, box):
        return analyze(np.array(box, float), W, H, label)

    E = SceneEvent
    batches = [
        [E("entered", "x", "info", "chair", 1, sp("chair", [20, 200, 120, 330]))],
        [E("entered", "x", "info", "cup", 2, sp("cup", [300, 300, 340, 350])),
         E("entered", "x", "info", "laptop", 3, sp("laptop", [500, 200, 620, 300])),
         E("entered", "x", "info", "bottle", 4, sp("bottle", [10, 10, 30, 60])),
         E("entered", "x", "info", "tv", 5, sp("tv", [200, 50, 400, 200]))],
        [E("entered", "x", "info", "chair", 6, sp("chair", [400, 200, 500, 330]))],  # cooldown
        [E("obstacle", "Caution: person close, ahead.", "warning", "person", 7, None),
         E("approaching", "Dog approaching, slightly left.", "warning", "dog", 8, None),
         E("obstacle", "Caution: chair close, ahead.", "warning", "chair", 9, None)],
        [E("left", "x", "info", "cup", 2, None), E("left", "x", "info", "laptop", 3, None)],
        [E("scene_change", "x", "info")],
    ]
    runs = []
    for settings in ({}, {"verbosity": "detailed", "announce_departures": True},
                     {"announcements_muted": True}, {"obstacle_alerts": False, "announce_scene_changes": False}):
        s = Settings(**settings)
        clock = Clock()
        spy = Spy()
        a = Announcer(lambda: s, spy, clock)
        out = []
        for b in batches:
            clock.t += 2.0
            out.append(a.handle(b))
        runs.append({"settings": settings, "said": spy.said, "returned": out})

    def evj(e):
        d = {"kind": e.kind, "message": e.message, "priority": e.priority, "label": e.label, "track_id": e.track_id}
        if e.spatial:
            d["box"] = None
        return d

    batch_json = []
    for b in batches:
        items = []
        for e in b:
            d = {"kind": e.kind, "message": e.message, "priority": e.priority, "label": e.label, "track_id": e.track_id}
            if e.spatial is not None:
                d["spatial"] = {"horizontal": e.spatial.horizontal, "direction": e.spatial.direction,
                                "distance": e.spatial.distance, "in_path": e.spatial.in_path,
                                "cx": e.spatial.cx, "area": e.spatial.area}
            items.append(d)
        batch_json.append(items)

    objs = [view(1, "chair", [20, 200, 120, 330]), view(2, "chair", [40, 210, 130, 340]),
            view(3, "cup", [300, 300, 340, 350]), view(4, "person", [220, 20, 420, 478]),
            view(5, "tv", [500, 50, 630, 200]), view(6, "laptop", [250, 250, 400, 360]),
            view(7, "bottle", [600, 10, 630, 60]), view(8, "dining table", [100, 300, 500, 478])]
    summaries = []
    for n in range(0, len(objs) + 1):
        for detailed in (True, False):
            summaries.append({"objects": [{"id": o.track_id, "label": o.label, "box": [o.box[0] * W, o.box[1] * H,
                                                                                       o.box[2] * W, o.box[3] * H]}
                                          for o in objs[:n]],
                              "max_groups": 3 if n > 6 else 6, "detailed": detailed,
                              "out": summarize_objects(objs[:n], 3 if n > 6 else 6, detailed)})
    texts = ["A woman sits at a desk. A computer screen glows on your right. There is a sofa.",
             "A category label.", "A bear statue near a teddy bear.", "Two cups and a mug.", "The fridge is open.",
             "Someone is holding a phone.", ""]
    mention = [{"label": lab, "text": t, "out": mentioned_in(lab, t)} for t in texts
               for lab in ["person", "tv", "laptop", "couch", "dining table", "cup", "chair", "teddy bear", "cat",
                           "refrigerator", "cell phone", "bottle", "wine glass"]]
    return {"batches": batch_json, "runs": runs, "summaries": summaries, "mentioned": mention}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {"tracker": tracker_scenarios(), "spatial": spatial_cases(), "verifier": verifier_scenarios(),
            "monitor": monitor_scenarios(), "announcer": announcer_cases()}
    for k, v in data.items():
        (OUT / f"{k}.json").write_text(json.dumps(v, indent=1), encoding="utf-8")
        print(f"wrote {k}.json")


if __name__ == "__main__":
    main()
