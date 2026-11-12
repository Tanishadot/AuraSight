"""Export, validate and describe the Android detector model (YOLO11n -> LiteRT/TFLite).

Two commands, both run in CI (.github/workflows/mobile.yml):

  export    Download the pinned yolo11n.pt (same file and SHA-256 as the desktop's
            scripts/setup_models.py), export a float16 TFLite model with a fixed
            320x320 input, validate it and write the app manifest
            (mobile/app/assets/models/detector.json) with its size and SHA-256.

  validate  Check a downloaded model against the committed manifest (size, SHA-256,
            tensor shapes) and compare its detections on the fixture images with
            the desktop detector's (integration_test/golden/desktop_detections_320.json).

The pre/post-processing here mirrors the Android code (aura_native YoloPost.kt):
square letterbox with grey padding, RGB/255, per-class NMS at IoU 0.5.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.detection.yolo_onnx import COCO_NAMES  # noqa: E402  (single source of labels)

PT_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
PT_SHA = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
SIZE = 320
FILE_NAME = "yolo11n_320_float16.tflite"
RELEASE_TAG = "mobile-models-v1"
REPO = "Tanishadot/AuraSight"
MANIFEST = ROOT / "mobile" / "app" / "assets" / "models" / "detector.json"
GOLDEN = ROOT / "mobile" / "app" / "integration_test" / "golden" / "desktop_detections_320.json"
SCENES = ROOT / "tests" / "fixtures" / "scenes"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def interpreter(path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=str(path))
    it.allocate_tensors()
    return it


# --------------------------------------------------------------------------- mobile pipeline (reference)
def letterbox_square(img_bgr: np.ndarray, size: int):
    h, w = img_bgr.shape[:2]
    r = min(size / w, size / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb[None], r, (pad_x, pad_y)


def run(it, img_bgr: np.ndarray, conf: float, normalized: bool):
    inp = it.get_input_details()[0]
    x, r, (px, py) = letterbox_square(img_bgr, SIZE)
    it.set_tensor(inp["index"], x.astype(inp["dtype"]))
    it.invoke()
    out = it.get_tensor(it.get_output_details()[0]["index"])[0]
    pred = out.T if out.shape[0] == 84 else out  # (N, 84)
    cls_scores = pred[:, 4:]
    cls = cls_scores.argmax(1)
    scores = cls_scores[np.arange(len(cls)), cls]
    keep = scores >= conf
    pred, scores, cls = pred[keep], scores[keep], cls[keep]
    if not len(scores):
        return []
    k = SIZE if normalized else 1.0
    cx, cy, w, h = pred[:, 0] * k, pred[:, 1] * k, pred[:, 2] * k, pred[:, 3] * k
    h0, w0 = img_bgr.shape[:2]
    x1 = ((cx - w / 2 - px) / r).clip(0, w0)
    y1 = ((cy - h / 2 - py) / r).clip(0, h0)
    x2 = ((cx + w / 2 - px) / r).clip(0, w0)
    y2 = ((cy + h / 2 - py) / r).clip(0, h0)
    xywh = np.stack([x1, y1, x2 - x1, y2 - y1], 1)
    idx = np.array(cv2.dnn.NMSBoxesBatched(xywh.tolist(), scores.tolist(), cls.tolist(), conf, 0.5)).reshape(-1)[:100]
    return [{"cls": int(cls[i]), "label": COCO_NAMES[int(cls[i])], "score": float(scores[i]),
             "box": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])]} for i in idx]


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1e-6)


def compare(reference: list[dict], mobile: list[dict], min_score: float = 0.4):
    """Match reference detections (score >= min_score) to mobile ones of the same class."""
    matched, ious, dscores, missed = 0, [], [], []
    used = set()
    refs = [d for d in reference if d["score"] >= min_score]
    for d in refs:
        best, bj = 0.0, -1
        for j, m in enumerate(mobile):
            if j in used or m["cls"] != d["cls"]:
                continue
            v = iou(d["box"], m["box"])
            if v > best:
                best, bj = v, j
        if best >= 0.5:
            matched += 1
            used.add(bj)
            ious.append(best)
            dscores.append(mobile[bj]["score"] - d["score"])
        else:
            missed.append(f'{d["label"]} {d["score"]:.2f}')
    extra = [f'{m["label"]} {m["score"]:.2f}' for j, m in enumerate(mobile) if j not in used and m["score"] >= min_score]
    return len(refs), matched, ious, dscores, missed, extra


def parity(model: Path, normalized: bool) -> dict:
    it = interpreter(model)
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    total = hit = 0
    all_ious, all_ds, rows = [], [], []
    for g in golden["images"]:
        img = cv2.imread(str(SCENES / g["file"]))
        mob = run(it, img, golden["conf"], normalized)
        n, m, ious, ds, missed, extra = compare(g["detections"], mob)
        total, hit = total + n, hit + m
        all_ious += ious
        all_ds += ds
        rows.append({"file": g["file"], "reference": n, "matched": m, "missed": missed, "extra": extra,
                     "mobile": [f'{d["label"]} {d["score"]:.2f}' for d in mob]})
    return {
        "reference_detections": total,
        "matched": hit,
        "recall": hit / total if total else 1.0,
        "mean_iou": float(np.mean(all_ious)) if all_ious else None,
        "mean_score_diff": float(np.mean(all_ds)) if all_ds else None,
        "max_abs_score_diff": float(np.max(np.abs(all_ds))) if all_ds else None,
        "images": rows,
    }


def describe(model: Path) -> dict:
    it = interpreter(model)
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    # Are box coordinates normalised to [0, 1]? Run one real image and look.
    img = cv2.imread(str(SCENES / "person_living_room.jpg"))
    x, _, _ = letterbox_square(img, SIZE)
    it.set_tensor(i["index"], x.astype(i["dtype"]))
    it.invoke()
    out = it.get_tensor(o["index"])[0]
    pred = out.T if out.shape[0] == 84 else out
    best = pred[pred[:, 4:].max(1).argmax()]
    normalized = bool(np.abs(best[:4]).max() <= 2.0)
    return {"input_shape": [int(v) for v in i["shape"]], "input_dtype": str(np.dtype(i["dtype"])),
            "output_shape": [int(v) for v in o["shape"]], "normalized": normalized}


def write_report(path: Path, info: dict, par: dict) -> None:
    lines = [
        "# Detector conversion parity (TFLite vs desktop ONNX)", "",
        f"- Model: `{FILE_NAME}`, input {info['input_shape']} {info['input_dtype']}, output {info['output_shape']}, "
        f"boxes normalised: {info['normalized']}",
        f"- Reference detections (score >= 0.40): {par['reference_detections']}, matched: {par['matched']} "
        f"(recall {par['recall']:.2%})",
        f"- Mean IoU of matches: {par['mean_iou']:.3f}" if par["mean_iou"] is not None else "- Mean IoU: n/a",
        f"- Mean score difference (mobile - desktop): {par['mean_score_diff']:+.3f}, max |diff| "
        f"{par['max_abs_score_diff']:.3f}" if par["mean_score_diff"] is not None else "", "",
        "Expected differences: the desktop pads to a multiple of 32 (non-square) while the phone model has a",
        "fixed square 320x320 input; float16 weights; and small resampling differences.", "",
        "| Image | Ref | Matched | Missed | Extra (mobile only) |", "|---|---|---|---|---|",
    ]
    for r in par["images"]:
        lines.append(f"| {r['file']} | {r['reference']} | {r['matched']} | {', '.join(r['missed']) or '-'} | "
                     f"{', '.join(r['extra']) or '-'} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_export(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pt = out_dir / "yolo11n.pt"
    if not pt.exists():
        urllib.request.urlretrieve(PT_URL, pt)
    if sha256(pt) != PT_SHA:
        sys.exit("yolo11n.pt checksum mismatch")
    import ultralytics
    from ultralytics import YOLO

    YOLO(str(pt)).export(format="tflite", imgsz=SIZE, half=True)
    found = glob.glob(str(out_dir / "**" / "*float16.tflite"), recursive=True) or glob.glob("**/*float16.tflite",
                                                                                           recursive=True)
    if not found:
        sys.exit("export produced no float16 .tflite")
    model = out_dir / FILE_NAME
    shutil.copy(found[0], model)
    info = describe(model)
    par = parity(model, info["normalized"])
    write_report(out_dir / "parity.md", info, par)
    try:
        import tensorflow as tf
        tf_version = tf.__version__
    except ImportError:
        tf_version = None
    manifest = {
        "id": "yolo11n-320-f16",
        "title": "Object detector (YOLO11n)",
        "purpose": "Finds and tracks everyday objects in the camera view.",
        "license": "AGPL-3.0, Ultralytics",
        "file": {
            "name": FILE_NAME,
            "url": f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{FILE_NAME}",
            "bytes": model.stat().st_size,
            "sha256": sha256(model),
        },
        "inputSize": SIZE,
        "inputShape": info["input_shape"],
        "outputShape": info["output_shape"],
        "normalizedBoxes": info["normalized"],
        "iouThreshold": 0.5,
        "labels": COCO_NAMES,
        "export": {"source": PT_URL, "sourceSha256": PT_SHA, "ultralytics": ultralytics.__version__,
                   "tensorflow": tf_version, "precision": "float16 weights, float32 input/output"},
    }
    (out_dir / "detector.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "labels"}, indent=1))
    print((out_dir / "parity.md").read_text(encoding="utf-8"))
    if par["recall"] < 0.85:
        sys.exit(f"parity too low: recall {par['recall']:.2%}")


def cmd_validate(model: Path, report: Path) -> None:
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    f = m["file"]
    errors = []
    if model.stat().st_size != f["bytes"]:
        errors.append(f"size {model.stat().st_size} != manifest {f['bytes']}")
    if sha256(model) != f["sha256"]:
        errors.append("SHA-256 does not match the manifest")
    info = describe(model)
    if info["input_shape"] != m["inputShape"] or info["output_shape"] != m["outputShape"]:
        errors.append(f"tensor shapes {info} differ from manifest")
    if len(m["labels"]) != 80 or m["labels"] != COCO_NAMES:
        errors.append("labels differ from the desktop COCO list")
    if info["normalized"] != m["normalizedBoxes"]:
        errors.append("box normalisation differs from manifest")
    par = parity(model, m["normalizedBoxes"])
    write_report(report, info, par)
    print(report.read_text(encoding="utf-8"))
    if par["recall"] < 0.85:
        errors.append(f"parity too low: recall {par['recall']:.2%}")
    if errors:
        sys.exit("\n".join(errors))
    print("model valid")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", type=Path, default=Path("dist"))
    v = sub.add_parser("validate")
    v.add_argument("--model", type=Path, required=True)
    v.add_argument("--report", type=Path, default=Path("parity.md"))
    a = ap.parse_args()
    cmd_export(a.out) if a.cmd == "export" else cmd_validate(a.model, a.report)
