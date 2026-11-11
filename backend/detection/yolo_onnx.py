"""YOLO11 object detection with ONNX Runtime (CPU-friendly, no PyTorch at runtime).

The model is exported once from Ultralytics (see scripts/export_yolo.py) with
dynamic input shapes, so the inference resolution can be changed at runtime.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("aurasight.detector")

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class Detections:
    boxes: np.ndarray  # (N,4) float32 xyxy in original image pixels
    scores: np.ndarray  # (N,)
    class_ids: np.ndarray  # (N,) int
    inference_ms: float = 0.0

    def __len__(self) -> int:
        return len(self.scores)

    @staticmethod
    def empty() -> "Detections":
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img
    # Pad to a multiple of 32 (not a full square) -> fewer pixels for 4:3 webcams.
    pw, ph = int(np.ceil(nw / 32) * 32), int(np.ceil(nh / 32) * 32)
    left, top = (pw - nw) // 2, (ph - nh) // 2
    out = cv2.copyMakeBorder(resized, top, ph - nh - top, left, pw - nw - left,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, r, (left, top)


class YoloDetector:
    """Lazily-initialised YOLO ONNX detector. Thread-safe for a single caller."""

    def __init__(self, model_path: Path, threads: int = 0, iou_threshold: float = 0.5):
        self.model_path = Path(model_path)
        self.threads = threads
        self.iou_threshold = iou_threshold
        self.names = COCO_NAMES
        self._session = None
        self._input_name = ""
        self._lock = threading.Lock()
        self.state = "not_loaded"  # not_loaded | loading | ready | error
        self.error: str | None = None
        self.last_inference_ms = 0.0

    def status(self) -> dict:
        return {
            "state": self.state,
            "error": self.error,
            "model": self.model_path.name,
            "inference_ms": round(self.last_inference_ms, 1),
        }

    def load(self) -> bool:
        with self._lock:
            if self._session is not None:
                return True
            self.state, self.error = "loading", None
            if not self.model_path.exists():
                self.state = "error"
                self.error = (f"Model file not found: {self.model_path}. "
                              "Run scripts/setup_models.py to download it.")
                return False
            try:
                import onnxruntime as ort

                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                # Leave headroom for the VLM and UI on small CPUs.
                so.intra_op_num_threads = self.threads or max(1, min(4, (os.cpu_count() or 2) // 2))
                so.inter_op_num_threads = 1
                self._session = ort.InferenceSession(
                    str(self.model_path), sess_options=so, providers=["CPUExecutionProvider"]
                )
                self._input_name = self._session.get_inputs()[0].name
                self.state = "ready"
                log.info("detector loaded: %s", self.model_path.name)
                return True
            except Exception as e:
                self.state, self.error = "error", f"Failed to load detector: {e}"
                log.exception("detector load failed")
                return False

    def detect(self, img_bgr: np.ndarray, imgsz: int = 416, conf: float = 0.1,
               max_det: int = 100) -> Detections:
        if self._session is None and not self.load():
            raise RuntimeError(self.error or "detector unavailable")
        h0, w0 = img_bgr.shape[:2]
        lb, r, (padx, pady) = letterbox(img_bgr, imgsz)
        blob = cv2.dnn.blobFromImage(lb, 1 / 255.0, swapRB=True)  # NCHW float32 RGB
        t0 = time.perf_counter()
        out = self._session.run(None, {self._input_name: blob})[0]  # (1, 84, N)
        self.last_inference_ms = (time.perf_counter() - t0) * 1000
        pred = out[0].T  # (N, 84)
        cls_scores = pred[:, 4:]
        class_ids = cls_scores.argmax(1)
        scores = cls_scores[np.arange(len(class_ids)), class_ids]
        keep = scores >= conf
        if not keep.any():
            d = Detections.empty()
            d.inference_ms = self.last_inference_ms
            return d
        pred, scores, class_ids = pred[keep], scores[keep], class_ids[keep]
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        x1 = (cx - w / 2 - padx) / r
        y1 = (cy - h / 2 - pady) / r
        x2 = (cx + w / 2 - padx) / r
        y2 = (cy + h / 2 - pady) / r
        boxes = np.stack([x1, y1, x2, y2], 1)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)
        xywh = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], 1)
        idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), scores.tolist(), class_ids.tolist(),
                                      conf, self.iou_threshold)
        idx = np.array(idx, dtype=np.int64).reshape(-1)[:max_det]
        return Detections(boxes[idx].astype(np.float32), scores[idx].astype(np.float32),
                          class_ids[idx].astype(np.int64), self.last_inference_ms)
