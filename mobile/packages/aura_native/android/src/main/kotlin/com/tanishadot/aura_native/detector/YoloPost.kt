package com.tanishadot.aura_native.detector

import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Square letterbox: the source (upright) image is scaled to fit [size] x [size]
 * keeping its aspect ratio, centred, and padded with grey (114).
 * Same maths as mobile/tools/export_detector.py `letterbox_square`.
 */
data class Letterbox(val srcW: Int, val srcH: Int, val size: Int) {
    val scale: Double = min(size.toDouble() / srcW, size.toDouble() / srcH)
    val newW: Int = (srcW * scale).roundToInt()
    val newH: Int = (srcH * scale).roundToInt()
    val padX: Int = (size - newW) / 2
    val padY: Int = (size - newH) / 2
}

data class Detection(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val score: Float, val cls: Int) {
    val area: Float get() = max(0f, x2 - x1) * max(0f, y2 - y1)
}

/**
 * YOLO11 output decoding and per-class NMS, matching the desktop detector
 * (backend/detection/yolo_onnx.py): argmax class, score >= conf, boxes mapped
 * back through the letterbox and clipped, then NMS per class at [iou].
 */
object YoloPost {
    /**
     * @param out flattened output tensor, either channels-first [4+C][N] or
     *   anchors-first [N][4+C] (see [channelsFirst])
     * @param normalized box coordinates are in [0, 1] of the model input
     */
    fun decode(
        out: FloatArray,
        numAnchors: Int,
        numClasses: Int,
        channelsFirst: Boolean,
        normalized: Boolean,
        lb: Letterbox,
        conf: Float,
        iou: Float = 0.5f,
        maxDet: Int = 100,
    ): List<Detection> {
        val ch = 4 + numClasses
        fun at(c: Int, a: Int): Float = if (channelsFirst) out[c * numAnchors + a] else out[a * ch + c]
        val k = if (normalized) lb.size.toFloat() else 1f
        val cands = ArrayList<Detection>()
        for (a in 0 until numAnchors) {
            var best = -1f
            var bestC = 0
            for (c in 0 until numClasses) {
                val s = at(4 + c, a)
                if (s > best) {
                    best = s
                    bestC = c
                }
            }
            if (best < conf) continue
            val cx = at(0, a) * k
            val cy = at(1, a) * k
            val w = at(2, a) * k
            val h = at(3, a) * k
            val s = lb.scale.toFloat()
            val x1 = ((cx - w / 2 - lb.padX) / s).coerceIn(0f, lb.srcW.toFloat())
            val y1 = ((cy - h / 2 - lb.padY) / s).coerceIn(0f, lb.srcH.toFloat())
            val x2 = ((cx + w / 2 - lb.padX) / s).coerceIn(0f, lb.srcW.toFloat())
            val y2 = ((cy + h / 2 - lb.padY) / s).coerceIn(0f, lb.srcH.toFloat())
            cands += Detection(x1, y1, x2, y2, best, bestC)
        }
        return nms(cands, iou, maxDet)
    }

    fun iou(a: Detection, b: Detection): Float {
        val ix = max(0f, min(a.x2, b.x2) - max(a.x1, b.x1))
        val iy = max(0f, min(a.y2, b.y2) - max(a.y1, b.y1))
        val inter = ix * iy
        return inter / max(a.area + b.area - inter, 1e-6f)
    }

    /** Greedy NMS within each class; highest scores first; at most [maxDet] kept. */
    fun nms(dets: List<Detection>, iou: Float, maxDet: Int): List<Detection> {
        val sorted = dets.sortedByDescending { it.score }
        val kept = ArrayList<Detection>()
        val removed = BooleanArray(sorted.size)
        for (i in sorted.indices) {
            if (removed[i]) continue
            val d = sorted[i]
            kept += d
            if (kept.size >= maxDet) break
            for (j in i + 1 until sorted.size) {
                if (!removed[j] && sorted[j].cls == d.cls && iou(d, sorted[j]) > iou) removed[j] = true
            }
        }
        return kept
    }
}
