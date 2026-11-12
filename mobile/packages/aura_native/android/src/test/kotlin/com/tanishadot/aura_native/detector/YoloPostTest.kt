package com.tanishadot.aura_native.detector

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class YoloPostTest {
    private val numClasses = 80

    /** Builds a channels-first [4+C][N] output with the given anchors set. */
    private fun output(n: Int, anchors: List<FloatArray>, channelsFirst: Boolean = true): FloatArray {
        val ch = 4 + numClasses
        val out = FloatArray(ch * n)
        anchors.forEachIndexed { a, v ->
            for (c in 0 until ch) {
                if (channelsFirst) out[c * n + a] = v[c] else out[a * ch + c] = v[c]
            }
        }
        return out
    }

    private fun anchor(cx: Float, cy: Float, w: Float, h: Float, cls: Int, score: Float): FloatArray =
        FloatArray(4 + numClasses).also {
            it[0] = cx; it[1] = cy; it[2] = w; it[3] = h; it[4 + cls] = score
        }

    @Test
    fun letterboxMatchesTheReferenceMaths() {
        val portrait = Letterbox(480, 640, 320) // upright phone frame
        assertEquals(0.5, portrait.scale)
        assertEquals(240, portrait.newW)
        assertEquals(320, portrait.newH)
        assertEquals(40, portrait.padX)
        assertEquals(0, portrait.padY)
        val landscape = Letterbox(640, 427, 320)
        assertEquals(320, landscape.newW)
        assertEquals(214, landscape.newH)
        assertEquals(53, landscape.padY)
    }

    @Test
    fun decodesNormalisedBoxesBackToSourcePixels() {
        val lb = Letterbox(480, 640, 320)
        // a person centred in the model input at (160, 160), 100x200 model pixels
        val out = output(4, listOf(anchor(0.5f, 0.5f, 100f / 320, 200f / 320, 0, 0.9f)))
        val d = YoloPost.decode(out, 4, numClasses, true, true, lb, conf = 0.25f).single()
        assertEquals(0, d.cls)
        assertEquals(0.9f, d.score)
        // model x 110..210 -> minus pad 40 -> 70..170 -> / 0.5 -> 140..340
        assertEquals(140f, d.x1, 1e-3f)
        assertEquals(340f, d.x2, 1e-3f)
        assertEquals(120f, d.y1, 1e-3f)
        assertEquals(520f, d.y2, 1e-3f)
    }

    @Test
    fun pixelBoxesAndAnchorsFirstLayout() {
        val lb = Letterbox(320, 320, 320)
        val out = output(2, listOf(anchor(100f, 100f, 20f, 20f, 41, 0.7f)), channelsFirst = false)
        val d = YoloPost.decode(out, 2, numClasses, false, false, lb, conf = 0.25f).single()
        assertEquals(41, d.cls) // cup
        assertEquals(90f, d.x1, 1e-3f)
    }

    @Test
    fun confidenceFilterAndClipping() {
        val lb = Letterbox(320, 320, 320)
        val out = output(3, listOf(
            anchor(0.5f, 0.5f, 0.2f, 0.2f, 2, 0.2f), // below threshold
            anchor(0.0f, 0.0f, 0.5f, 0.5f, 3, 0.8f), // spills off the top-left
        ))
        val dets = YoloPost.decode(out, 3, numClasses, true, true, lb, conf = 0.25f)
        assertEquals(1, dets.size)
        assertEquals(0f, dets[0].x1)
        assertEquals(0f, dets[0].y1)
    }

    @Test
    fun nmsSuppressesOverlapsOnlyWithinAClass() {
        val a = Detection(0f, 0f, 100f, 100f, 0.9f, 0)
        val b = Detection(5f, 5f, 105f, 105f, 0.8f, 0) // same class, IoU ~0.82 -> removed
        val c = Detection(5f, 5f, 105f, 105f, 0.7f, 56) // different class -> kept
        val d = Detection(200f, 200f, 260f, 260f, 0.6f, 0) // no overlap -> kept
        val kept = YoloPost.nms(listOf(b, d, a, c), 0.5f, 100)
        assertEquals(listOf(a, c, d), kept)
    }

    @Test
    fun nmsRespectsMaxDetections() {
        val many = (0 until 150).map { Detection(it * 10f, 0f, it * 10f + 5f, 5f, 1f - it / 1000f, 0) }
        assertEquals(100, YoloPost.nms(many, 0.5f, 100).size)
    }
}
