package com.tanishadot.aura_native.detector

import java.nio.FloatBuffer
import kotlin.test.Test
import kotlin.test.assertEquals

class PixelSourceTest {
    /** 4x2 sensor image whose luma encodes the pixel index; neutral chroma (grey). */
    private fun yuv(rotation: Int, crop: IntArray? = null): YuvSource {
        val w = 4
        val h = 2
        val y = ByteArray(w * h) { (it * 10).toByte() }
        val u = ByteArray(2 * 1) { 128.toByte() }
        val v = ByteArray(2 * 1) { 128.toByte() }
        return YuvSource(y, w, u, v, 2, 1, w, h, rotation, crop)
    }

    private fun luma(s: PixelSource, x: Int, y: Int) = s.rgb(x, y) and 0xFF // grey: r = g = b = y

    @Test
    fun noRotation() {
        val s = yuv(0)
        assertEquals(4, s.width)
        assertEquals(10 * 5, luma(s, 1, 1)) // index 1*4+1
    }

    @Test
    fun rotation90MakesThePortraitUpright() {
        // Back camera, phone upright: sensor rotated 90 deg clockwise.
        val s = yuv(90)
        assertEquals(2, s.width)
        assertEquals(4, s.height)
        // upright top-left comes from the sensor's bottom-left (x=0, y=1) -> index 4
        assertEquals(40, luma(s, 0, 0))
        // upright top-right comes from the sensor's top-left -> index 0
        assertEquals(0, luma(s, 1, 0))
        // upright bottom-right comes from the sensor's top-right -> index 3
        assertEquals(30, luma(s, 1, 3))
    }

    @Test
    fun rotation180And270() {
        assertEquals(70, luma(yuv(180), 0, 0)) // sensor bottom-right (index 7)
        val s = yuv(270)
        assertEquals(30, luma(s, 0, 0)) // sensor top-right (index 3)
    }

    @Test
    fun cropOffsetsIntoTheUprightImage() {
        val s = yuv(0, intArrayOf(2, 0, 4, 2))
        assertEquals(2, s.width)
        assertEquals(20, luma(s, 0, 0)) // upright (2, 0)
    }

    @Test
    fun bt601Colours() {
        val red = YuvSource.yuvToRgb(76, -43, 127) // BT.601 encoding of pure red
        kotlin.test.assertTrue((red shr 16 and 0xFF) > 240 && (red shr 8 and 0xFF) < 20 && (red and 0xFF) < 20, "red=$red")
        assertEquals(0x808080, YuvSource.yuvToRgb(128, 0, 0)) // grey
    }

    @Test
    fun preprocessPadsWithGreyAndKeepsAspect() {
        // 2x4 white upright image -> 8x8 input: 4 wide, centred, grey bars left/right
        val white = ArgbSource(IntArray(8) { 0xFFFFFFFF.toInt() }, 2, 4)
        val lb = Letterbox(2, 4, 8)
        val out = FloatBuffer.allocate(8 * 8 * 3)
        Preprocess.fill(white, lb, out)
        val pad = 114f / 255f
        assertEquals(pad, out.get(0), 1e-6f) // (0,0) padding
        val centre = (4 * 8 + 4) * 3
        assertEquals(1f, out.get(centre), 1e-6f) // inside the image
        assertEquals(pad, out.get((4 * 8 + 7) * 3), 1e-6f) // right padding
    }
}
