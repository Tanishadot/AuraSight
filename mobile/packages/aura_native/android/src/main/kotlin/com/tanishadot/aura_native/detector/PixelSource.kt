package com.tanishadot.aura_native.detector

import java.nio.FloatBuffer
import kotlin.math.floor

/**
 * An upright RGB image the detector can sample. Coordinates are in upright
 * pixels; implementations handle sensor rotation and cropping.
 */
interface PixelSource {
    val width: Int
    val height: Int

    /** RGB at an integer upright pixel (clamped to the image), packed 0xRRGGBB. */
    fun rgb(x: Int, y: Int): Int
}

/** ARGB_8888 pixels (decoded JPEG/PNG), already upright. */
class ArgbSource(private val pixels: IntArray, override val width: Int, override val height: Int) : PixelSource {
    override fun rgb(x: Int, y: Int): Int {
        val cx = x.coerceIn(0, width - 1)
        val cy = y.coerceIn(0, height - 1)
        return pixels[cy * width + cx] and 0xFFFFFF
    }
}

/**
 * Android YUV_420_888 camera frame (separate Y, U, V planes with their row and
 * pixel strides; covers both I420 and NV12/NV21 layouts).
 *
 * @param rotation clockwise degrees that make the sensor image upright
 * @param crop optional upright sub-rectangle [x1, y1, x2, y2] to expose as the image
 */
class YuvSource(
    private val y: ByteArray, private val yRow: Int,
    private val u: ByteArray, private val v: ByteArray, private val uvRow: Int, private val uvPixel: Int,
    private val sensorW: Int, private val sensorH: Int, private val rotation: Int,
    crop: IntArray? = null,
) : PixelSource {
    val uprightW = if (rotation % 180 == 0) sensorW else sensorH
    val uprightH = if (rotation % 180 == 0) sensorH else sensorW
    private val ox = crop?.get(0) ?: 0
    private val oy = crop?.get(1) ?: 0
    override val width = crop?.let { it[2] - it[0] } ?: uprightW
    override val height = crop?.let { it[3] - it[1] } ?: uprightH

    override fun rgb(x: Int, y: Int): Int {
        val ux = (x + ox).coerceIn(0, uprightW - 1)
        val uy = (y + oy).coerceIn(0, uprightH - 1)
        // upright (u, v) -> sensor (sx, sy) for a clockwise rotation
        val sx: Int
        val sy: Int
        when (rotation) {
            90 -> { sx = uy; sy = sensorH - 1 - ux }
            180 -> { sx = sensorW - 1 - ux; sy = sensorH - 1 - uy }
            270 -> { sx = sensorW - 1 - uy; sy = ux }
            else -> { sx = ux; sy = uy }
        }
        val yy = (this.y[sy * yRow + sx].toInt() and 0xFF)
        val ci = (sy shr 1) * uvRow + (sx shr 1) * uvPixel
        val uu = (u[ci].toInt() and 0xFF) - 128
        val vv = (v[ci].toInt() and 0xFF) - 128
        return yuvToRgb(yy, uu, vv)
    }

    companion object {
        /** BT.601 full-range YUV -> RGB (what Android cameras deliver). */
        fun yuvToRgb(y: Int, u: Int, v: Int): Int {
            val r = (y + 1.402f * v).toInt().coerceIn(0, 255)
            val g = (y - 0.344136f * u - 0.714136f * v).toInt().coerceIn(0, 255)
            val b = (y + 1.772f * u).toInt().coerceIn(0, 255)
            return (r shl 16) or (g shl 8) or b
        }
    }
}

/**
 * Fills a [size]x[size]x3 float input (RGB / 255, NHWC) with the letterboxed
 * image, using bilinear sampling with half-pixel centres like OpenCV's
 * INTER_LINEAR, and grey (114) padding.
 */
object Preprocess {
    private const val PAD = 114f / 255f

    fun fill(src: PixelSource, lb: Letterbox, out: FloatBuffer) {
        out.rewind()
        val inv = 1.0 / lb.scale
        for (dy in 0 until lb.size) {
            val inY = dy >= lb.padY && dy < lb.padY + lb.newH
            val syf = ((dy - lb.padY + 0.5) * inv - 0.5).coerceAtLeast(0.0)
            val y0 = floor(syf).toInt()
            val fy = (syf - y0).toFloat()
            for (dx in 0 until lb.size) {
                if (!inY || dx < lb.padX || dx >= lb.padX + lb.newW) {
                    out.put(PAD); out.put(PAD); out.put(PAD)
                    continue
                }
                val sxf = ((dx - lb.padX + 0.5) * inv - 0.5).coerceAtLeast(0.0)
                val x0 = floor(sxf).toInt()
                val fx = (sxf - x0).toFloat()
                val p00 = src.rgb(x0, y0)
                val p10 = src.rgb(x0 + 1, y0)
                val p01 = src.rgb(x0, y0 + 1)
                val p11 = src.rgb(x0 + 1, y0 + 1)
                out.put(lerp(p00, p10, p01, p11, 16, fx, fy))
                out.put(lerp(p00, p10, p01, p11, 8, fx, fy))
                out.put(lerp(p00, p10, p01, p11, 0, fx, fy))
            }
        }
        out.rewind()
    }

    @Suppress("NOTHING_TO_INLINE")
    private inline fun lerp(p00: Int, p10: Int, p01: Int, p11: Int, shift: Int, fx: Float, fy: Float): Float {
        val a = (p00 shr shift and 0xFF) * (1 - fx) + (p10 shr shift and 0xFF) * fx
        val b = (p01 shr shift and 0xFF) * (1 - fx) + (p11 shr shift and 0xFF) * fx
        return (a * (1 - fy) + b * fy) / 255f
    }
}
