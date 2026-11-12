package com.tanishadot.aura_native

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Handler
import android.os.Looper
import android.os.Process
import com.tanishadot.aura_native.detector.ArgbSource
import com.tanishadot.aura_native.detector.Detection
import com.tanishadot.aura_native.detector.PixelSource
import com.tanishadot.aura_native.detector.YoloDetector
import com.tanishadot.aura_native.detector.YuvSource
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.util.concurrent.Executors

/**
 * `aurasight/detector`: YOLO11n on LiteRT. All work runs on one background
 * thread so the UI thread never blocks; replies are posted to the main thread.
 */
internal class DetectorHandler : MethodChannel.MethodCallHandler {
    private val exec = Executors.newSingleThreadExecutor { r ->
        Thread({
            Process.setThreadPriority(Process.THREAD_PRIORITY_DISPLAY) // keep up with the camera, below the UI
            r.run()
        }, "aura-detector")
    }
    private val main = Handler(Looper.getMainLooper())
    private var detector: YoloDetector? = null
    private var iou = 0.5f

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "load" -> work(result) { load(call) }
            "detect" -> work(result) { detect(frameSource(call.argument<Map<String, Any?>>("frame")!!, null), call) }
            "detectCrop" -> work(result) {
                val rect = call.argument<List<Int>>("rect")!!.toIntArray()
                detect(frameSource(call.argument<Map<String, Any?>>("frame")!!, rect), call)
            }
            "detectImage" -> work(result) { detect(imageSource(call.argument<ByteArray>("bytes")!!), call) }
            "dispose" -> work(result) {
                detector?.close()
                detector = null
                null
            }
            else -> result.notImplemented()
        }
    }

    private fun work(result: MethodChannel.Result, block: () -> Any?) {
        exec.execute {
            try {
                val r = block()
                main.post { result.success(r) }
            } catch (e: Throwable) {
                main.post { result.error("detector", e.message ?: e.javaClass.simpleName, null) }
            }
        }
    }

    private fun load(call: MethodCall): Map<String, Any?> {
        detector?.close()
        detector = null
        val file = File(call.argument<String>("modelPath")!!)
        require(file.isFile) { "model file not found: ${file.path}" }
        iou = (call.argument<Double>("iou") ?: 0.5).toFloat()
        val d = YoloDetector.load(
            file,
            inputSize = call.argument<Int>("inputSize")!!,
            numClasses = call.argument<Int>("numClasses")!!,
            normalized = call.argument<Boolean>("normalized")!!,
            backend = call.argument<String>("backend") ?: "auto",
            threads = call.argument<Int>("threads")?.takeIf { it > 0 } ?: defaultThreads(),
        )
        detector = d
        return mapOf(
            "backend" to d.backend,
            "inputSize" to d.inputSize,
            "loadMs" to d.loadMs,
            "fallbackReason" to d.fallbackReason,
            "anchors" to d.numAnchors,
        )
    }

    /** Big cores only: the Snapdragon 695 has 2 fast + 6 efficiency cores. */
    private fun defaultThreads() = (Runtime.getRuntime().availableProcessors() / 2).coerceIn(2, 4)

    private fun detect(src: PixelSource, call: MethodCall): Map<String, Any?> {
        val d = detector ?: throw IllegalStateException("detector not loaded")
        val conf = (call.argument<Double>("conf") ?: 0.1).toFloat()
        val (dets, t) = d.detect(src, conf, iou)
        return mapOf(
            "boxes" to dets.map { listOf(it.x1.toDouble(), it.y1.toDouble(), it.x2.toDouble(), it.y2.toDouble()) },
            "scores" to dets.map { it.score.toDouble() },
            "classes" to dets.map(Detection::cls),
            "width" to src.width,
            "height" to src.height,
            "inferenceMs" to t.totalMs,
            "preMs" to t.preMs,
            "modelMs" to t.modelMs,
            "postMs" to t.postMs,
            "backend" to d.backend,
        )
    }

    @Suppress("UNCHECKED_CAST")
    private fun frameSource(frame: Map<String, Any?>, crop: IntArray?): PixelSource {
        val planes = frame["planes"] as List<Map<String, Any?>>
        require(planes.size >= 3) { "expected 3 YUV planes, got ${planes.size}" }
        val (y, u, v) = planes
        return YuvSource(
            y["bytes"] as ByteArray, y["rowStride"] as Int,
            u["bytes"] as ByteArray, v["bytes"] as ByteArray, u["rowStride"] as Int, u["pixelStride"] as Int,
            frame["width"] as Int, frame["height"] as Int, frame["rotation"] as Int, crop,
        )
    }

    private fun imageSource(bytes: ByteArray): PixelSource {
        val opts = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
        val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts)
            ?: throw IllegalArgumentException("could not decode image")
        val px = IntArray(bmp.width * bmp.height)
        bmp.getPixels(px, 0, bmp.width, 0, 0, bmp.width, bmp.height)
        val src = ArgbSource(px, bmp.width, bmp.height)
        bmp.recycle()
        return src
    }

    fun shutdown() {
        exec.execute {
            detector?.close()
            detector = null
        }
        exec.shutdown()
    }
}
