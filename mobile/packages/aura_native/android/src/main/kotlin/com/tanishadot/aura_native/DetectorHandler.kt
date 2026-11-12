package com.tanishadot.aura_native

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Debug
import android.os.Handler
import android.os.Looper
import android.os.Process
import android.util.Log
import com.tanishadot.aura_native.detector.ArgbSource
import com.tanishadot.aura_native.detector.Detection
import com.tanishadot.aura_native.detector.DetectorGuard
import com.tanishadot.aura_native.detector.LoadKey
import com.tanishadot.aura_native.detector.PixelSource
import com.tanishadot.aura_native.detector.RequestGate
import com.tanishadot.aura_native.detector.YoloDetector
import com.tanishadot.aura_native.detector.YuvSource
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.util.concurrent.Executors

/**
 * `aurasight/detector`: YOLO11n on LiteRT.
 *
 * - All native work runs on one background thread, so there is only ever one
 *   interpreter and one inference at a time; replies are posted to the main thread.
 * - Inference requests are refused with code "busy" while another is pending
 *   (never queued), so frames can't accumulate.
 * - A `load` identical to the current one reuses the loaded model.
 * - Every native step is journalled by [DetectorGuard] (see `guardStatus`).
 * - Failures, including OutOfMemoryError, come back as channel errors; the app stays up.
 */
internal class DetectorHandler(filesDir: File) : MethodChannel.MethodCallHandler {
    private val exec = Executors.newSingleThreadExecutor { r ->
        Thread({
            Process.setThreadPriority(Process.THREAD_PRIORITY_DISPLAY) // keep up with the camera, below the UI
            r.run()
        }, "aura-detector")
    }
    private val main = Handler(Looper.getMainLooper())
    private val gate = RequestGate()
    val guard = DetectorGuard(File(filesDir, "detector_guard"), log = { Log.i(TAG, it) })
    private var detector: YoloDetector? = null
    private var loadedKey: LoadKey? = null
    private var iou = 0.5f

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "load" -> work(result) { load(call) }
            "detect" -> inference(result) { detect(frameSource(call.argument<Map<String, Any?>>("frame")!!, null), call) }
            "detectCrop" -> inference(result) {
                val rect = call.argument<List<Int>>("rect")!!.toIntArray()
                detect(frameSource(call.argument<Map<String, Any?>>("frame")!!, rect), call)
            }
            "detectImage" -> inference(result) { detect(imageSource(call.argument<ByteArray>("bytes")!!), call) }
            "dispose" -> work(result) {
                release()
                null
            }
            "guardStatus" -> result.success(guardStatus())
            "markGood" -> {
                guard.markGood(call.argument<String>("backend") ?: "cpu")
                result.success(null)
            }
            "clearGpuBlock" -> {
                guard.clearGpuBlock()
                result.success(null)
            }
            else -> result.notImplemented()
        }
    }

    private fun guardStatus(): Map<String, Any?> = mapOf(
        "interrupted" to guard.interrupted?.let {
            mapOf("stage" to it.stage, "backend" to it.backend, "atMs" to it.atMs, "detail" to it.detail)
        },
        "gpuBlocked" to guard.gpuBlockedReason,
        "lastGood" to guard.lastGoodBackend,
    )

    private fun inference(result: MethodChannel.Result, block: () -> Any?) {
        if (!gate.tryEnter()) {
            result.error("busy", "another detection is still running", null)
            return
        }
        work(result, onDone = gate::leave, block = block)
    }

    private fun work(result: MethodChannel.Result, onDone: () -> Unit = {}, block: () -> Any?) {
        try {
            exec.execute {
                try {
                    val r = block()
                    main.post { result.success(r) }
                } catch (e: OutOfMemoryError) {
                    Log.e(TAG, "out of memory", e)
                    val msg = "The phone ran out of memory for the detector (${memoryLine()})."
                    main.post { result.error("oom", msg, null) }
                } catch (e: Throwable) {
                    Log.e(TAG, "detector call failed", e)
                    main.post { result.error("detector", e.message ?: e.javaClass.simpleName, null) }
                } finally {
                    onDone()
                }
            }
        } catch (e: Throwable) { // executor already shut down
            onDone()
            result.error("detector", "detector is shut down", null)
        }
    }

    private fun load(call: MethodCall): Map<String, Any?> {
        val file = File(call.argument<String>("modelPath")!!)
        require(file.isFile) { "model file not found: ${file.path}" }
        val backend = call.argument<String>("backend") ?: "cpu"
        val threads = call.argument<Int>("threads")?.takeIf { it > 0 } ?: defaultThreads()
        val key = LoadKey(file.path, file.length(), file.lastModified(), call.argument<Int>("inputSize")!!,
            call.argument<Int>("numClasses")!!, call.argument<Boolean>("normalized")!!, backend, threads)
        iou = (call.argument<Double>("iou") ?: 0.5).toFloat()
        val current = detector
        if (current != null && key == loadedKey) {
            Log.i(TAG, "load: same model and backend already loaded, reusing it")
            return loadInfo(current, cached = true)
        }
        release()
        Log.i(TAG, "load: ${file.name} backend=$backend threads=$threads (${memoryLine()})")
        val d = YoloDetector.load(file, key.inputSize, key.numClasses, key.normalized, backend, threads, guard)
        detector = d
        loadedKey = key
        Log.i(TAG, "loaded on ${d.backend} in ${"%.0f".format(d.loadMs)} ms, warm-up ${"%.0f".format(d.warmupMs)} ms" +
            (d.fallbackReason?.let { ", fallback: $it" } ?: "") + " (${memoryLine()})")
        return loadInfo(d, cached = false)
    }

    private fun loadInfo(d: YoloDetector, cached: Boolean) = mapOf(
        "backend" to d.backend,
        "inputSize" to d.inputSize,
        "loadMs" to d.loadMs,
        "warmupMs" to d.warmupMs,
        "fallbackReason" to d.fallbackReason,
        "anchors" to d.numAnchors,
        "cached" to cached,
    ) + memory()

    private fun release() {
        detector?.close()
        detector = null
        loadedKey = null
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
        ) + memory()
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
        try {
            val px = IntArray(bmp.width * bmp.height)
            bmp.getPixels(px, 0, bmp.width, 0, 0, bmp.width, bmp.height)
            return ArgbSource(px, bmp.width, bmp.height)
        } finally {
            bmp.recycle()
        }
    }

    fun shutdown() {
        try {
            exec.execute { release() }
        } catch (_: Throwable) {
        }
        exec.shutdown()
    }

    companion object {
        const val TAG = "AuraDetector"

        /** Cheap memory readings (no PSS walk), attached to every reply. */
        fun memory(): Map<String, Any?> {
            val rt = Runtime.getRuntime()
            return mapOf(
                "nativeHeapMb" to Debug.getNativeHeapAllocatedSize() / (1024.0 * 1024.0),
                "javaHeapMb" to (rt.totalMemory() - rt.freeMemory()) / (1024.0 * 1024.0),
                "javaHeapMaxMb" to rt.maxMemory() / (1024.0 * 1024.0),
            )
        }

        fun memoryLine(): String {
            val m = memory()
            return "native heap %.1f MB, java heap %.1f of %.0f MB".format(
                m["nativeHeapMb"] as Double, m["javaHeapMb"] as Double, m["javaHeapMaxMb"] as Double)
        }
    }
}
