package com.tanishadot.aura_native.detector

import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * YOLO11n on LiteRT (TensorFlow Lite).
 *
 * The CPU (XNNPACK) is the default. The GPU delegate is only tried when asked
 * for ("gpu" or "auto"), and never when [DetectorGuard] recorded that a GPU
 * step killed the process before: GPU delegate and driver crashes are native
 * signals that no Kotlin `catch` can stop. Every native step is journalled in
 * the guard so the next start can say exactly where a crash happened.
 *
 * Not thread-safe: the channel handler calls it from one background thread.
 */
class YoloDetector private constructor(
    private val interpreter: Interpreter,
    private val gpu: GpuDelegate?,
    val backend: String,
    val inputSize: Int,
    val numClasses: Int,
    val numAnchors: Int,
    val channelsFirst: Boolean,
    private val normalized: Boolean,
    private val guard: DetectorGuard?,
) : AutoCloseable {
    var loadMs = 0.0
        private set
    var fallbackReason: String? = null
        private set
    var warmupMs = 0.0
        private set

    /** True once one real inference has completed; until then inferences are journalled. */
    private var proven = false
    private var closed = false

    private val input = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3).order(ByteOrder.nativeOrder())
    private val inputF = input.asFloatBuffer()
    private val output = ByteBuffer.allocateDirect(4 * (4 + numClasses) * numAnchors).order(ByteOrder.nativeOrder())
    private val outF = FloatArray((4 + numClasses) * numAnchors)

    class Timing(val preMs: Double, val modelMs: Double, val postMs: Double) {
        val totalMs get() = preMs + modelMs + postMs
    }

    fun detect(src: PixelSource, conf: Float, iou: Float): Pair<List<Detection>, Timing> {
        check(!closed) { "detector was closed" }
        val t0 = System.nanoTime()
        val lb = Letterbox(src.width, src.height, inputSize)
        Preprocess.fill(src, lb, inputF)
        val t1 = System.nanoTime()
        if (proven || guard == null) {
            runModel()
        } else {
            guard.step("first_inference", backend) { runModel() }
            proven = true
        }
        val t2 = System.nanoTime()
        val dets = YoloPost.decode(outF, numAnchors, numClasses, channelsFirst, normalized, lb, conf, iou)
        val t3 = System.nanoTime()
        return dets to Timing((t1 - t0) / 1e6, (t2 - t1) / 1e6, (t3 - t2) / 1e6)
    }

    private fun runModel() {
        input.rewind()
        output.rewind()
        interpreter.run(input, output)
        output.rewind()
        output.asFloatBuffer().get(outF)
    }

    /** Runs the model once on a plain grey image and checks the output is finite. */
    private fun warmUp() {
        val t0 = System.nanoTime()
        inputF.rewind()
        repeat(inputSize * inputSize * 3) { inputF.put(114f / 255f) }
        runModel()
        require(outF.all { it.isFinite() }) { "model produced non-finite output" }
        warmupMs = (System.nanoTime() - t0) / 1e6
    }

    override fun close() {
        if (closed) return
        closed = true
        // Interpreter first: it still references the delegate.
        interpreter.close()
        gpu?.close()
    }

    companion object {
        /**
         * @param backend "cpu" (default), "gpu" (fail if unavailable) or "auto" (GPU, else CPU)
         */
        fun load(
            file: File,
            inputSize: Int,
            numClasses: Int,
            normalized: Boolean,
            backend: String,
            threads: Int,
            guard: DetectorGuard? = null,
        ): YoloDetector {
            require(backend in setOf("cpu", "gpu", "auto")) { "unknown backend $backend" }
            val t0 = System.nanoTime()
            val model = step(guard, "map_model", backend) { mapFile(file) }
            var reason: String? = null
            if (backend != "cpu") {
                val blocked = guard?.gpuBlockedReason
                if (blocked != null) {
                    reason = "GPU switched off: $blocked"
                } else {
                    try {
                        val gpu = tryGpu(model, inputSize, numClasses, normalized, guard)
                        if (gpu != null) return gpu.also { it.loadMs = (System.nanoTime() - t0) / 1e6 }
                        reason = "GPU delegate not supported on this device"
                    } catch (e: Throwable) {
                        reason = "GPU failed: ${e.message ?: e.javaClass.simpleName}"
                    }
                }
                if (backend == "gpu") throw IllegalStateException(reason)
            }
            val interpreter = step(guard, "cpu_interpreter_create", "cpu", "threads=$threads") {
                Interpreter(model, Interpreter.Options().setNumThreads(threads).setUseXNNPACK(true))
            }
            return create(interpreter, null, "cpu", inputSize, numClasses, normalized, guard).also {
                it.loadMs = (System.nanoTime() - t0) / 1e6
                it.fallbackReason = reason
            }
        }

        /** The GPU detector, or null when the delegate isn't supported here. Java failures throw. */
        private fun tryGpu(model: MappedByteBuffer, inputSize: Int, numClasses: Int, normalized: Boolean,
                           guard: DetectorGuard?): YoloDetector? {
            val options = step(guard, "gpu_compat_check", "gpu") {
                val compat = CompatibilityList()
                try {
                    if (compat.isDelegateSupportedOnThisDevice) compat.bestOptionsForThisDevice else null
                } finally {
                    compat.close() // frees its native handle
                }
            } ?: return null
            val delegate = step(guard, "gpu_delegate_create", "gpu") { GpuDelegate(options) }
            try {
                val interpreter = step(guard, "gpu_interpreter_create", "gpu") {
                    Interpreter(model, Interpreter.Options().addDelegate(delegate))
                }
                return create(interpreter, delegate, "gpu", inputSize, numClasses, normalized, guard)
            } catch (e: Throwable) {
                delegate.close()
                throw e
            }
        }

        private fun create(it: Interpreter, gpu: GpuDelegate?, backend: String, inputSize: Int, numClasses: Int,
                           normalized: Boolean, guard: DetectorGuard?): YoloDetector {
            try {
                val (inShape, outShape) = step(guard, "tensor_shapes", backend) {
                    it.getInputTensor(0).shape() to it.getOutputTensor(0).shape()
                }
                val layout = checkShapes(inShape, outShape, inputSize, numClasses)
                val d = YoloDetector(it, gpu, backend, inputSize, numClasses, layout.second, layout.first,
                    normalized, guard)
                step(guard, "${backend}_warmup", backend) { d.warmUp() }
                return d
            } catch (e: Throwable) {
                it.close()
                throw e
            }
        }

        /**
         * Validates [1, S, S, 3] input and [1, 4+C, N] or [1, N, 4+C] output.
         * @return (channelsFirst, numAnchors)
         */
        fun checkShapes(inShape: IntArray, outShape: IntArray, inputSize: Int, numClasses: Int): Pair<Boolean, Int> {
            require(inShape.size == 4 && inShape[0] == 1 && inShape[1] == inputSize && inShape[2] == inputSize &&
                inShape[3] == 3) { "unexpected input shape ${inShape.contentToString()}, expected [1, $inputSize, $inputSize, 3]" }
            require(outShape.size == 3 && outShape[0] == 1) { "unexpected output shape ${outShape.contentToString()}" }
            val ch = 4 + numClasses
            return when (ch) {
                outShape[1] -> true to outShape[2]
                outShape[2] -> false to outShape[1]
                else -> throw IllegalArgumentException("output ${outShape.contentToString()} doesn't fit $numClasses classes")
            }
        }

        private inline fun <T> step(guard: DetectorGuard?, stage: String, backend: String, detail: String = "",
                                    crossinline block: () -> T): T =
            if (guard == null) block() else guard.step(stage, backend, detail) { block() }

        private fun mapFile(file: File): MappedByteBuffer = FileInputStream(file).use { s ->
            s.channel.map(FileChannel.MapMode.READ_ONLY, 0, s.channel.size())
        }
    }
}
