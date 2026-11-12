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
 * YOLO11n on LiteRT (TensorFlow Lite). Tries the GPU delegate first (Adreno on
 * Snapdragon), verified with a warm-up inference, and falls back to the CPU
 * (XNNPACK) if the GPU is unsupported or fails. Not thread-safe: the channel
 * handler calls it from one background thread.
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
) : AutoCloseable {
    var loadMs = 0.0
        private set
    var fallbackReason: String? = null
        private set

    private val input = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3).order(ByteOrder.nativeOrder())
    private val inputF = input.asFloatBuffer()
    private val output = ByteBuffer.allocateDirect(4 * (4 + numClasses) * numAnchors).order(ByteOrder.nativeOrder())
    private val outF = FloatArray((4 + numClasses) * numAnchors)

    class Timing(val preMs: Double, val modelMs: Double, val postMs: Double) {
        val totalMs get() = preMs + modelMs + postMs
    }

    fun detect(src: PixelSource, conf: Float, iou: Float): Pair<List<Detection>, Timing> {
        val t0 = System.nanoTime()
        val lb = Letterbox(src.width, src.height, inputSize)
        Preprocess.fill(src, lb, inputF)
        val t1 = System.nanoTime()
        runModel()
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

    /** Runs the model on a plain grey image and checks the output is sane. */
    private fun selfCheck() {
        inputF.rewind()
        repeat(inputSize * inputSize * 3) { inputF.put(114f / 255f) }
        runModel()
        require(outF.all { it.isFinite() }) { "model produced non-finite output" }
    }

    override fun close() {
        interpreter.close()
        gpu?.close()
    }

    companion object {
        /**
         * @param backend "auto" (GPU, else CPU), "gpu" (fail if unavailable) or "cpu"
         */
        fun load(file: File, inputSize: Int, numClasses: Int, normalized: Boolean, backend: String, threads: Int): YoloDetector {
            val t0 = System.nanoTime()
            val model = mapFile(file)
            var reason: String? = null
            if (backend == "auto" || backend == "gpu") {
                try {
                    val compat = CompatibilityList()
                    if (compat.isDelegateSupportedOnThisDevice) {
                        val delegate = GpuDelegate(compat.bestOptionsForThisDevice)
                        try {
                            return create(Interpreter(model, Interpreter.Options().addDelegate(delegate)), delegate, "gpu",
                                inputSize, numClasses, normalized).also { it.loadMs = (System.nanoTime() - t0) / 1e6 }
                        } catch (e: Throwable) {
                            delegate.close()
                            throw e
                        }
                    }
                    reason = "GPU delegate not supported on this device"
                } catch (e: Throwable) {
                    reason = "GPU failed: ${e.message ?: e.javaClass.simpleName}"
                }
                if (backend == "gpu") throw IllegalStateException(reason)
            }
            val opts = Interpreter.Options().setNumThreads(threads).setUseXNNPACK(true)
            return create(Interpreter(model, opts), null, "cpu", inputSize, numClasses, normalized).also {
                it.loadMs = (System.nanoTime() - t0) / 1e6
                it.fallbackReason = reason
            }
        }

        private fun create(it: Interpreter, gpu: GpuDelegate?, backend: String, inputSize: Int, numClasses: Int,
                           normalized: Boolean): YoloDetector {
            try {
                val inShape = it.getInputTensor(0).shape() // [1, S, S, 3]
                require(inShape.size == 4 && inShape[1] == inputSize && inShape[2] == inputSize && inShape[3] == 3) {
                    "unexpected input shape ${inShape.contentToString()}"
                }
                val outShape = it.getOutputTensor(0).shape() // [1, 4+C, N] or [1, N, 4+C]
                require(outShape.size == 3) { "unexpected output shape ${outShape.contentToString()}" }
                val ch = 4 + numClasses
                val channelsFirst = when (ch) {
                    outShape[1] -> true
                    outShape[2] -> false
                    else -> throw IllegalArgumentException("output ${outShape.contentToString()} doesn't fit $numClasses classes")
                }
                val anchors = if (channelsFirst) outShape[2] else outShape[1]
                return YoloDetector(it, gpu, backend, inputSize, numClasses, anchors, channelsFirst, normalized)
                    .also { d -> d.selfCheck() }
            } catch (e: Throwable) {
                it.close()
                throw e
            }
        }

        private fun mapFile(file: File): MappedByteBuffer = FileInputStream(file).use { s ->
            s.channel.map(FileChannel.MapMode.READ_ONLY, 0, s.channel.size())
        }
    }
}
