package com.tanishadot.aura_native.detector

import java.io.File

/**
 * Crash journal for the detector's native steps.
 *
 * A crash inside native code (the LiteRT GPU delegate, a vendor GPU driver,
 * XNNPACK) kills the process with a signal; no Kotlin `catch` sees it. So
 * before each risky step the step's name is written to a small file, and it
 * is deleted when the step returns (normally or with an exception). If the
 * file is still there when the next process starts, the previous process
 * died inside that step. That is reported to the user, and when it was the
 * GPU the GPU is switched off for the detector until the user clears it.
 *
 * Writes go to the page cache, which survives a process crash, so no fsync
 * is needed. Plain text (no org.json) so it runs in JVM unit tests.
 */
class DetectorGuard(
    private val dir: File,
    private val clock: () -> Long = System::currentTimeMillis,
    private val log: (String) -> Unit = {},
) {
    data class Interrupted(val stage: String, val backend: String, val atMs: Long, val detail: String)

    private val inflight = File(dir, "inflight.txt")
    private val gpuBlock = File(dir, "gpu_blocked.txt")
    private val lastGood = File(dir, "last_good.txt")

    /** The step the previous process died in, if it did (read once, at construction). */
    val interrupted: Interrupted?

    init {
        dir.mkdirs()
        interrupted = read()
        interrupted?.let { i ->
            log("previous process died during ${i.stage} (${i.backend}) ${i.detail}")
            lastGood.delete() // the last configuration is no longer known to be good
            if (i.backend == "gpu") {
                blockGpu("AuraSight closed while the GPU was running the detector (step: ${i.stage}).")
            }
            inflight.delete()
        }
    }

    fun begin(stage: String, backend: String, detail: String = "") {
        log("begin $stage ($backend) $detail")
        inflight.writeText("$stage\n$backend\n${clock()}\n${detail.replace('\n', ' ')}\n")
    }

    fun end() {
        inflight.delete()
    }

    /** Runs [block] journalled as [stage]. A Java exception ends the step too; only a process death leaves it. */
    fun <T> step(stage: String, backend: String, detail: String = "", block: () -> T): T {
        begin(stage, backend, detail)
        try {
            return block()
        } finally {
            end()
        }
    }

    val gpuBlockedReason: String? get() = gpuBlock.takeIf { it.isFile }?.readText()?.trim()?.ifEmpty { null }

    fun blockGpu(reason: String) {
        gpuBlock.writeText(reason)
    }

    fun clearGpuBlock() {
        gpuBlock.delete()
    }

    /** The backend whose load + self-test last completed, or null (then the app doesn't auto-load). */
    val lastGoodBackend: String? get() = lastGood.takeIf { it.isFile }?.readText()?.trim()?.ifEmpty { null }

    fun markGood(backend: String) {
        lastGood.writeText(backend)
    }

    fun clearGood() {
        lastGood.delete()
    }

    private fun read(): Interrupted? {
        if (!inflight.isFile) return null
        val l = inflight.readLines()
        return Interrupted(
            stage = l.getOrNull(0)?.ifBlank { null } ?: "unknown",
            backend = l.getOrNull(1)?.ifBlank { null } ?: "unknown",
            atMs = l.getOrNull(2)?.toLongOrNull() ?: 0L,
            detail = l.getOrNull(3) ?: "",
        )
    }
}
