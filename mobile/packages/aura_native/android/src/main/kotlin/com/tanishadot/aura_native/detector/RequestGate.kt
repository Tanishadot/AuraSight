package com.tanishadot.aura_native.detector

import java.util.concurrent.atomic.AtomicBoolean

/**
 * Admits one inference request at a time. A request that arrives while
 * another is queued or running is refused ("busy") instead of being queued,
 * so camera frames can never pile up in native memory behind a slow model.
 */
class RequestGate {
    private val busy = AtomicBoolean(false)

    fun tryEnter(): Boolean = busy.compareAndSet(false, true)

    fun leave() {
        busy.set(false)
    }

    val isBusy: Boolean get() = busy.get()
}

/** Identifies a loaded model configuration, so an identical `load` reuses it instead of reloading. */
data class LoadKey(
    val path: String,
    val bytes: Long,
    val modified: Long,
    val inputSize: Int,
    val numClasses: Int,
    val normalized: Boolean,
    val backend: String,
    val threads: Int,
)
