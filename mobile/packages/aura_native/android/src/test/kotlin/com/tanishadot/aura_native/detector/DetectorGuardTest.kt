package com.tanishadot.aura_native.detector

import java.io.File
import java.nio.file.Files
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class DetectorGuardTest {
    private val dir: File = Files.createTempDirectory("guard").toFile()

    @AfterTest
    fun cleanUp() {
        dir.deleteRecursively()
    }

    /** What a native crash looks like to the journal: a step begun and never ended. */
    private fun dieDuring(stage: String, backend: String) {
        DetectorGuard(dir, clock = { 1234L }).begin(stage, backend, "threads=4")
    }

    @Test
    fun `a clean run leaves nothing behind`() {
        val g = DetectorGuard(dir)
        assertEquals(7, g.step("cpu_warmup", "cpu") { 7 })
        assertNull(DetectorGuard(dir).interrupted)
        assertNull(DetectorGuard(dir).gpuBlockedReason)
    }

    @Test
    fun `a Java exception ends the step, it is not a crash`() {
        val g = DetectorGuard(dir)
        assertFailsWith<IllegalStateException> { g.step("gpu_interpreter_create", "gpu") { error("unsupported op") } }
        assertNull(DetectorGuard(dir).interrupted)
        assertNull(DetectorGuard(dir).gpuBlockedReason)
    }

    @Test
    fun `a process that died during a GPU step is reported and the GPU is switched off`() {
        DetectorGuard(dir).markGood("gpu")
        dieDuring("gpu_warmup", "gpu")
        val next = DetectorGuard(dir)
        val i = assertNotNull(next.interrupted)
        assertEquals("gpu_warmup", i.stage)
        assertEquals("gpu", i.backend)
        assertEquals(1234L, i.atMs)
        assertEquals("threads=4", i.detail)
        assertTrue(next.gpuBlockedReason!!.contains("gpu_warmup"))
        assertNull(next.lastGoodBackend, "a crashed configuration is no longer known to be good")
        // Reported once: the start after that is clean, but the GPU stays off.
        val third = DetectorGuard(dir)
        assertNull(third.interrupted)
        assertNotNull(third.gpuBlockedReason)
        third.clearGpuBlock()
        assertNull(DetectorGuard(dir).gpuBlockedReason)
    }

    @Test
    fun `a CPU crash is reported but does not switch the GPU off`() {
        dieDuring("first_inference", "cpu")
        val next = DetectorGuard(dir)
        assertEquals("first_inference", next.interrupted!!.stage)
        assertNull(next.gpuBlockedReason)
    }

    @Test
    fun `last good backend round-trips`() {
        val g = DetectorGuard(dir)
        assertNull(g.lastGoodBackend)
        g.markGood("cpu")
        assertEquals("cpu", DetectorGuard(dir).lastGoodBackend)
        g.clearGood()
        assertNull(DetectorGuard(dir).lastGoodBackend)
    }

    @Test
    fun `a damaged journal still counts as a crash`() {
        dir.mkdirs()
        File(dir, "inflight.txt").writeText("")
        val i = DetectorGuard(dir).interrupted!!
        assertEquals("unknown", i.stage)
        assertEquals("unknown", i.backend)
    }

    @Test
    fun `the gate admits one request at a time`() {
        val gate = RequestGate()
        assertTrue(gate.tryEnter())
        assertFalse(gate.tryEnter())
        assertTrue(gate.isBusy)
        gate.leave()
        assertTrue(gate.tryEnter())
    }

    @Test
    fun `shape check accepts the published model and rejects mismatches`() {
        // yolo11n_320_float32.tflite: input [1,320,320,3] float32, output [1,84,2100] float32
        assertEquals(true to 2100, YoloDetector.checkShapes(intArrayOf(1, 320, 320, 3), intArrayOf(1, 84, 2100), 320, 80))
        assertEquals(false to 2100, YoloDetector.checkShapes(intArrayOf(1, 320, 320, 3), intArrayOf(1, 2100, 84), 320, 80))
        assertFailsWith<IllegalArgumentException> {
            YoloDetector.checkShapes(intArrayOf(1, 640, 640, 3), intArrayOf(1, 84, 8400), 320, 80)
        }
        assertFailsWith<IllegalArgumentException> {
            YoloDetector.checkShapes(intArrayOf(1, 3, 320, 320), intArrayOf(1, 84, 2100), 320, 80) // NCHW
        }
        assertFailsWith<IllegalArgumentException> {
            YoloDetector.checkShapes(intArrayOf(1, 320, 320, 3), intArrayOf(1, 85, 2100), 320, 80)
        }
    }

    @Test
    fun `load keys differ by backend`() {
        val a = LoadKey("/m.tflite", 10, 1, 320, 80, true, "cpu", 4)
        assertEquals(a, a.copy())
        assertFalse(a == a.copy(backend = "auto"))
    }
}
