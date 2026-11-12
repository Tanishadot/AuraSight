package com.tanishadot.aura_native

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.os.Build
import android.util.Log
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter

/**
 * Why the app's previous processes ended, readable from inside the app, so a
 * crash on a phone can be diagnosed without adb.
 *
 * - [ApplicationExitInfo] (Android 11+): the reason (native crash, Java crash,
 *   low memory kill, ANR...), and for native crashes (Android 12+) and ANRs the
 *   full tombstone / trace, which is saved to `files/crash/`.
 * - Uncaught Java exceptions on any thread are written to `files/crash/` before
 *   the default handler ends the process.
 */
internal object CrashReports {
    private const val TAG = "AuraCrash"
    private const val MAX_TRACE_BYTES = 512 * 1024

    fun dir(context: Context) = File(context.filesDir, "crash").also { it.mkdirs() }

    @Volatile
    private var installed = false

    fun installJavaRecorder(context: Context) {
        if (installed) return
        installed = true
        val dir = dir(context)
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, e ->
            try {
                val sw = StringWriter()
                e.printStackTrace(PrintWriter(sw))
                File(dir, "java_crash_${System.currentTimeMillis()}.txt")
                    .writeText("thread: ${thread.name}\n${DetectorHandler.memoryLine()}\n\n$sw")
            } catch (_: Throwable) {
            }
            previous?.uncaughtException(thread, e)
        }
    }

    /** The most recent process exits (newest first), with saved trace files. */
    fun recentExits(context: Context, max: Int = 5): List<Map<String, Any?>> {
        val out = mutableListOf<Map<String, Any?>>()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val infos = try {
                am.getHistoricalProcessExitReasons(context.packageName, 0, max)
            } catch (e: Throwable) {
                Log.w(TAG, "exit reasons unavailable", e)
                emptyList()
            }
            for (i in infos) out += describe(context, i)
        }
        dir(context).listFiles { f -> f.name.startsWith("java_crash_") }?.sortedByDescending { it.name }?.take(3)?.forEach {
            out += mapOf(
                "reason" to "java_crash_file",
                "timestampMs" to it.lastModified(),
                "description" to it.useLines { l -> l.drop(3).firstOrNull() },
                "tracePath" to it.path,
                "traceHead" to it.readText().take(8000),
            )
        }
        return out
    }

    @android.annotation.TargetApi(Build.VERSION_CODES.R)
    private fun describe(context: Context, i: ApplicationExitInfo): Map<String, Any?> {
        var tracePath: String? = null
        var head: String? = null
        try {
            i.traceInputStream?.use { s ->
                val bytes = s.readNBytesCompat(MAX_TRACE_BYTES)
                val f = File(dir(context), "exit_${i.timestamp}_${reasonName(i.reason)}.txt")
                if (!f.exists()) f.writeBytes(bytes)
                tracePath = f.path
                head = String(bytes, Charsets.UTF_8).lineSequence().take(120).joinToString("\n")
            }
        } catch (e: Throwable) {
            Log.w(TAG, "could not read exit trace", e)
        }
        return mapOf(
            "reason" to reasonName(i.reason),
            "status" to i.status,
            "description" to i.description,
            "timestampMs" to i.timestamp,
            "pssMb" to i.pss / 1024,
            "rssMb" to i.rss / 1024,
            "importance" to i.importance,
            "tracePath" to tracePath,
            "traceHead" to head,
        )
    }

    private fun java.io.InputStream.readNBytesCompat(max: Int): ByteArray {
        val buf = java.io.ByteArrayOutputStream()
        val chunk = ByteArray(16 * 1024)
        while (buf.size() < max) {
            val n = read(chunk, 0, minOf(chunk.size, max - buf.size()))
            if (n < 0) break
            buf.write(chunk, 0, n)
        }
        return buf.toByteArray()
    }

    fun reasonName(r: Int): String = when (r) {
        ApplicationExitInfo.REASON_ANR -> "anr"
        ApplicationExitInfo.REASON_CRASH -> "java_crash"
        ApplicationExitInfo.REASON_CRASH_NATIVE -> "native_crash"
        ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE -> "excessive_resource_usage"
        ApplicationExitInfo.REASON_EXIT_SELF -> "exit_self"
        ApplicationExitInfo.REASON_INITIALIZATION_FAILURE -> "initialization_failure"
        ApplicationExitInfo.REASON_LOW_MEMORY -> "low_memory_kill"
        ApplicationExitInfo.REASON_OTHER -> "other"
        ApplicationExitInfo.REASON_PERMISSION_CHANGE -> "permission_change"
        ApplicationExitInfo.REASON_SIGNALED -> "signaled"
        ApplicationExitInfo.REASON_USER_REQUESTED -> "user_requested"
        ApplicationExitInfo.REASON_USER_STOPPED -> "user_stopped"
        ApplicationExitInfo.REASON_DEPENDENCY_DIED -> "dependency_died"
        else -> "unknown_$r"
    }
}
