package com.tanishadot.aura_native

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.PowerManager
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel

/** Thermal and memory state for the adaptive detection rate (`aurasight/system`). */
internal class SystemHandler(private val context: Context) : MethodChannel.MethodCallHandler {
    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        if (call.method != "snapshot") return result.notImplemented()
        val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        val thermal = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            when (pm.currentThermalStatus) {
                PowerManager.THERMAL_STATUS_NONE, PowerManager.THERMAL_STATUS_LIGHT -> "nominal"
                PowerManager.THERMAL_STATUS_MODERATE -> "fair"
                PowerManager.THERMAL_STATUS_SEVERE -> "serious"
                else -> "critical"
            }
        } else {
            "nominal"
        }
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        result.success(
            mapOf(
                "thermal" to thermal,
                "availMemMb" to (mi.availMem / (1024 * 1024)).toInt(),
                "lowMemory" to mi.lowMemory,
            ),
        )
    }
}
