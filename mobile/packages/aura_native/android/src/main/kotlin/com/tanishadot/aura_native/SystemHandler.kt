package com.tanishadot.aura_native

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.os.Process
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel

/**
 * Device state (`aurasight/system`): thermal and memory for the adaptive
 * detection rate, plus battery and app memory for benchmark reports.
 */
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
        val appPssKb = try {
            am.getProcessMemoryInfo(intArrayOf(Process.myPid()))[0].totalPss
        } catch (e: Exception) {
            -1
        }
        // Sticky broadcast: no receiver registration needed.
        val battery = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val tempTenths = battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE) ?: Int.MIN_VALUE
        val plugged = (battery?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        result.success(
            mapOf(
                "thermal" to thermal,
                "availMemMb" to (mi.availMem / (1024 * 1024)).toInt(),
                "lowMemory" to mi.lowMemory,
                "appMemMb" to if (appPssKb >= 0) appPssKb / 1024 else null,
                "batteryTempC" to if (tempTenths != Int.MIN_VALUE) tempTenths / 10.0 else null,
                "batteryPct" to bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY),
                "batteryCurrentUa" to bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW),
                "charging" to plugged,
                "device" to "${Build.MANUFACTURER} ${Build.MODEL}",
                "os" to "Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})",
                "soc" to if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else Build.HARDWARE,
            ),
        )
    }
}
