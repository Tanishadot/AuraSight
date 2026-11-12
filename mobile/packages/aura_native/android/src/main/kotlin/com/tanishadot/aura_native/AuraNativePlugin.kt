package com.tanishadot.aura_native

import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodChannel

/** Registers AuraSight's native channels (see app/lib/platform/channels.dart). */
class AuraNativePlugin : FlutterPlugin {
    private val channels = mutableListOf<MethodChannel>()
    private var tts: TtsHandler? = null

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        val ctx = binding.applicationContext
        val speech = TtsHandler(ctx)
        tts = speech
        channels += MethodChannel(binding.binaryMessenger, "aurasight/tts").also { it.setMethodCallHandler(speech) }
        channels += MethodChannel(binding.binaryMessenger, "aurasight/system").also {
            it.setMethodCallHandler(SystemHandler(ctx))
        }
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channels.forEach { it.setMethodCallHandler(null) }
        channels.clear()
        tts?.shutdown()
        tts = null
    }
}
