package com.tanishadot.aura_native

import android.content.Context
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.speech.tts.Voice
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.util.Locale
import java.util.concurrent.atomic.AtomicLong

/**
 * Android TextToSpeech behind the `aurasight/tts` channel.
 *
 * `speak` completes when the utterance ends: true if it finished, false if it
 * was stopped. The Dart speech queue decides what plays and when, so one
 * utterance at a time is all this needs (QUEUE_FLUSH).
 * Only offline voices are used, so speech works with no connection.
 */
internal class TtsHandler(private val context: Context) : MethodChannel.MethodCallHandler {
    private val main = Handler(Looper.getMainLooper())
    private var tts: TextToSpeech? = null
    private var ready = false
    private val waitingForInit = mutableListOf<MethodChannel.Result>()
    private val pending = HashMap<String, MethodChannel.Result>()
    private val ids = AtomicLong()

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "init" -> init(result)
            "voices" -> if (ready) result.success(voiceList()) else result.success(emptyList<Any>())
            "speak" -> if (ready) speak(call, result) else result.error("tts_not_ready", "Text-to-speech is not ready", null)
            "stop" -> {
                tts?.stop()
                result.success(null)
            }
            else -> result.notImplemented()
        }
    }

    private fun init(result: MethodChannel.Result) {
        if (ready) {
            result.success(info())
            return
        }
        waitingForInit += result
        if (tts != null) return // already starting
        val engine = TextToSpeech(context) { status ->
            main.post {
                ready = status == TextToSpeech.SUCCESS
                val waiting = waitingForInit.toList()
                waitingForInit.clear()
                for (r in waiting) {
                    if (ready) r.success(info()) else r.error("tts_init", "The text-to-speech engine failed to start", null)
                }
                if (!ready) {
                    tts?.shutdown()
                    tts = null
                }
            }
        }
        engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String) {}
            override fun onDone(utteranceId: String) = complete(utteranceId, true)
            override fun onStop(utteranceId: String, interrupted: Boolean) = complete(utteranceId, false)

            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String) = fail(utteranceId, "Speech failed")
            override fun onError(utteranceId: String, errorCode: Int) = fail(utteranceId, "Speech failed (code $errorCode)")
        })
        tts = engine
    }

    private fun complete(id: String, finished: Boolean) = main.post { pending.remove(id)?.success(finished) }

    private fun fail(id: String, message: String) = main.post { pending.remove(id)?.error("tts_speak", message, null) }

    private fun info(): Map<String, Any> =
        mapOf("engine" to (tts?.defaultEngine ?: "unknown"), "offlineVoices" to offlineVoices().size)

    private fun offlineVoices(): List<Voice> = try {
        tts?.voices.orEmpty().filter {
            !it.isNetworkConnectionRequired && !it.features.contains(TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED)
        }
    } catch (e: Exception) {
        emptyList() // some engines throw while voice data is updating
    }

    private fun voiceList(): List<Map<String, Any>> = try {
        tts?.voices.orEmpty().map {
            mapOf(
                "id" to it.name,
                "name" to it.name,
                "language" to it.locale.toLanguageTag(),
                "offline" to (!it.isNetworkConnectionRequired &&
                    !it.features.contains(TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED)),
            )
        }.sortedBy { it["language"] as String }
    } catch (e: Exception) {
        emptyList()
    }

    /** The requested voice if it is offline, else the best offline voice for the phone's language. */
    private fun pickVoice(voiceId: String?): Voice? {
        val offline = offlineVoices()
        if (voiceId != null) offline.firstOrNull { it.name == voiceId }?.let { return it }
        val current = tts?.voice
        if (current != null && offline.any { it.name == current.name }) return current
        val lang = Locale.getDefault().language
        return offline.filter { it.locale.language == lang }.maxByOrNull { it.quality } ?: offline.maxByOrNull { it.quality }
    }

    private fun speak(call: MethodCall, result: MethodChannel.Result) {
        val engine = tts ?: return result.error("tts_not_ready", "Text-to-speech is not ready", null)
        val text = call.argument<String>("text") ?: return result.error("bad_args", "text is required", null)
        val rate = call.argument<Double>("rate") ?: 1.0
        val volume = call.argument<Double>("volume") ?: 1.0
        pickVoice(call.argument<String>("voiceId"))?.let { if (engine.voice?.name != it.name) engine.voice = it }
        engine.setSpeechRate(rate.toFloat())
        val params = Bundle().apply { putFloat(TextToSpeech.Engine.KEY_PARAM_VOLUME, volume.toFloat()) }
        val id = "aura-" + ids.incrementAndGet()
        pending[id] = result
        if (engine.speak(text, TextToSpeech.QUEUE_FLUSH, params, id) != TextToSpeech.SUCCESS) {
            pending.remove(id)
            result.error("tts_speak", "Could not start speaking", null)
        }
    }

    fun shutdown() {
        tts?.stop()
        tts?.shutdown()
        tts = null
        ready = false
        pending.values.forEach { it.success(false) }
        pending.clear()
    }
}
