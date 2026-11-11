"""Offline text-to-speech engines.

Evaluation summary (see README for detail):
  * **SAPI 5 (Windows, via comtypes)** – default on Windows. Ships with the OS,
    zero downloads, ~instant start, supports true asynchronous speech with
    immediate purge/interrupt, rate and volume control.
  * **pyttsx3** – portable fallback (SAPI5 / NSSpeechSynthesizer / eSpeak).
    Interrupting is less reliable, so it is used only off-Windows.
  * **Piper** – more natural neural voices, but needs an extra ~60 MB voice per
    language and a separate player; supported as an *optional* engine when a
    ``piper`` executable and voice model are installed (see README).

Every engine is driven from **one** worker thread (COM objects are apartment
threaded), so ``speak`` / ``synthesize_to_file`` must be called from that thread
while ``interrupt`` may be called from anywhere.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Protocol

log = logging.getLogger("aurasight.speech")


class TTSEngine(Protocol):
    name: str

    def speak(self, text: str, rate: float, volume: float, voice_id: str | None) -> bool:
        """Speak synchronously. Returns False if interrupted."""

    def interrupt(self) -> None: ...

    def synthesize_to_file(self, text: str, path: Path, rate: float, volume: float,
                           voice_id: str | None) -> None: ...

    def voices(self) -> list[dict]: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
class SapiEngine:
    """Windows SAPI 5 through comtypes. Must be created on its worker thread."""

    name = "sapi5"
    SVSFlagsAsync = 1
    SVSFPurgeBeforeSpeak = 2

    def __init__(self):
        import comtypes
        import comtypes.client

        comtypes.CoInitialize()
        self._voice = comtypes.client.CreateObject("SAPI.SpVoice")
        self._default_output = self._voice.AudioOutputStream
        self._interrupt = threading.Event()

    @staticmethod
    def _sapi_rate(rate: float) -> int:
        # SAPI rate is -10..10 where +10 is ~3x and -10 is ~1/3x normal speed.
        return int(max(-10, min(10, round(10 * math.log(max(rate, 0.1)) / math.log(3)))))

    def _configure(self, rate: float, volume: float, voice_id: str | None) -> None:
        self._voice.Rate = self._sapi_rate(rate)
        self._voice.Volume = int(max(0, min(100, round(volume * 100))))
        if voice_id:
            for tok in self._voice.GetVoices():
                if tok.Id == voice_id:
                    self._voice.Voice = tok
                    break

    def speak(self, text: str, rate: float, volume: float, voice_id: str | None) -> bool:
        self._interrupt.clear()
        self._voice.AudioOutputStream = self._default_output
        self._configure(rate, volume, voice_id)
        self._voice.Speak(text, self.SVSFlagsAsync | self.SVSFPurgeBeforeSpeak)
        while not self._voice.WaitUntilDone(40):
            if self._interrupt.is_set():
                self._voice.Speak("", self.SVSFlagsAsync | self.SVSFPurgeBeforeSpeak)
                return False
        return not self._interrupt.is_set()

    def interrupt(self) -> None:
        self._interrupt.set()

    def synthesize_to_file(self, text, path, rate, volume, voice_id) -> None:
        import comtypes.client

        stream = comtypes.client.CreateObject("SAPI.SpFileStream")
        stream.Format.Type = 34  # SAFT22kHz16BitMono
        stream.Open(str(path), 3)  # SSFMCreateForWrite
        try:
            self._voice.AudioOutputStream = stream
            self._configure(rate, volume, voice_id)
            self._voice.Speak(text, 0)
        finally:
            stream.Close()
            self._voice.AudioOutputStream = self._default_output

    def voices(self) -> list[dict]:
        out = []
        for tok in self._voice.GetVoices():
            out.append({"id": tok.Id, "name": tok.GetDescription()})
        return out

    def close(self) -> None:
        try:
            self._voice.Speak("", self.SVSFlagsAsync | self.SVSFPurgeBeforeSpeak)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
class Pyttsx3Engine:
    """Portable fallback. Interrupts by stopping the run loop."""

    name = "pyttsx3"

    def __init__(self):
        import pyttsx3

        self._engine = pyttsx3.init()
        self._interrupted = False

    def _configure(self, rate, volume, voice_id):
        self._engine.setProperty("rate", int(190 * rate))
        self._engine.setProperty("volume", float(volume))
        if voice_id:
            self._engine.setProperty("voice", voice_id)

    def speak(self, text, rate, volume, voice_id) -> bool:
        self._interrupted = False
        self._configure(rate, volume, voice_id)
        self._engine.say(text)
        self._engine.runAndWait()
        return not self._interrupted

    def interrupt(self) -> None:
        self._interrupted = True
        try:
            self._engine.stop()
        except Exception:
            pass

    def synthesize_to_file(self, text, path, rate, volume, voice_id) -> None:
        self._configure(rate, volume, voice_id)
        self._engine.save_to_file(text, str(path))
        self._engine.runAndWait()

    def voices(self) -> list[dict]:
        return [{"id": v.id, "name": v.name} for v in self._engine.getProperty("voices")]

    def close(self) -> None:
        self.interrupt()


# --------------------------------------------------------------------------- #
class PiperEngine:
    """Optional neural TTS: ``piper`` CLI renders WAV, we play it interruptibly.

    Enabled when AURASIGHT_PIPER_EXE and AURASIGHT_PIPER_VOICE point to an
    installed piper executable and an ``.onnx`` voice model.
    """

    name = "piper"

    def __init__(self, exe: str, voice: str):
        if not shutil.which(exe) and not Path(exe).exists():
            raise FileNotFoundError(exe)
        if not Path(voice).exists():
            raise FileNotFoundError(voice)
        self.exe, self.voice = exe, voice
        self._interrupt = threading.Event()
        self._tmp = Path(tempfile.gettempdir()) / "aurasight_tts"
        self._tmp.mkdir(exist_ok=True)

    def synthesize_to_file(self, text, path, rate, volume, voice_id) -> None:
        length_scale = 1.0 / max(rate, 0.1)
        subprocess.run(
            [self.exe, "--model", self.voice, "--output_file", str(path),
             "--length_scale", f"{length_scale:.2f}", "--volume", f"{volume:.2f}"],
            input=text.encode("utf-8"), check=True, capture_output=True, timeout=60,
        )

    def speak(self, text, rate, volume, voice_id) -> bool:
        self._interrupt.clear()
        wav = self._tmp / f"utt_{threading.get_ident()}.wav"
        self.synthesize_to_file(text, wav, rate, volume, voice_id)
        return play_wav_interruptible(wav, self._interrupt)

    def interrupt(self) -> None:
        self._interrupt.set()

    def voices(self) -> list[dict]:
        return [{"id": self.voice, "name": Path(self.voice).stem}]

    def close(self) -> None:
        self.interrupt()


def wav_duration(path: Path) -> float:
    import wave

    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def play_wav_interruptible(path: Path, stop: threading.Event) -> bool:
    """Play a WAV file; return False if ``stop`` was set before it finished."""
    dur = wav_duration(path)
    if sys.platform == "win32":
        import winsound

        winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        end = time.monotonic() + dur + 0.05
        while time.monotonic() < end:
            if stop.wait(0.03):
                winsound.PlaySound(None, winsound.SND_PURGE)
                return False
        return True
    player = shutil.which("aplay") or shutil.which("afplay") or shutil.which("paplay")
    if not player:
        raise RuntimeError("No audio player (aplay/afplay/paplay) found")
    proc = subprocess.Popen([player, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while proc.poll() is None:
        if stop.wait(0.03):
            proc.terminate()
            return False
    return True


# --------------------------------------------------------------------------- #
def create_engine() -> TTSEngine:
    """Pick the best available offline engine for this machine."""
    exe, voice = os.environ.get("AURASIGHT_PIPER_EXE"), os.environ.get("AURASIGHT_PIPER_VOICE")
    if exe and voice:
        try:
            return PiperEngine(exe, voice)
        except Exception as e:
            log.warning("Piper requested but unavailable (%s); falling back", e)
    if sys.platform == "win32":
        try:
            return SapiEngine()
        except Exception as e:
            log.warning("SAPI unavailable (%s); trying pyttsx3", e)
    return Pyttsx3Engine()
