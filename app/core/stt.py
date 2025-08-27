# STTBuffer extracted from server.py
import os, asyncio, base64, json, logging, time
from io import BytesIO
import wave, audioop, contextlib
from pathlib import Path
from app.config import *
from .recorder import Recorder

_openai_client = None
def set_openai_client(c):
    global _openai_client
    _openai_client = c

class STTBuffer:
    def __init__(self, on_text=None, recorder: Recorder | None = None):
        self.linear_pcm = bytearray()
        self.inflight = False
        self.on_text = on_text
        self.last_voice_at = time.monotonic()
        self.last_whisper_at = 0.0
        self.loop = asyncio.get_running_loop()
        self.recorder = recorder

    def add_ulaw_b64(self, payload_b64: str):
        if self.recorder and RECORD_CALL:
            self.recorder.add_in_ulaw_b64(payload_b64)

        try:
            ulaw = base64.b64decode(payload_b64)
            lin = audioop.ulaw2lin(ulaw, BYTES_PER_SAMPLE)
        except Exception as e:
            log.warning(f"Failed ulaw decode: {e}")
            return
        self.linear_pcm.extend(lin)

        if STT_FLUSH_ON_SILENCE:
            rms = _rms(lin); now = time.monotonic()
            if rms > VAD_RMS_THRESHOLD:
                self.last_voice_at = now
            quiet_ms = (now - self.last_voice_at) * 1000.0
            if quiet_ms >= VAD_QUIET_MS and len(self.linear_pcm) >= max(2, int(SAMPLE_RATE_IN*BYTES_PER_SAMPLE*STT_MIN_MS/1000)):
                self.flush_to_whisper_async(); return

        if len(self.linear_pcm) >= LINEAR_BYTES_TARGET:
            self.flush_to_whisper_async()

    def finish(self):
        if self.linear_pcm and not self.inflight:
            self.flush_to_whisper_async()

    def _to_wav_bytes(self, pcm_bytes: bytes) -> bytes:
        buf = BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(BYTES_PER_SAMPLE); wf.setframerate(SAMPLE_RATE_IN)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    def flush_to_whisper_async(self):
        if not self.linear_pcm or self.inflight:
            return
        chunk = bytes(self.linear_pcm); self.linear_pcm.clear()
        rms_total = _rms(chunk)
        now = time.monotonic()
        if rms_total < STT_RMS_GATE:
            log.info(f"[stt] skip (quiet) rms={rms_total}")
            return
        if (now - self.last_whisper_at) * 1000.0 < STT_MIN_GAP_BETWEEN_CALLS_MS:
            log.info("[stt] skip (gap)")
            return

        self.inflight = True
        self.last_whisper_at = now
        self.loop.create_task(self._do_whisper(chunk))

    async def _do_whisper(self, chunk: bytes):
        try:
            if _openai_client is None:
                log.info("[stt] (dry-run) bytes=%d", len(chunk)); return
            def _call_whisper(bytes_chunk: bytes):
                bio = BytesIO(self._to_wav_bytes(bytes_chunk)); bio.name = "clip.wav"; bio.seek(0)
                return _openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=bio,
                    response_format="json",
                    language="en",
                    prompt=WHISPER_PROMPT,
                )
            resp = await asyncio.to_thread(_call_whisper, chunk)
            text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None)
            if text and len(text.strip()) >= 2 and not any(re.search(p, text, re.I) for p in IGNORE_PATTERNS):
                log.info(f"[stt] {text}")
                if self.recorder and RECORD_CALL:
                    self.recorder.add_stt(text)
                if self.on_text:
                    try: self.on_text(text, chunk)
                    except Exception as e: log.warning(f"[stt] on_text error: {e}")
            else:
                log.info("[stt] (filtered or empty)")
        except Exception as e:
            log.warning(f"[stt] Whisper error: {e}")
        finally:
            self.inflight = False

# ---------- TTS helpers ----------

