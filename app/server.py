import os
import asyncio
import base64
import json
import logging
from io import BytesIO
import wave
import audioop
import time
import math
import re
import contextlib
from pathlib import Path
import zipfile

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, HTTPException
from fastapi.responses import PlainTextResponse, FileResponse
from starlette.websockets import WebSocketState

# OpenAI SDK (v1+) for Whisper
try:
    from openai import OpenAI
    _openai_client = OpenAI()
except Exception:
    _openai_client = None

log = logging.getLogger("hid.media")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

app = FastAPI(title="Horizon AI Dispatcher (Track B)")

# ---------- Config ----------
SAMPLE_RATE_IN = 8000
BYTES_PER_SAMPLE = 2

# Recording / debug
RECORD_CALL = os.getenv("RECORD_CALL", "0") == "1"
RECORD_DIR = os.getenv("RECORD_DIR", "/tmp/recordings")
Path(RECORD_DIR).mkdir(parents=True, exist_ok=True)

# Endpointing
STT_CHUNK_S = float(os.getenv("STT_CHUNK_S", "1.4"))
LINEAR_BYTES_TARGET = int(SAMPLE_RATE_IN * BYTES_PER_SAMPLE * STT_CHUNK_S)
STT_FLUSH_ON_SILENCE = os.getenv("STT_FLUSH_ON_SILENCE", "1") == "1"
STT_MIN_MS = int(os.getenv("STT_MIN_MS", "600"))
VAD_RMS_THRESHOLD = int(os.getenv("VAD_RMS_THRESHOLD", "700"))
VAD_QUIET_MS = int(os.getenv("VAD_QUIET_MS", "250"))
STT_RMS_GATE = int(os.getenv("STT_RMS_GATE", "350"))
STT_MIN_GAP_BETWEEN_CALLS_MS = int(os.getenv("STT_MIN_GAP_BETWEEN_CALLS_MS", "800"))

# TTS
TTS_GAIN_DB = float(os.getenv("TTS_GAIN_DB", "6"))
TTS_PEAK_TARGET = float(os.getenv("TTS_PEAK_TARGET", "0.78"))
TTS_MIN_GAP_MS = int(os.getenv("TTS_MIN_GAP_MS", "2400"))
TTS_MIN_GAP_AFTER_BARGE_MS = int(os.getenv("TTS_MIN_GAP_AFTER_BARGE_MS", "1200"))
TTS_PAD_PRE_MS = int(os.getenv("TTS_PAD_PRE_MS", "600"))
TTS_PAD_POST_MS = int(os.getenv("TTS_PAD_POST_MS", "500"))
ELEVEN_FMT = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "pcm_16000").lower()

ELEVEN_LATENCY_MODE = os.getenv("ELEVEN_LATENCY_MODE", "1")
ELEVEN_SPEED = float(os.getenv("ELEVEN_SPEED", "0.95"))
ELEVEN_STABILITY = float(os.getenv("ELEVEN_STABILITY", "0.5"))
ELEVEN_SIMILARITY = float(os.getenv("ELEVEN_SIMILARITY", "0.75"))
ELEVEN_STYLE = float(os.getenv("ELEVEN_STYLE", "0.35"))
ELEVEN_SPEAKER_BOOST = os.getenv("ELEVEN_SPEAKER_BOOST", "1") == "1"

# Turn-taking
FIRST_REPLY_WAIT_FOR_QUIET = os.getenv("FIRST_REPLY_WAIT_FOR_QUIET", "0") == "1"
ENABLE_BARGE_IN = os.getenv("ENABLE_BARGE_IN", "1") == "1"
BARGE_GRACE_MS = int(os.getenv("BARGE_GRACE_MS", "900"))
FIRST_REPLY_NO_BARGE = os.getenv("FIRST_REPLY_NO_BARGE", "1") == "1"

# Diagnostics
LOG_TTS_DEBUG = os.getenv("LOG_TTS_DEBUG", "0") == "1"
AUTO_TEST_TTS = os.getenv("AUTO_TEST_TTS", "0") == "0"

WHISPER_PROMPT = os.getenv("WHISPER_PROMPT", "Towing and roadside assistance. Year make model, location, issue, urgency. Keep transcripts concise.")
TWILIO_RECORD_CALL = os.getenv("TWILIO_RECORD_CALL", "0") == "1"

# ---------- Utils ----------
def _public_ws_url(request: Request) -> str:
    host = request.headers.get("x-forwarded-host") or request.url.hostname
    scheme = "wss"
    return f"{scheme}://{host}/twilio/media"

def _public_host(request: Request) -> str:
    return request.headers.get("x-forwarded-host") or request.url.hostname

def _rms(pcm16: bytes) -> float:
    try:
        return audioop.rms(pcm16, 2)
    except Exception:
        return 0.0

def _silence_ulaw(ms: int) -> bytes:
    samples = int(8000 * ms / 1000)
    return bytes([0xFF]) * samples  # μ-law silence byte

def _to_wav(path, pcm16: bytes, sr: int = 8000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(pcm16)

# ---------- HTTP ----------
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/twilio/voice", response_class=PlainTextResponse)
async def twilio_voice_webhook(request: Request):
    ws_url = _public_ws_url(request)
    start_record = "<Start><Recording/></Start>" if TWILIO_RECORD_CALL else ""
    twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
  {start_record}
  <Say>Horizon Road Rescue dispatcher connected. Please speak after the tone.</Say>
  <Connect><Stream url="{ws_url}" /></Connect>
</Response>'''
    return Response(content=twiml, media_type="application/xml")

# ---------- Recording manager ----------
class Recorder:
    def __init__(self, call_sid: str):
        self.call_sid = call_sid or f"call-{int(time.time())}"
        self.in_ulaw = bytearray()
        self.out_ulaw = bytearray()
        self.stt_lines = []
        self.tts_lines = []

    def add_in_ulaw_b64(self, b64: str):
        try:
            self.in_ulaw.extend(base64.b64decode(b64))
        except Exception:
            pass

    def add_out_ulaw_chunk(self, ulaw: bytes):
        self.out_ulaw.extend(ulaw)

    def add_stt(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.stt_lines.append(f"[{ts}] {text}")

    def add_tts(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.tts_lines.append(f"[{ts}] {text}")

    def finalize_files(self) -> dict:
        def ulaw_to_pcm16(ulaw: bytes) -> bytes:
            try:
                return audioop.ulaw2lin(ulaw, 2)
            except Exception:
                return b""

        in_pcm = ulaw_to_pcm16(bytes(self.in_ulaw))
        out_pcm = ulaw_to_pcm16(bytes(self.out_ulaw))

        base = Path(RECORD_DIR) / self.call_sid
        in_path = base.with_suffix(".in.wav")
        out_path = base.with_suffix(".out.wav")
        stt_path = base.with_suffix(".stt.txt")
        tts_path = base.with_suffix(".tts.txt")
        bundle_path = base.with_suffix(".zip")

        _to_wav(in_path, in_pcm, 8000)
        _to_wav(out_path, out_pcm, 8000)
        with open(stt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.stt_lines))
        with open(tts_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.tts_lines))

        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in (in_path, out_path, stt_path, tts_path):
                zf.write(p, arcname=p.name)

        return {"in_wav": str(in_path), "out_wav": str(out_path),
                "stt": str(stt_path), "tts": str(tts_path), "zip": str(bundle_path)}

# ---------- STT (non-blocking) ----------
IGNORE_PATTERNS = [r"www\\.", r"http[s]?://", r"FEMA\\.gov", r"un\\.org", r"For more UN videos", r"For more information"]

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
def _apply_gain_linear(frames: bytes, sampwidth: int, gain_db: float) -> bytes:
    if gain_db <= 0: return frames
    factor = pow(10.0, gain_db / 20.0)
    try: return audioop.mul(frames, sampwidth, factor)
    except Exception: return frames

def _parse_wav(data: bytes):
    with wave.open(BytesIO(data), "rb") as wf:
        return wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.readframes(wf.getnframes())

def _peak_normalize(pcm16: bytes, target_ratio: float) -> bytes:
    if not pcm16: return pcm16
    peak = 1
    for i in range(0, len(pcm16), 2):
        s = int.from_bytes(pcm16[i:i+2], "little", signed=True)
        if abs(s) > peak: peak = abs(s)
    max_allowed = int(32767 * target_ratio)
    if peak <= 0 or peak <= max_allowed: return pcm16
    scale = max_allowed / float(peak)
    out = bytearray(len(pcm16))
    for i in range(0, len(pcm16), 2):
        s = int.from_bytes(pcm16[i:i+2], "little", signed=True)
        s = int(s * scale)
        if s > 32767: s = 32767
        if s < -32768: s = -32768
        out[i:i+2] = s.to_bytes(2, "little", signed=True)
    return bytes(out)

def _to_ulaw_8k_from_linear(frames: bytes, sw: int, sr: int, nch: int, gain_db: float = 0.0) -> bytes:
    if sw != 2: frames = audioop.lin2lin(frames, sw, 2); sw = 2
    if nch == 2: frames = audioop.tomono(frames, 2, 0.5, 0.5); nch = 1
    if sr != 8000: frames, _ = audioop.ratecv(frames, 2, 1, sr, 8000, None); sr = 8000
    frames = _peak_normalize(frames, TTS_PEAK_TARGET)
    if gain_db > 0: frames = _apply_gain_linear(frames, 2, gain_db)
    return audioop.lin2ulaw(frames, 2)

async def elevenlabs_tts_bytes(text: str) -> tuple[bytes, str]:
    api_key = os.getenv("ELEVENLABS_API_KEY"); voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    if not api_key or not voice_id: raise RuntimeError("Missing ELEVENLABS env vars")
    fmt = ELEVEN_FMT
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    params = {"optimize_streaming_latency": os.getenv("ELEVEN_LATENCY_MODE", "1"), "output_format": fmt}
    accept = "audio/wav" if fmt == "wav" else "audio/pcm"
    headers = {"xi-api-key": api_key, "accept": accept, "content-type": "application/json"}
    voice_settings = {
        "stability": float(os.getenv("ELEVEN_STABILITY", "0.5")),
        "similarity_boost": float(os.getenv("ELEVEN_SIMILARITY", "0.75")),
        "style": float(os.getenv("ELEVEN_STYLE", "0.35")),
        "use_speaker_boost": os.getenv("ELEVEN_SPEAKER_BOOST", "1") == "1",
        "speed": float(os.getenv("ELEVEN_SPEED", "0.95")),
    }
    body = {"text": text, "model_id": os.getenv("ELEVENLABS_MODEL_ID", "eleven_monolingual_v1"),
            "voice_settings": voice_settings}
    async with httpx.AsyncClient(timeout=httpx.Timeout(35.0)) as client:
        r = await client.post(url, headers=headers, params=params, json=body)
        if r.status_code != 200:
            raise RuntimeError(f"ElevenLabs TTS failed: {r.status_code} {r.text[:200]}")
        return r.content, fmt

def convert_elevenlabs_to_ulaw8k(data: bytes, fmt: str, gain_db: float) -> bytes:
    if fmt == "wav":
        nch, sw, sr, frames = _parse_wav(data)
    elif fmt == "pcm_16000":
        nch, sw, sr, frames = 1, 2, 16000, data
    else:
        try: nch, sw, sr, frames = _parse_wav(data)
        except Exception: nch, sw, sr, frames = 1, 2, 16000, data
    ulaw = _to_ulaw_8k_from_linear(frames, sw, sr, nch, gain_db=gain_db)
    pre = _silence_ulaw(TTS_PAD_PRE_MS); post = _silence_ulaw(TTS_PAD_POST_MS)
    return pre + ulaw + post

# ---------- WS ----------
@app.websocket("/twilio/media")
async def twilio_media_ws(ws: WebSocket):
    await ws.accept(subprotocol="audio")
    frames = 0; speaker = None; recorder = None; call_sid = None

    def on_text(text: str, pcm: bytes):
        nonlocal speaker, recorder
        if recorder and RECORD_CALL:
            recorder.add_stt(text)
        prompt = "Thanks, I hear you. What is the vehicle year, make, and model?"
        asyncio.create_task(speaker.enqueue_text(prompt, protect=False))

    stt = STTBuffer(on_text=on_text)  # recorder set after start

    try:
        while True:
            try:
                message = await ws.receive()
            except WebSocketDisconnect:
                break
            except RuntimeError:
                break

            if message["type"] == "websocket.receive":
                payload_text = message.get("text") or (message.get("bytes") or b"").decode("utf-8","ignore")
                if not payload_text: continue
                try:
                    data = json.loads(payload_text)
                except Exception:
                    continue

                event = data.get("event")
                if event == "start":
                    stream_sid = data.get("start", {}).get("streamSid")
                    call_sid = data.get("start", {}).get("callSid") or stream_sid
                    recorder = Recorder(call_sid) if RECORD_CALL else None
                    stt.recorder = recorder
                    speaker = TTSSpeaker(ws, stream_sid, recorder)
                    asyncio.create_task(speaker.beep())
                    if AUTO_TEST_TTS:
                        asyncio.create_task(speaker.enqueue_text("System check. You should hear this test phrase.", protect=True))
                elif event == "media":
                    payload_b64 = data.get("media", {}).get("payload", "")
                    stt.add_ulaw_b64(payload_b64)
                    frames += 1
                elif event == "stop":
                    stt.finish(); break
    finally:
        if speaker: await speaker.close()
        if recorder and RECORD_CALL:
            paths = recorder.finalize_files()
            log.info(f"[record] saved: {paths}")
        try:
            if ws.application_state == WebSocketState.CONNECTED:
                await ws.close()
        except Exception:
            pass

# ---------- Download endpoints ----------
@app.get("/recordings/{name}")
async def get_recording(name: str):
    base = Path(RECORD_DIR) / name
    if not base.exists() or not base.is_file():
        for suf in (".in.wav", ".out.wav", ".zip", ".stt.txt", ".tts.txt"):
            cand = base.with_suffix(suf)
            if cand.exists():
                base = cand; break
    if not base.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(base))

@app.get("/debug/{call_id}")
async def get_debug_bundle(call_id: str):
    base = Path(RECORD_DIR) / call_id
    zip_path = base.with_suffix(".zip")
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(zip_path))
