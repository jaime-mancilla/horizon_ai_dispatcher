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

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import PlainTextResponse
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

# Endpointing (fast first reply)
STT_CHUNK_S = float(os.getenv("STT_CHUNK_S", "1.6"))
LINEAR_BYTES_TARGET = int(SAMPLE_RATE_IN * BYTES_PER_SAMPLE * STT_CHUNK_S)
STT_FLUSH_ON_SILENCE = os.getenv("STT_FLUSH_ON_SILENCE", "1") == "1"
STT_MIN_MS = int(os.getenv("STT_MIN_MS", "700"))
VAD_RMS_THRESHOLD = int(os.getenv("VAD_RMS_THRESHOLD", "500"))
VAD_QUIET_MS = int(os.getenv("VAD_QUIET_MS", "250"))

# TTS loudness/flow
TTS_GAIN_DB = float(os.getenv("TTS_GAIN_DB", "8"))
TTS_PEAK_TARGET = float(os.getenv("TTS_PEAK_TARGET", "0.82"))
TTS_MIN_GAP_MS = int(os.getenv("TTS_MIN_GAP_MS", "3200"))
TTS_MIN_GAP_AFTER_BARGE_MS = int(os.getenv("TTS_MIN_GAP_AFTER_BARGE_MS", "900"))
TTS_PAD_PRE_MS = int(os.getenv("TTS_PAD_PRE_MS", "220"))
TTS_PAD_POST_MS = int(os.getenv("TTS_PAD_POST_MS", "260"))
ELEVEN_FMT = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "pcm_16000").lower()

# Interaction behavior
FIRST_REPLY_WAIT_FOR_QUIET = os.getenv("FIRST_REPLY_WAIT_FOR_QUIET", "0") == "1"  # default off
ENABLE_BARGE_IN = os.getenv("ENABLE_BARGE_IN", "1") == "1"
BARGE_GRACE_MS = int(os.getenv("BARGE_GRACE_MS", "250"))  # minimum playback before barge can cancel

WHISPER_PROMPT = os.getenv("WHISPER_PROMPT", "Towing and roadside assistance. Year make model, location, issue, urgency.")

# ---------- Utils ----------
def _public_ws_url(request: Request) -> str:
    host = request.headers.get("x-forwarded-host") or request.url.hostname
    scheme = "wss"
    return f"{scheme}://{host}/twilio/media"

def _rms(pcm16: bytes) -> float:
    try:
        return audioop.rms(pcm16, 2)
    except Exception:
        return 0.0

def _silence_ulaw(ms: int) -> bytes:
    samples = int(8000 * ms / 1000)
    return bytes([0xFF]) * samples  # μ-law silence byte

# ---------- HTTP ----------
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/twilio/voice", response_class=PlainTextResponse)
async def twilio_voice_webhook(request: Request):
    ws_url = _public_ws_url(request)
    log.info(f"Returning TwiML with Media Stream URL: {ws_url}")
    twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Horizon Road Rescue dispatcher connected. Please speak after the tone.</Say>
  <Connect><Stream url="{ws_url}" /></Connect>
</Response>'''
    return Response(content=twiml, media_type="application/xml")

# ---------- STT with adaptive endpointing ----------
IGNORE_PATTERNS = [r"www\\.", r"http[s]?://", r"FEMA\\.gov", r"un\\.org", r"For more UN videos", r"For more information"]

class STTBuffer:
    def __init__(self, on_text=None):
        self.linear_pcm = bytearray()
        self.inflight = False
        self.on_text = on_text
        self.last_voice_at = time.monotonic()

    def add_ulaw_b64(self, payload_b64: str):
        try:
            ulaw = base64.b64decode(payload_b64)
            lin = audioop.ulaw2lin(ulaw, BYTES_PER_SAMPLE)
        except Exception as e:
            log.warning(f"Failed ulaw decode: {e}")
            return
        self.linear_pcm.extend(lin)

        # Flush earlier if we detect enough + silence
        if STT_FLUSH_ON_SILENCE:
            rms = _rms(lin); now = time.monotonic()
            if rms > VAD_RMS_THRESHOLD:
                self.last_voice_at = now
            quiet_ms = (now - self.last_voice_at) * 1000.0
            if quiet_ms >= VAD_QUIET_MS and len(self.linear_pcm) >= max(2, int(SAMPLE_RATE_IN*BYTES_PER_SAMPLE*STT_MIN_MS/1000)):
                self.flush_to_whisper(); return

        if len(self.linear_pcm) >= LINEAR_BYTES_TARGET:
            self.flush_to_whisper()

    def finish(self):
        if self.linear_pcm and not self.inflight:
            self.flush_to_whisper()

    def _to_wav_bytes(self, pcm_bytes: bytes) -> bytes:
        buf = BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(BYTES_PER_SAMPLE); wf.setframerate(SAMPLE_RATE_IN)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    def flush_to_whisper(self):
        if not self.linear_pcm or self.inflight:
            return
        chunk = bytes(self.linear_pcm); self.linear_pcm.clear()

        if _openai_client is None:
            log.info("[stt] (dry-run) bytes=%d", len(chunk)); return

        self.inflight = True
        try:
            bio = BytesIO(self._to_wav_bytes(chunk)); bio.name = "clip.wav"; bio.seek(0)
            resp = _openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=bio,
                response_format="json",
                language="en",
                prompt=WHISPER_PROMPT,
            )
            text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None)
            if text and len(text.strip()) >= 2 and not any(re.search(p, text, re.I) for p in IGNORE_PATTERNS):
                log.info(f"[stt] {text}")
                if self.on_text:
                    try: self.on_text(text, chunk)
                    except Exception as e: log.warning(f"[stt] on_text error: {e}")
            else:
                log.info("[stt] (filtered or empty)")
        except Exception as e:
            log.warning(f"[stt] Whisper error: {e}")
        finally:
            self.inflight = False

# ---------- TTS helpers (normalization + padding) ----------
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
    params = {"optimize_streaming_latency": "2", "output_format": fmt}
    accept = "audio/wav" if fmt == "wav" else "audio/pcm"
    headers = {"xi-api-key": api_key, "accept": accept, "content-type": "application/json"}
    body = {"text": text, "model_id": os.getenv("ELEVENLABS_MODEL_ID", "eleven_monolingual_v1"),
            "voice_settings": {"stability": 0.4, "similarity_boost": 0.7}}
    async with httpx.AsyncClient(timeout=httpx.Timeout(35.0)) as client:
        r = await client.post(url, headers=headers, params=params, json=body)
        if r.status_code != 200: raise RuntimeError(f"ElevenLabs TTS failed: {r.status_code} {r.text[:200]}")
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

# ---------- Outbound Speaker with barge-in grace ----------
class TTSSpeaker:
    def __init__(self, ws: WebSocket, stream_sid: str):
        self.ws = ws; self.stream_sid = stream_sid
        self.q = asyncio.Queue()
        self._task = asyncio.create_task(self._worker())
        self._last_reply_at = 0.0
        self._last_hash = None
        self.playing = False
        self.cancel_event = asyncio.Event()
        self.after_barge = False

    async def _worker(self):
        try:
            while True:
                ulaw, mark_name = await self.q.get()
                await self._stream_ulaw(ulaw, mark_name)
                self.q.task_done()
        except asyncio.CancelledError:
            pass

    async def _stream_ulaw(self, ulaw: bytes, mark_name: str):
        FRAME_BYTES = 160  # 20 ms per frame
        total = 0
        self.playing = True
        self.cancel_event.clear()
        frames_played = 0
        while frames_played * FRAME_BYTES < len(ulaw):
            start = frames_played * FRAME_BYTES
            end = start + FRAME_BYTES
            payload = base64.b64encode(ulaw[start:end]).decode("ascii")
            msg = {"event": "media", "streamSid": self.stream_sid, "media": {"payload": payload}}
            await self.ws.send_text(json.dumps(msg))
            total += end - start
            frames_played += 1

            # Allow barge-in cancellation only after grace window
            played_ms = frames_played * 20
            if self.cancel_event.is_set() and ENABLE_BARGE_IN and played_ms >= BARGE_GRACE_MS:
                log.info(f"[barge] aborting playback after {played_ms} ms (grace={BARGE_GRACE_MS} ms)")
                break

            await asyncio.sleep(0.02)

        self.playing = False
        if not self.cancel_event.is_set() or not ENABLE_BARGE_IN:
            await self.ws.send_text(json.dumps({"event": "mark", "streamSid": self.stream_sid, "mark": {"name": mark_name}}))
            log.info(f"[tts] streamed {total} ulaw bytes to Twilio (mark={mark_name})")
        else:
            self.after_barge = True
            log.info(f"[tts] aborted after {total} bytes due to barge-in")

    async def enqueue_text(self, text: str):
        now = time.monotonic()
        min_gap = TTS_MIN_GAP_AFTER_BARGE_MS if self.after_barge else TTS_MIN_GAP_MS
        if (now - self._last_reply_at) * 1000 < min_gap:
            log.info("[tts] suppressed (cooldown)"); return
        h = hash(text.strip().lower())
        if h == self._last_hash:
            log.info("[tts] suppressed (dedupe)"); return
        self._last_hash = h; self._last_reply_at = now

        data, fmt = await elevenlabs_tts_bytes(text)
        ulaw = convert_elevenlabs_to_ulaw8k(data, fmt, gain_db=TTS_GAIN_DB)
        log.info(f"[tts] reply='{text[:60]}...' fmt={fmt} bytes_in={len(data)} -> ulaw={len(ulaw)} gain_db={TTS_GAIN_DB}")
        await self.q.put((ulaw, "tts-finished"))

    async def beep(self):
        if os.getenv("TTS_BEEP_ON_CONNECT", "0") != "1": return
        sr = 8000; n = int(sr * 0.5); amp = int(32767 * 0.28)
        pcm = bytearray()
        for i in range(n):
            s = int(amp * math.sin(2 * math.pi * 880.0 * (i / sr)))
            pcm.extend(s.to_bytes(2, byteorder="little", signed=True))
        ulaw = audioop.lin2ulaw(bytes(pcm), 2)
        log.info(f"[beep] sending {len(ulaw)} ulaw bytes")
        await self.q.put((ulaw, "beep"))

    async def close(self):
        self._task.cancel()
        with contextlib.suppress(Exception):
            await self._task

# ---------- WebSocket ----------
@app.websocket("/twilio/media")
async def twilio_media_ws(ws: WebSocket):
    await ws.accept(subprotocol="audio")
    frames = 0; stream_sid = None; speaker = None
    first_reply_sent = False

    def on_text(text: str, pcm: bytes):
        nonlocal first_reply_sent, speaker
        # Always send the first reply
        if not first_reply_sent:
            first_reply_sent = True
            asyncio.create_task(speaker.enqueue_text("Thanks, I hear you. What is the vehicle year, make, and model?"))
            return
        # Don't gate on 'quiet' here — let barge-in handle interruptions
        asyncio.create_task(speaker.enqueue_text("Could you also tell me your exact location?"))

    stt = STTBuffer(on_text=on_text)
    log.info("WS accepted (subprotocol=audio): /twilio/media")

    try:
        while True:
            try:
                message = await ws.receive()
            except WebSocketDisconnect:
                log.info("WS disconnect (peer)"); break
            except RuntimeError as e:
                log.info(f"WS receive after disconnect: {e}"); break

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
                    call_sid = data.get("start", {}).get("callSid")
                    log.info(f"[media] start callSid={call_sid} streamSid={stream_sid}")
                    speaker = TTSSpeaker(ws, stream_sid)
                    asyncio.create_task(speaker.beep())
                elif event == "media":
                    payload_b64 = data.get("media", {}).get("payload", "")
                    # barge-in detection
                    if speaker and speaker.playing and ENABLE_BARGE_IN:
                        try:
                            ulaw = base64.b64decode(payload_b64)
                            lin = audioop.ulaw2lin(ulaw, BYTES_PER_SAMPLE)
                            if _rms(lin) > VAD_RMS_THRESHOLD:
                                speaker.cancel_event.set()
                        except Exception:
                            pass
                    stt.add_ulaw_b64(payload_b64)
                    frames += 1
                    if frames % 25 == 0: log.info(f"[media] frames={frames}")
                elif event == "mark":
                    log.info(f"[media] other event: mark ({data.get('mark',{}).get('name')})")
                elif event == "stop":
                    log.info(f"[media] stop after frames={frames}")
                    stt.finish(); break
    finally:
        if speaker: await speaker.close()
        if ws.application_state == WebSocketState.CONNECTED: await ws.close()
        log.info("WS closed")
