import os
import asyncio
import base64
import json
import logging
from io import BytesIO
import wave
import audioop
import math
import time

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import PlainTextResponse
from starlette.websockets import WebSocketState

try:
    from openai import OpenAI
    _openai_client = OpenAI()
except Exception:
    _openai_client = None

log = logging.getLogger("hid.media")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

app = FastAPI(title="Horizon AI Dispatcher (Track B)")

SAMPLE_RATE_IN = 8000
BYTES_PER_SAMPLE = 2
SECONDS_PER_CHUNK = 2.5  # slightly longer for better STT
LINEAR_BYTES_TARGET = int(SAMPLE_RATE_IN * BYTES_PER_SAMPLE * SECONDS_PER_CHUNK)

def _public_ws_url(request: Request) -> str:
    host = request.headers.get("x-forwarded-host") or request.url.hostname
    scheme = "wss"
    return f"{scheme}://{host}/twilio/media"

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

class STTBuffer:
    def __init__(self, on_text=None):
        self.linear_pcm = bytearray()
        self.inflight = False
        self.chunks_sent = 0
        self.text_total = 0
        self.on_text = on_text
        self.last_rms = 0

    def add_ulaw_b64(self, payload_b64: str):
        try:
            ulaw = base64.b64decode(payload_b64)
            lin = audioop.ulaw2lin(ulaw, BYTES_PER_SAMPLE)
        except Exception as e:
            log.warning(f"Failed ulaw decode: {e}")
            return
        # track energy for simple voice-activity detection
        try:
            self.last_rms = audioop.rms(lin, BYTES_PER_SAMPLE)
        except Exception:
            self.last_rms = 0
        self.linear_pcm.extend(lin)
        if len(self.linear_pcm) >= LINEAR_BYTES_TARGET:
            self.flush_to_whisper()

    def finish(self):
        if self.linear_pcm and not self.inflight:
            self.flush_to_whisper()

    def _to_wav_bytes(self, pcm_bytes: bytes) -> bytes:
        buf = BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(BYTES_PER_SAMPLE)
            wf.setframerate(SAMPLE_RATE_IN)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    def flush_to_whisper(self):
        if not self.linear_pcm or self.inflight:
            return
        chunk = bytes(self.linear_pcm)
        self.linear_pcm.clear()
        wav_bytes = self._to_wav_bytes(chunk)

        if _openai_client is None:
            log.info("[stt] (dry-run) WAV bytes=%d", len(wav_bytes))
            return

        self.inflight = True
        try:
            bio = BytesIO(wav_bytes)
            bio.name = "clip.wav"
            bio.seek(0)
            resp = _openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=bio,
                response_format="json",
                language="en",
                prompt=os.getenv("WHISPER_PROMPT", "Towing and roadside assistance. Year make model, location, issue, urgency.")
            )
            text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None)
            self.chunks_sent += 1
            if text:
                self.text_total += len(text)
                log.info(f"[stt] {text}")
                if self.on_text:
                    try:
                        self.on_text(text)
                    except Exception as e:
                        log.warning(f"[stt] on_text callback error: {e}")
            else:
                log.info("[stt] (no text)")
        except Exception as e:
            log.warning(f"[stt] Whisper error: {e}")
        finally:
            self.inflight = False

# ---------- TTS helpers ----------
def _parse_wav(data: bytes):
    with wave.open(BytesIO(data), "rb") as wf:
        return wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.readframes(wf.getnframes())

def _apply_gain_linear(frames: bytes, sampwidth: int, gain_db: float) -> bytes:
    if gain_db <= 0:
        return frames
    factor = pow(10.0, gain_db / 20.0)
    try:
        out = audioop.mul(frames, sampwidth, factor)
        return out
    except Exception:
        return frames

def _to_ulaw_8k_from_linear(frames: bytes, sw: int, sr: int, nch: int, gain_db: float = 0.0) -> bytes:
    if sw != 2:
        frames = audioop.lin2lin(frames, sw, 2)
        sw = 2
    if nch == 2:
        frames = audioop.tomono(frames, 2, 0.5, 0.5)
        nch = 1
    if sr != 8000:
        frames, _ = audioop.ratecv(frames, 2, 1, sr, 8000, None)
        sr = 8000
    if gain_db > 0:
        frames = _apply_gain_linear(frames, 2, gain_db)
    return audioop.lin2ulaw(frames, 2)

def _ulaw_silence_ms(ms: int) -> bytes:
    # 0 linear maps to 0xFF in μ-law; create ms of silence
    samples = int(8000 * (ms/1000.0))
    return bytes([255]) * samples

async def elevenlabs_tts_bytes(text: str) -> tuple[bytes, str]:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    fmt = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "pcm_16000").lower()
    if not api_key or not voice_id:
        raise RuntimeError("Missing ELEVENLABS env vars")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    params = {"optimize_streaming_latency": "2", "output_format": fmt}
    accept = "audio/wav" if fmt == "wav" else "audio/pcm"
    headers = {"xi-api-key": api_key, "accept": accept, "content-type": "application/json"}
    body = {"text": text, "model_id": os.getenv("ELEVENLABS_MODEL_ID", "eleven_monolingual_v1"),
            "voice_settings": {"stability": 0.4, "similarity_boost": 0.7}}
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
        try:
            nch, sw, sr, frames = _parse_wav(data)
        except Exception:
            nch, sw, sr, frames = 1, 2, 16000, data
    return _to_ulaw_8k_from_linear(frames, sw, sr, nch, gain_db=gain_db)

async def stream_ulaw_to_twilio(ws: WebSocket, stream_sid: str, ulaw: bytes, mark_name: str):
    FRAME_BYTES = 160  # 20ms @ 8kHz
    total = 0
    # prepend + append short silence to avoid clipping by jitter buffers
    pad_pre = int(os.getenv("TTS_PAD_PRE_MS", "180"))
    pad_post = int(os.getenv("TTS_PAD_POST_MS", "220"))
    ulaw = _ulaw_silence_ms(pad_pre) + ulaw + _ulaw_silence_ms(pad_post)

    for i in range(0, len(ulaw), FRAME_BYTES):
        payload = base64.b64encode(ulaw[i:i+FRAME_BYTES]).decode("ascii")
        msg = {"event": "media", "streamSid": stream_sid, "media": {"payload": payload}}
        await ws.send_text(json.dumps(msg))
        total += len(ulaw[i:i+FRAME_BYTES])
        await asyncio.sleep(0.02)
    await ws.send_text(json.dumps({"event": "mark", "streamSid": stream_sid, "mark": {"name": mark_name}}))
    log.info(f"[tts] streamed {total} ulaw bytes to Twilio (mark={mark_name})")

def _sine_beep_ulaw(duration_ms: int = 600, freq: float = 880.0, gain_db: float = 6.0) -> bytes:
    # Generate a sine beep in linear PCM @ 8kHz, clamp safely, then μ-law encode
    sr = 8000
    n = int(sr * (duration_ms / 1000.0))
    # Clamp amplitude to 32767
    amp = min(32767, int(32767 * pow(10.0, gain_db/20.0) * 0.35))
    pcm = bytearray()
    for i in range(n):
        sample = int(amp * math.sin(2 * math.pi * freq * (i / sr)))
        if sample > 32767: sample = 32767
        if sample < -32768: sample = -32768
        pcm.extend(sample.to_bytes(2, byteorder="little", signed=True))
    return audioop.lin2ulaw(bytes(pcm), 2)

async def speak_reply(ws: WebSocket, stream_sid: str, text: str):
    gain_db = float(os.getenv("TTS_GAIN_DB", "12"))
    data, fmt = await elevenlabs_tts_bytes(text)
    ulaw = convert_elevenlabs_to_ulaw8k(data, fmt, gain_db=gain_db)
    log.info(f"[tts] reply='{text[:60]}...' fmt={fmt} bytes_in={len(data)} -> ulaw={len(ulaw)} gain_db={gain_db}")
    await stream_ulaw_to_twilio(ws, stream_sid, ulaw, mark_name="tts-finished")

@app.websocket("/twilio/media")
async def twilio_media_ws(ws: WebSocket):
    await ws.accept(subprotocol="audio")
    frames = 0
    stream_sid = None
    reply_task = None
    last_tts_end = 0.0

    stt_prompted = os.getenv("STT_FIRST_PROMPT", "Thanks, I hear you. What is the vehicle year, make, and model?")
    vad_threshold = int(os.getenv("VAD_RMS_THRESHOLD", "500"))   # simple energy gate
    vad_quiet_ms = int(os.getenv("VAD_QUIET_MS", "300"))         # wait for 300ms of quiet
    min_gap_ms = int(os.getenv("TTS_MIN_GAP_MS", "3500"))        # don't speak more often than this
    last_spoken_at = 0.0

    async def maybe_beep():
        if os.getenv("TTS_BEEP_ON_CONNECT", "0") == "1" and stream_sid:
            ulaw_beep = _sine_beep_ulaw(700, 880.0, gain_db=6.0)
            log.info(f"[beep] sending {len(ulaw_beep)} ulaw bytes")
            await stream_ulaw_to_twilio(ws, stream_sid, ulaw_beep, mark_name="beep")

    async def wait_for_quiet(stt_obj: STTBuffer):
        # spin until quiet or timeout
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if stt_obj.last_rms < vad_threshold:
                # require continuous quiet
                t0 = time.time()
                while time.time() - t0 < (vad_quiet_ms / 1000.0):
                    if stt_obj.last_rms >= vad_threshold:
                        break
                    await asyncio.sleep(0.02)
                else:
                    return True
            await asyncio.sleep(0.02)
        return False

    async def maybe_reply(text: str):
        nonlocal last_spoken_at, reply_task, last_tts_end
        now = time.time()
        if now - last_spoken_at < (min_gap_ms/1000.0):
            return
        if reply_task and not reply_task.done():
            return
        # wait for a short quiet window to avoid talking over the caller
        await wait_for_quiet(stt)
        last_spoken_at = time.time()
        reply_task = asyncio.create_task(speak_reply(ws, stream_sid, stt_prompted))

    def on_text(text: str):
        # schedule reply for every chunk (gated by min_gap & VAD)
        if stream_sid:
            asyncio.create_task(maybe_reply(text))

    stt = STTBuffer(on_text=on_text)
    log.info("WS accepted (subprotocol=audio): /twilio/media")

    try:
        while True:
            try:
                message = await ws.receive()
            except WebSocketDisconnect:
                log.info("WS disconnect (peer)")
                break
            except RuntimeError as e:
                log.info(f"WS receive after disconnect: {e}")
                break

            if message["type"] == "websocket.receive":
                payload_text = None
                if "text" in message and message["text"] is not None:
                    payload_text = message["text"]
                elif "bytes" in message and message["bytes"] is not None:
                    payload_text = message["bytes"].decode("utf-8", "ignore")
                if not payload_text:
                    continue

                try:
                    data = json.loads(payload_text)
                except Exception as e:
                    log.warning(f"Non-JSON frame (ignored): {payload_text[:64]}... ({e})")
                    continue

                event = data.get("event")
                if event == "start":
                    stream_sid = data.get("start", {}).get("streamSid")
                    call_sid = data.get("start", {}).get("callSid")
                    log.info(f"[media] start callSid={call_sid} streamSid={stream_sid}")
                    asyncio.create_task(maybe_beep())
                elif event == "media":
                    payload_b64 = data.get("media", {}).get("payload", "")
                    stt.add_ulaw_b64(payload_b64)
                    frames += 1
                    if frames % 25 == 0:
                        log.info(f"[media] frames={frames}")
                elif event == "mark":
                    mark_name = data.get("mark", {}).get("name")
                    log.info(f"[media] other event: mark ({mark_name})")
                    if mark_name == "tts-finished":
                        last_tts_end = time.time()
                elif event == "stop":
                    log.info(f"[media] stop after frames={frames}")
                    stt.finish()
                    break
                else:
                    log.info(f"[media] other event: {event}")
    finally:
        try:
            if reply_task:
                await asyncio.wait_for(reply_task, timeout=10.0)
        except Exception as e:
            log.info(f"[tts] reply task end: {e}")
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.close()
        log.info("WS closed")
