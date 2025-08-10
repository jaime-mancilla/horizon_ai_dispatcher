import os
import asyncio
import base64
import json
import logging
import time
from io import BytesIO
import wave
import audioop

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

SAMPLE_RATE_IN = 8000          # Twilio inbound
BYTES_PER_SAMPLE = 2           # 16-bit linear PCM
SECONDS_PER_CHUNK = 2.0        # Whisper micro-batch
LINEAR_BYTES_TARGET = int(SAMPLE_RATE_IN * BYTES_PER_SAMPLE * SECONDS_PER_CHUNK)

# --- Helpers ---
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
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>Horizon Road Rescue dispatcher connected. Please speak after the tone.</Say>
  <Connect><Stream url="{ws_url}" /></Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

# --- STT buffering ---
class STTBuffer:
    def __init__(self, on_text=None):
        self.linear_pcm = bytearray()
        self.inflight = False
        self.chunks_sent = 0
        self.text_total = 0
        self.on_text = on_text

    def add_ulaw_b64(self, payload_b64: str):
        try:
            ulaw = base64.b64decode(payload_b64)
            lin = audioop.ulaw2lin(ulaw, BYTES_PER_SAMPLE)
        except Exception as e:
            log.warning(f"Failed ulaw decode: {e}")
            return
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
                language="en"
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

# --- TTS (clean playback) ---
async def elevenlabs_tts_wav(text: str) -> bytes:
    """
    Ask ElevenLabs for WAV. We'll resample/convert locally.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    if not api_key or not voice_id:
        raise RuntimeError("ElevenLabs env vars missing (ELEVENLABS_API_KEY / ELEVENLABS_VOICE_ID)")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "accept": "audio/wav",
        "content-type": "application/json",
    }
    body = {
        "text": text,
        "model_id": os.getenv("ELEVENLABS_MODEL_ID", "eleven_monolingual_v1"),
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.7},
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        r = await client.post(url, headers=headers, json=body)
        if r.status_code != 200:
            raise RuntimeError(f"ElevenLabs TTS failed: {r.status_code} {r.text[:200]}")
        return r.content

def wav_to_ulaw_8k(wav_bytes: bytes) -> bytes:
    """
    Convert arbitrary WAV (any sr/ch/width) to ulaw@8000 mono.
    """
    with wave.open(BytesIO(wav_bytes), "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    # Ensure 16-bit linear
    if sw != 2:
        frames = audioop.lin2lin(frames, sw, 2)
        sw = 2

    # Stereo -> mono
    if nch == 2:
        frames = audioop.tomono(frames, sw, 0.5, 0.5)
        nch = 1

    # Resample to 8000 Hz
    if sr != 8000:
        frames, _ = audioop.ratecv(frames, sw, nch, sr, 8000, None)
        sr = 8000

    # Convert to μ-law
    ulaw = audioop.lin2ulaw(frames, 2)  # input width=2
    return ulaw

async def stream_ulaw_to_twilio(ws: WebSocket, stream_sid: str, ulaw: bytes):
    """
    Send μ-law bytes to Twilio in 20ms frames (160 bytes each) at ~real-time pace.
    """
    FRAME_SAMPLES = 160  # 20ms @ 8kHz
    FRAME_BYTES = FRAME_SAMPLES  # μ-law is 8-bit
    total = 0
    for i in range(0, len(ulaw), FRAME_BYTES):
        chunk = ulaw[i:i+FRAME_BYTES]
        if not chunk:
            continue
        payload = base64.b64encode(chunk).decode("ascii")
        msg = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload},
            "track": "outbound",
        }
        await ws.send_text(json.dumps(msg))
        total += len(chunk)
        await asyncio.sleep(0.02)  # pace in real time
    # send a mark to indicate playback done
    mark = {"event": "mark", "streamSid": stream_sid, "mark": {"name": "tts-finished"}}
    await ws.send_text(json.dumps(mark))
    log.info(f"[tts] streamed {total} ulaw bytes to Twilio")

async def speak_reply(ws: WebSocket, stream_sid: str, text: str):
    wav = await elevenlabs_tts_wav(text)
    ulaw = wav_to_ulaw_8k(wav)
    log.info(f"[tts] reply='{text[:60]}...' wav={len(wav)} bytes -> ulaw={len(ulaw)} bytes")
    await stream_ulaw_to_twilio(ws, stream_sid, ulaw)

# --- WebSocket handler ---
@app.websocket("/twilio/media")
async def twilio_media_ws(ws: WebSocket):
    await ws.accept(subprotocol="audio")
    frames = 0
    bytes_total = 0
    stream_sid = None
    first_reply_task = None
    replied = False

    def on_text(text: str):
        nonlocal first_reply_task, replied, stream_sid
        if not replied and stream_sid:
            replied = True
            # Ask year/make/model after the caller starts talking
            first_reply_task = asyncio.create_task(speak_reply(ws, stream_sid, "Thanks, I hear you. What is the vehicle year, make, and model?"))

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
                elif event == "media":
                    payload_b64 = data.get("media", {}).get("payload", "")
                    frames += 1
                    bytes_total += len(payload_b64) * 3 // 4
                    stt.add_ulaw_b64(payload_b64)
                    if frames % 25 == 0:
                        log.info(f"[media] frames={frames} approx_bytes={bytes_total}")
                elif event == "stop":
                    log.info(f"[media] stop after frames={frames} approx_bytes={bytes_total}")
                    stt.finish()
                    break
                else:
                    log.info(f"[media] other event: {event}")
            # ignore other types

    finally:
        # Wait for TTS task to finish sending (with timeout), then close if still open
        try:
            if first_reply_task:
                await asyncio.wait_for(first_reply_task, timeout=10.0)
        except Exception as e:
            log.info(f"[tts] reply task end: {e}")
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.close()
        log.info(f"WS closed summary: frames={frames} chunks={stt.chunks_sent} text_chars={stt.text_total}")
