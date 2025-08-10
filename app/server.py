import base64
import json
import logging
import os
import time
from io import BytesIO
import wave
import audioop
from typing import Optional

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import PlainTextResponse
from starlette.websockets import WebSocketState

# OpenAI SDK (v1+)
try:
    from openai import OpenAI
    _openai_client = OpenAI()
except Exception:
    _openai_client = None

log = logging.getLogger("hid.media")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

app = FastAPI(title="Horizon AI Dispatcher (Track B)")

SAMPLE_RATE = 8000            # Hz
BYTES_PER_SAMPLE = 2          # 16-bit linear PCM
SECONDS_PER_CHUNK = 2.0       # Whisper batch size
LINEAR_BYTES_TARGET = int(SAMPLE_RATE * BYTES_PER_SAMPLE * SECONDS_PER_CHUNK)

ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVEN_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
ELEVEN_TTS_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}" if ELEVEN_VOICE_ID else ""

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

class STTBuffer:
    def __init__(self):
        self.linear_pcm = bytearray()
        self.inflight = False
        self.chunks_sent = 0
        self.text_total = 0

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
            wf.setframerate(SAMPLE_RATE)
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
                on_transcript(text)  # fire-and-forget simple callback (sets a global flag)
            else:
                log.info("[stt] (no text)")
        except Exception as e:
            log.warning(f"[stt] Whisper error: {e}")
        finally:
            self.inflight = False

# ---- Simple agent state (very basic) ----
FIRST_REPLY_SENT = False
LAST_STREAM_SID = None
WS_FOR_SEND: Optional[WebSocket] = None

def on_transcript(text: str):
    global FIRST_REPLY_SENT
    # Send one short TTS prompt the first time we detect speech.
    if not FIRST_REPLY_SENT and WS_FOR_SEND and LAST_STREAM_SID:
        FIRST_REPLY_SENT = True
        # Keep it short to minimize latency/cost.
        msg = "Thanks, I hear you. What is the vehicle year, make, and model?"
        try:
            import asyncio
            asyncio.create_task(send_tts_over_ws(WS_FOR_SEND, LAST_STREAM_SID, msg))
        except Exception as e:
            log.warning(f"TTS schedule failed: {e}")

async def send_tts_over_ws(ws: WebSocket, stream_sid: str, text: str):
    if not ELEVEN_API_KEY or not ELEVEN_VOICE_ID:
        log.info("[tts] ElevenLabs not configured; skipping send")
        return
    if ws.application_state != WebSocketState.CONNECTED:
        log.info("[tts] WS not connected; skipping")
        return

    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "accept": "audio/basic",  # μ-law 8k is often served as audio/basic
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        # Request ulaw_8000 directly to avoid local transcoding
        "output_format": "ulaw_8000",
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.8},
        # Low-latency streaming optimization if supported
        "optimize_streaming_latency": 2
    }

    url = ELEVEN_TTS_URL or ""
    if not url:
        log.info("[tts] No ELEVENLABS_VOICE_ID; skipping")
        return

    log.info(f"[tts] requesting ElevenLabs ulaw_8000 for '{text}'")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            audio = r.content  # Expect μ-law 8k raw/ or WAV/basic; assume raw bytes
    except Exception as e:
        log.warning(f"[tts] ElevenLabs request failed: {e}")
        return

    # Chunk μ-law 8k bytes into ~320-byte frames (~40ms) and send as Twilio media events
    CHUNK = 320
    sent = 0
    for i in range(0, len(audio), CHUNK):
        part = audio[i:i+CHUNK]
        if not part:
            continue
        b64 = base64.b64encode(part).decode("ascii")
        msg = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": b64},
        }
        try:
            await ws.send_text(json.dumps(msg))
            sent += len(part)
        except Exception as e:
            log.warning(f"[tts] send failed at {sent} bytes: {e}")
            break

    # Optional: send a mark to indicate end of audio
    try:
        await ws.send_text(json.dumps({"event": "mark", "streamSid": stream_sid, "mark": {"name": "tts-complete"}}))
    except Exception:
        pass
    log.info(f"[tts] sent {sent} ulaw bytes to Twilio")

@app.websocket("/twilio/media")
async def twilio_media_ws(ws: WebSocket):
    global LAST_STREAM_SID, WS_FOR_SEND, FIRST_REPLY_SENT
    await ws.accept(subprotocol="audio")
    frames = 0
    bytes_total = 0
    stt = STTBuffer()
    WS_FOR_SEND = ws
    FIRST_REPLY_SENT = False
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
                    call_sid = data.get("start", {}).get("callSid")
                    stream_sid = data.get("start", {}).get("streamSid")
                    LAST_STREAM_SID = stream_sid
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
            # ignore other message types

    finally:
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.close()
        WS_FOR_SEND = None
        log.info(f"WS closed summary: frames={frames} chunks={stt.chunks_sent} text_chars={stt.text_total}")
