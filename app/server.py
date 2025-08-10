import base64
import json
import logging
import time
from io import BytesIO
import wave
import audioop
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import PlainTextResponse
from starlette.websockets import WebSocketState

# OpenAI SDK (v1+)
try:
    from openai import OpenAI
    _openai_client = OpenAI()
except Exception as e:
    _openai_client = None

log = logging.getLogger("hid.media")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

app = FastAPI(title="Horizon AI Dispatcher (Track B)")

SAMPLE_RATE = 8000  # Hz
BYTES_PER_SAMPLE = 2  # 16-bit linear PCM after ulaw decode
SECONDS_PER_CHUNK = 1.0  # micro-batch size for Whisper
LINEAR_BYTES_TARGET = int(SAMPLE_RATE * BYTES_PER_SAMPLE * SECONDS_PER_CHUNK)

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
    # Use <Connect><Stream> to enable bidirectional later (when we add TTS)
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}" />
  </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

class STTBuffer:
    def __init__(self):
        self.linear_pcm = bytearray()
        self.last_flush = time.time()

    def add_ulaw_b64(self, payload_b64: str):
        # Decode base64 -> 8-bit mu-law -> 16-bit linear PCM
        try:
            ulaw = base64.b64decode(payload_b64)
            lin = audioop.ulaw2lin(ulaw, BYTES_PER_SAMPLE)  # 16-bit
        except Exception as e:
            log.warning(f"Failed ulaw decode: {e}")
            return

        self.linear_pcm.extend(lin)
        if len(self.linear_pcm) >= LINEAR_BYTES_TARGET:
            self.flush_to_whisper()

    def finish(self):
        if self.linear_pcm:
            self.flush_to_whisper()

    def _to_wav_bytes(self, pcm_bytes: bytes) -> bytes:
        buf = BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(BYTES_PER_SAMPLE)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    def flush_to_whisper(self):
        if not self.linear_pcm:
            return
        chunk = bytes(self.linear_pcm)
        self.linear_pcm.clear()

        wav_bytes = self._to_wav_bytes(chunk)

        if _openai_client is None:
            log.info("[stt] (dry-run) Whisper disabled (no client). WAV bytes=%d", len(wav_bytes))
            return

        try:
            bio = BytesIO(wav_bytes)
            bio.name = "clip.wav"
            resp = _openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=bio,
                response_format="json",
                language="en"
            )
            text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None)
            if text:
                log.info(f"[stt] {text}")
            else:
                log.info("[stt] (no text)")
        except Exception as e:
            log.warning(f"[stt] Whisper error: {e}")

@app.websocket("/twilio/media")
async def twilio_media_ws(ws: WebSocket):
    await ws.accept(subprotocol="audio")
    frames = 0
    bytes_total = 0
    stt = STTBuffer()
    log.info("WS accepted (subprotocol=audio): /twilio/media")
    try:
        while True:
            message = await ws.receive()
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
    except WebSocketDisconnect:
        log.info("WS disconnect")
    finally:
        if ws.application_state != WebSocketState.DISCONNECTED:
            await ws.close()
        log.info("WS closed")
