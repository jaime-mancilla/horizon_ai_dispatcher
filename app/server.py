import base64, json, logging, time
from io import BytesIO
import wave, audioop
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import PlainTextResponse
from starlette.websockets import WebSocketState

try:
    from openai import OpenAI
    _openai = OpenAI()
except Exception:
    _openai = None

log = logging.getLogger("hid.media")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

app = FastAPI(title="Horizon AI Dispatcher (Track B)")

SAMPLE_RATE = 8000
BYTES_PER_SAMPLE = 2  # 16-bit PCM
SECONDS_PER_CHUNK = 3.0  # <- was 2.0; lower request rate
LINEAR_BYTES_TARGET = int(SAMPLE_RATE * BYTES_PER_SAMPLE * SECONDS_PER_CHUNK)

# STT rate limit: never send more than one request per MIN_SECS
MIN_SECS_BETWEEN_TRANSCRIBE = 3.5

# Very simple VAD: only keep frames whose RMS exceeds threshold.
# Threshold picked empirically for 8kHz ulaw->lin audio.
RMS_THRESHOLD = 400  # tweak if we drop too much/too little

def _public_ws_url(request: Request) -> str:
    host = request.headers.get("x-forwarded-host") or request.url.hostname
    return f"wss://{host}/twilio/media"

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
        self.last_sent_ts = 0.0
        self.backoff_until = 0.0

    def _append_linear(self, lin: bytes):
        # VAD: only add if voice energy above threshold
        if not lin:
            return
        # compute rms in 20ms slices to discard silence
        frame = int(SAMPLE_RATE * BYTES_PER_SAMPLE * 0.02)
        kept = bytearray()
        for i in range(0, len(lin), frame):
            seg = lin[i:i+frame]
            if audioop.rms(seg, BYTES_PER_SAMPLE) >= RMS_THRESHOLD:
                kept.extend(seg)
        if kept:
            self.linear_pcm.extend(kept)

    def add_ulaw_b64(self, payload_b64: str):
        try:
            ulaw = base64.b64decode(payload_b64)
            lin = audioop.ulaw2lin(ulaw, BYTES_PER_SAMPLE)
        except Exception as e:
            log.warning(f"Failed ulaw decode: {e}")
            return
        self._append_linear(lin)
        # Flush if we have enough buffered AND not rate-limited
        if len(self.linear_pcm) >= LINEAR_BYTES_TARGET:
            self.flush_to_whisper()

    def finish(self):
        self.flush_to_whisper(force=True)

    def _to_wav(self, pcm: bytes) -> bytes:
        buf = BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(BYTES_PER_SAMPLE)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm)
        return buf.getvalue()

    def flush_to_whisper(self, force: bool=False):
        if not self.linear_pcm or self.inflight:
            return
        now = time.time()
        if not force and now < max(self.backoff_until, self.last_sent_ts + MIN_SECS_BETWEEN_TRANSCRIBE):
            # Hold until the rate-limit window passes
            return

        chunk = bytes(self.linear_pcm)
        self.linear_pcm.clear()
        wav = self._to_wav(chunk)

        if _openai is None:
            log.info("[stt] (dry-run) WAV bytes=%d", len(wav))
            self.last_sent_ts = now
            return

        self.inflight = True
        try:
            bio = BytesIO(wav); bio.name = "clip.wav"
            resp = _openai.audio.transcriptions.create(model="whisper-1", file=bio, response_format="json", language="en")
            text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None)
            self.chunks_sent += 1
            self.last_sent_ts = time.time()
            if text: 
                self.text_total += len(text)
                log.info(f"[stt] {text}")
            else:
                log.info("[stt] (no text)")
        except Exception as e:
            # crude parse: if "429" appears, back off hard for 20s
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg:
                self.backoff_until = time.time() + 20.0
                log.warning("[stt] Whisper 429; backing off 20s")
            else:
                log.warning(f"[stt] Whisper error: {e}")
        finally:
            self.inflight = False

@app.websocket("/twilio/media")
async def twilio_media_ws(ws: WebSocket):
    await ws.accept(subprotocol="audio")
    frames = 0; bytes_total = 0
    stt = STTBuffer()
    log.info("WS accepted (subprotocol=audio): /twilio/media")
    try:
        while True:
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                log.info("WS disconnect (peer)")
                break
            except RuntimeError as e:
                log.info(f"WS receive after disconnect: {e}")
                break

            if msg["type"] != "websocket.receive":
                continue

            payload_text = msg.get("text")
            if payload_text is None and msg.get("bytes") is not None:
                payload_text = msg["bytes"].decode("utf-8", "ignore")
            if not payload_text:
                continue

            try:
                data = json.loads(payload_text)
            except Exception:
                continue

            ev = data.get("event")
            if ev == "start":
                s = data.get("start", {})
                log.info(f"[media] start callSid={s.get('callSid')} streamSid={s.get('streamSid')}")
            elif ev == "media":
                b64 = data.get("media", {}).get("payload", "")
                frames += 1
                bytes_total += len(b64) * 3 // 4
                stt.add_ulaw_b64(b64)
                # opportunistic flush if buffer big and not rate-limited
                stt.flush_to_whisper()
                if frames % 25 == 0:
                    log.info(f"[media] frames={frames} approx_bytes={bytes_total}")
            elif ev == "stop":
                log.info(f"[media] stop after frames={frames} approx_bytes={bytes_total}")
                stt.finish()
                break
    finally:
        # Let Twilio close; avoid double-close crash
        if ws.client_state == WebSocketState.CONNECTED and ws.application_state == WebSocketState.CONNECTED:
            try:
                await ws.close()
            except Exception:
                pass
        log.info(f"WS closed summary: frames={frames} chunks={stt.chunks_sent} text_chars={stt.text_total}")
