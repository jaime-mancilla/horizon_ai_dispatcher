import base64
import json
import logging
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import PlainTextResponse
from starlette.websockets import WebSocketState

log = logging.getLogger("hid.media")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

app = FastAPI(title="Horizon AI Dispatcher (Track B)")

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

def _public_ws_url(request: Request) -> str:
    # Render/ngrok friendly: prefer forwarded host, fall back to request host
    host = request.headers.get("x-forwarded-host") or request.url.hostname
    scheme = "wss"
    return f"{scheme}://{host}/twilio/media"

@app.post("/twilio/voice", response_class=PlainTextResponse)
async def twilio_voice_webhook(request: Request):
    ws_url = _public_ws_url(request)
    log.info(f"Returning TwiML with Media Stream URL: {ws_url}")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Start>
    <Stream url="{ws_url}" />
  </Start>
  <Say>Thanks for calling Horizon Road Rescue. You are connected to our dispatcher. Please start speaking after the tone.</Say>
  <Pause length="60"/>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

@app.websocket("/twilio/media")
async def twilio_media_ws(ws: WebSocket):
    # Accept Twilio's required subprotocol
    await ws.accept(subprotocol="audio")
    frames = 0
    bytes_total = 0
    log.info("WS accepted (subprotocol=audio): /twilio/media")
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.receive":
                # Twilio sends JSON text frames with {event: start|media|stop|mark}
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
                    # Approximate decoded size (base64 → bytes)
                    bytes_total += len(payload_b64) * 3 // 4
                    if frames % 25 == 0:
                        log.info(f"[media] frames={frames} approx_bytes={bytes_total}")
                elif event == "stop":
                    log.info(f"[media] stop after frames={frames} approx_bytes={bytes_total}")
                    break
                else:
                    log.info(f"[media] other event: {event}")
            # else: ignore other ws events
    except WebSocketDisconnect:
        log.info("WS disconnect")
    finally:
        if ws.application_state != WebSocketState.DISCONNECTED:
            await ws.close()
        log.info("WS closed")
