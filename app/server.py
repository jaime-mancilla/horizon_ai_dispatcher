import base64
import json
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import PlainTextResponse
from starlette.websockets import WebSocketState

app = FastAPI(title="Horizon AI Dispatcher (Track B)")

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

def _public_ws_url(request: Request) -> str:
    # Build wss://<host>/twilio/media from the incoming request host.
    # Works for both ngrok and Render.
    host = request.headers.get("x-forwarded-host") or request.url.hostname
    scheme = "wss"
    return f"{scheme}://{host}/twilio/media"

@app.post("/twilio/voice", response_class=PlainTextResponse)
async def twilio_voice_webhook(request: Request):
    # Return TwiML that starts a Media Stream to our WebSocket endpoint.
    ws_url = _public_ws_url(request)
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
    # Accept WebSocket from Twilio Media Streams
    await ws.accept()
    frames = 0
    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            event = data.get("event")
            if event == "start":
                call_sid = data.get("start", {}).get("callSid")
                stream_sid = data.get("start", {}).get("streamSid")
                print(f"[media] start callSid={call_sid} streamSid={stream_sid}")
            elif event == "media":
                # Media payload is base64-encoded 16-bit PCM @ 8kHz, mono
                frames += 1
                payload_b64 = data["media"]["payload"]
                # raw = base64.b64decode(payload_b64)  # bytes of PCM frame (160 samples typical)
                # TODO: feed to a streaming STT (buffer small windows -> Whisper/other)
            elif event == "stop":
                print(f"[media] stop after {frames} frames")
                break
    except WebSocketDisconnect:
        pass
    finally:
        if ws.application_state != WebSocketState.DISCONNECTED:
            await ws.close()
