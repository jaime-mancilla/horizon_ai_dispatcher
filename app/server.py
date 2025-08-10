from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import PlainTextResponse
from app.llm.agent import LLMDispatcher
from app.voice.transcriber import Transcriber
from app.voice.synthesizer import Synthesizer

# Minimal server with health check and a Twilio webhook stub.
# Twilio will POST form-encoded data here.
app = FastAPI(title="Horizon AI Dispatcher")

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/twilio/voice", response_class=PlainTextResponse)
async def twilio_voice_webhook(
    request: Request,
    SpeechResult: str = Form(default=None),  # Twilio <Gather input="speech"> result (optional)
):
    # For MVP, return TwiML that simply speaks and ends.
    # We'll later integrate ElevenLabs audio via <Play> or Media Streams.
    response_text = "Thanks for calling Horizon Road Rescue. This number is connected. Please try again later."
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>{response_text}</Say>
  <Hangup/>
</Response>"""
    return Response(content=twiml, media_type="application/xml")
