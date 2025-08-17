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
from app.core.recorder import Recorder
from app.core.stt import STTBuffer, set_openai_client as stt_set_client
from app.core.dialog import DialogState
from app.core.tts import TTSSpeaker

from fastapi.responses import PlainTextResponse, FileResponse
from starlette.websockets import WebSocketState

# OpenAI SDK (v1+)
try:
    from openai import OpenAI
    _openai_client = OpenAI()
except Exception:
    _openai_client = None
# wire STTBuffer to the OpenAI client
try:
    stt_set_client(_openai_client)
except Exception:
    pass


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
TWILIO_RECORD_CALL = os.getenv("TWILIO_RECORD_CALL", "0") == "1"

# Endpointing
STT_CHUNK_S = float(os.getenv("STT_CHUNK_S", "1.2"))
LINEAR_BYTES_TARGET = int(SAMPLE_RATE_IN * BYTES_PER_SAMPLE * STT_CHUNK_S)
STT_FLUSH_ON_SILENCE = os.getenv("STT_FLUSH_ON_SILENCE", "1") == "1"
STT_MIN_MS = int(os.getenv("STT_MIN_MS", "400"))
VAD_RMS_THRESHOLD = int(os.getenv("VAD_RMS_THRESHOLD", "260"))
VAD_QUIET_MS = int(os.getenv("VAD_QUIET_MS", "300"))
STT_RMS_GATE = int(os.getenv("STT_RMS_GATE", "300"))
STT_MIN_GAP_BETWEEN_CALLS_MS = int(os.getenv("STT_MIN_GAP_BETWEEN_CALLS_MS", "1200"))

# TTS
TTS_GAIN_DB = float(os.getenv("TTS_GAIN_DB", "0"))
TTS_PEAK_TARGET = float(os.getenv("TTS_PEAK_TARGET", "0.65"))
TTS_MIN_GAP_MS = int(os.getenv("TTS_MIN_GAP_MS", "2400"))
TTS_MIN_GAP_AFTER_BARGE_MS = int(os.getenv("TTS_MIN_GAP_AFTER_BARGE_MS", "2200"))
TTS_PAD_PRE_MS = int(os.getenv("TTS_PAD_PRE_MS", "700"))
TTS_PAD_POST_MS = int(os.getenv("TTS_PAD_POST_MS", "600"))
ELEVEN_FMT = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "pcm_16000").lower()

ELEVEN_LATENCY_MODE = os.getenv("ELEVEN_LATENCY_MODE", "1")
ELEVEN_SPEED = float(os.getenv("ELEVEN_SPEED", "0.86"))
ELEVEN_STABILITY = float(os.getenv("ELEVEN_STABILITY", "0.65"))
ELEVEN_SIMILARITY = float(os.getenv("ELEVEN_SIMILARITY", "0.75"))
ELEVEN_STYLE = float(os.getenv("ELEVEN_STYLE", "0.20"))
ELEVEN_SPEAKER_BOOST = os.getenv("ELEVEN_SPEAKER_BOOST", "1") == "1"

# Soft-barge / ducking
ENABLE_BARGE_IN = os.getenv("ENABLE_BARGE_IN", "1") == "1"
BARGE_GRACE_MS = int(os.getenv("BARGE_GRACE_MS", "1500"))
FIRST_REPLY_NO_BARGE = os.getenv("FIRST_REPLY_NO_BARGE", "1") == "1"
FIRST_REPLY_WAIT_FOR_QUIET = os.getenv("FIRST_REPLY_WAIT_FOR_QUIET", "0") == "1"

TTS_BARGE_FADE_MS = int(os.getenv("TTS_BARGE_FADE_MS", "140"))         # fade-out when barge triggers
TTS_BARGE_POST_SILENCE_MS = int(os.getenv("TTS_BARGE_POST_SILENCE_MS", "180"))  # pad of silence after stop
TTS_DUCK_DB = float(os.getenv("TTS_DUCK_DB", "6"))                     # how much to duck while caller is talking
TTS_DUCK_DECAY_MS = int(os.getenv("TTS_DUCK_DECAY_MS", "800"))         # how long duck lasts after last detected speech

# Diagnostics
LOG_TTS_DEBUG = os.getenv("LOG_TTS_DEBUG", "0") == "1"
AUTO_TEST_TTS = os.getenv("AUTO_TEST_TTS", "0") == "0"

WHISPER_PROMPT = os.getenv("WHISPER_PROMPT", "Towing and roadside assistance. Year make model, location, issue, urgency. Keep transcripts concise.")

# ---------- Utils ----------
def _public_ws_url(request: Request) -> str:
    host = request.headers.get("x-forwarded-host") or request.url.hostname
    scheme = "wss" # Render is behind HTTPS; Twilio Media Stream should use wss
    return f"{scheme}://{host}/twilio/media"

def _rms(pcm16: bytes) -> float:
    try:
        return audioop.rms(pcm16, 2)
    except Exception:
        return 0.0

def _silence_ulaw(ms: int) -> bytes:
    samples = int(8000 * ms / 1000)
    return bytes([0xFF]) * samples  # μ-law silence byte

def _to_wav(path: Path, pcm16: bytes, sr: int = 8000):
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
    log.info(f"Returning TwiML with Media Stream URL: {ws_url}")
    twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
  {start_record}
  <Say>Horizon Road Rescue dispatcher connected. Please speak after the tone.</Say>
  <Connect><Stream url="{ws_url}" /></Connect>
</Response>'''
    return Response(content=twiml, media_type="application/xml")

# ---------- Recording manager ----------
# moved to app.core.recorder
