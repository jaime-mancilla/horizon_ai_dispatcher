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
from app.core.audio_utils import _rms, _silence_ulaw, _to_wav, _apply_gain_linear, _parse_wav, _peak_normalize, _to_ulaw_8k_from_linear, convert_elevenlabs_to_ulaw8k
from app import config as _config  # ensure env constants are initialized
from fastapi.responses import PlainTextResponse, FileResponse
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
stt_set_client(_openai_client)

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

async def elevenlabs_tts_bytes(text: str, speed: float | None = None, style: float | None = None) -> tuple[bytes, str]:
    api_key = os.getenv("ELEVENLABS_API_KEY"); voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    if not api_key or not voice_id: raise RuntimeError("Missing ELEVENLABS env vars")
    fmt = ELEVEN_FMT
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    params = {"optimize_streaming_latency": str(ELEVEN_LATENCY_MODE), "output_format": fmt}
    accept = "audio/wav" if fmt == "wav" else "audio/pcm"
    headers = {"xi-api-key": api_key, "accept": accept, "content-type": "application/json"}
    voice_settings = {
        "stability": ELEVEN_STABILITY,
        "similarity_boost": ELEVEN_SIMILARITY,
        "style": (style if style is not None else ELEVEN_STYLE),
        "use_speaker_boost": ELEVEN_SPEAKER_BOOST,
        "speed": (speed if speed is not None else ELEVEN_SPEED),
    }
    body = {"text": text, "model_id": os.getenv("ELEVENLABS_MODEL_ID", "eleven_monolingual_v1"),
            "voice_settings": voice_settings}
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
        try: nch, sw, sr, frames = _parse_wav(data)
        except Exception: nch, sw, sr, frames = 1, 2, 16000, data
    ulaw = _to_ulaw_8k_from_linear(frames, sw, sr, nch, gain_db=gain_db)
    pre = _silence_ulaw(TTS_PAD_PRE_MS); post = _silence_ulaw(TTS_PAD_POST_MS)
    return pre + ulaw + post

# ---------- Conversation state (simple) ----------
VEHICLE_MAKES = r"(ford|chevy|chevrolet|gmc|ram|dodge|toyota|honda|nissan|jeep|bmw|mercedes|benz|kia|hyundai|subaru|vw|volkswagen|audi|lexus|infiniti|acura|mazda|buick|cadillac|chrysler|lincoln|volvo|porsche|jaguar|land rover|mini|mitsubishi|tesla)"
VEHICLE_TYPES = r"(car|truck|suv|van|pickup|tow ?truck|box ?truck)"
LOC_HINTS = r"(nashville|tennessee|i[- ]?\d+|hwy|highway|exit|rd|road|st|street|ave|avenue|blvd|pike|tn-?\d+)"
ISSUE_HINTS = r"(flat|tire|battery|dead|won'?t start|no start|engine|overheat|overheating|accident|lock(ed)? out|fuel|gas|out of gas|tow|winch|ditch|stuck)"

