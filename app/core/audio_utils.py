# Audio utility helpers extracted from server.py
from io import BytesIO
import wave, audioop, math, base64, contextlib
from pathlib import Path


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
