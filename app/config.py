"""Single source of environment-backed settings for the app."""
import os
from pathlib import Path

# Audio constants
SAMPLE_RATE_IN = 8000
BYTES_PER_SAMPLE = 2

# Recording / debug
RECORD_CALL = os.getenv("RECORD_CALL", "0") == "1"
RECORD_DIR = os.getenv("RECORD_DIR", "/tmp/recordings")
TWILIO_RECORD_CALL = os.getenv("TWILIO_RECORD_CALL", "0") == "1"

# STT / VAD
STT_CHUNK_S = float(os.getenv("STT_CHUNK_S", "1.2"))
LINEAR_BYTES_TARGET = int(BYTES_PER_SAMPLE * STT_CHUNK_S * SAMPLE_RATE_IN)
STT_FLUSH_ON_SILENCE = os.getenv("STT_FLUSH_ON_SILENCE", "1") == "1"
STT_MIN_MS = int(os.getenv("STT_MIN_MS", "400"))
VAD_RMS_THRESHOLD = int(os.getenv("VAD_RMS_THRESHOLD", "260"))
VAD_QUIET_MS = int(os.getenv("VAD_QUIET_MS", "300"))
STT_RMS_GATE = int(os.getenv("STT_RMS_GATE", "300"))
STT_MIN_GAP_BETWEEN_CALLS_MS = int(os.getenv("STT_MIN_GAP_BETWEEN_CALLS_MS", "1200"))

# TTS / padding & cool-downs
TTS_GAIN_DB = float(os.getenv("TTS_GAIN_DB", "0"))
TTS_PEAK_TARGET = float(os.getenv("TTS_PEAK_TARGET", "0.65"))
TTS_MIN_GAP_MS = int(os.getenv("TTS_MIN_GAP_MS", "2400"))
TTS_MIN_GAP_AFTER_BARGE_MS = int(os.getenv("TTS_MIN_GAP_AFTER_BARGE_MS", "2200"))
TTS_PAD_PRE_MS = int(os.getenv("TTS_PAD_PRE_MS", "700"))
TTS_PAD_POST_MS = int(os.getenv("TTS_PAD_POST_MS", "600"))
ELEVENLABS_OUTPUT_FORMAT = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "pcm_16000").lower()

# ElevenLabs pacing
ELEVEN_LATENCY_MODE = os.getenv("ELEVEN_LATENCY_MODE", "1")
ELEVEN_SPEED = float(os.getenv("ELEVEN_SPEED", "0.86"))
ELEVEN_STABILITY = float(os.getenv("ELEVEN_STABILITY", "0.65"))
ELEVEN_SIMILARITY = float(os.getenv("ELEVEN_SIMILARITY", "0.75"))
ELEVEN_STYLE = float(os.getenv("ELEVEN_STYLE", "0.20"))
ELEVEN_SPEAKER_BOOST = os.getenv("ELEVEN_SPEAKER_BOOST", "1") == "1"

# Barge / duck
ENABLE_BARGE_IN = os.getenv("ENABLE_BARGE_IN", "1") == "1"
BARGE_GRACE_MS = int(os.getenv("BARGE_GRACE_MS", "1500"))
FIRST_REPLY_NO_BARGE = os.getenv("FIRST_REPLY_NO_BARGE", "1") == "1"
FIRST_REPLY_WAIT_FOR_QUIET = os.getenv("FIRST_REPLY_WAIT_FOR_QUIET", "0") == "1"
TTS_BARGE_FADE_MS = int(os.getenv("TTS_BARGE_FADE_MS", "140"))
TTS_BARGE_POST_SILENCE_MS = int(os.getenv("TTS_BARGE_POST_SILENCE_MS", "180"))
TTS_DUCK_DB = int(os.getenv("TTS_DUCK_DB", "6"))
TTS_DUCK_DECAY_MS = int(os.getenv("TTS_DUCK_DECAY_MS", "800"))

# Prompts / hints
WHISPER_PROMPT = os.getenv("WHISPER_PROMPT", "Towing and roadside assistance. Year make model, location, issue, urgency. Keep transcripts concise.")
IGNORE_PATTERNS = (r"(?i)^ok(ay)?[.!]?$", r"(?i)^hmm[.!]?$")
VEHICLE_MAKES = r"(ford|chevy|chevrolet|dodge|toyota|honda|nissan|jeep|bmw|mercedes|benz|kia|hyundai|subaru|vw|volkswagen|audi)"
VEHICLE_TYPES = r"(car|suv|van|pickup|box truck|tow truck)"
LOC_HINTS = r"(nashville|tennessee|i-?24|i-?40|hwy|highway|exit|road|street|lane|avenue|blvd|pike|tn-\d+)"
ISSUE_HINTS = r"(flat|tire|battery|dead|won’t start|no start|engine|overheat|overheating|accident|lock(ed)? out|fuel|gas|out of gas)"

# Files
FRAME_BYTES = 160 * 2 * 2  # 20 ms @ 8k, 16-bit mono

# Secrets are read by callers directly from env (never printed)
