import os
from pathlib import Path

SAMPLE_RATE_IN = 8000
BYTES_PER_SAMPLE = 2
RECORD_CALL = os.getenv("RECORD_CALL", "0") == "1"
RECORD_DIR = os.getenv("RECORD_DIR", "/tmp/recordings")
TWILIO_RECORD_CALL = os.getenv("TWILIO_RECORD_CALL", "0") == "1"
STT_CHUNK_S = float(os.getenv("STT_CHUNK_S", "1.2"))
LINEAR_BYTES_TARGET = int(SAMPLE_RATE_IN * BYTES_PER_SAMPLE * STT_CHUNK_S)
STT_FLUSH_ON_SILENCE = os.getenv("STT_FLUSH_ON_SILENCE", "1") == "1"
STT_MIN_MS = int(os.getenv("STT_MIN_MS", "400"))
VAD_RMS_THRESHOLD = int(os.getenv("VAD_RMS_THRESHOLD", "260"))
VAD_QUIET_MS = int(os.getenv("VAD_QUIET_MS", "300"))
STT_RMS_GATE = int(os.getenv("STT_RMS_GATE", "300"))
STT_MIN_GAP_BETWEEN_CALLS_MS = int(os.getenv("STT_MIN_GAP_BETWEEN_CALLS_MS", "1200"))
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
ENABLE_BARGE_IN = os.getenv("ENABLE_BARGE_IN", "1") == "1"
BARGE_GRACE_MS = int(os.getenv("BARGE_GRACE_MS", "1500"))
FIRST_REPLY_NO_BARGE = os.getenv("FIRST_REPLY_NO_BARGE", "1") == "1"
FIRST_REPLY_WAIT_FOR_QUIET = os.getenv("FIRST_REPLY_WAIT_FOR_QUIET", "0") == "1"
TTS_BARGE_FADE_MS = int(os.getenv("TTS_BARGE_FADE_MS", "140"))         # fade-out when barge triggers
TTS_BARGE_POST_SILENCE_MS = int(os.getenv("TTS_BARGE_POST_SILENCE_MS", "180"))  # pad of silence after stop
TTS_DUCK_DB = float(os.getenv("TTS_DUCK_DB", "6"))                     # how much to duck while caller is talking
TTS_DUCK_DECAY_MS = int(os.getenv("TTS_DUCK_DECAY_MS", "800"))         # how long duck lasts after last detected speech
LOG_TTS_DEBUG = os.getenv("LOG_TTS_DEBUG", "0") == "1"
AUTO_TEST_TTS = os.getenv("AUTO_TEST_TTS", "0") == "0"
WHISPER_PROMPT = os.getenv("WHISPER_PROMPT", "Towing and roadside assistance. Year make model, location, issue, urgency. Keep transcripts concise.")
IGNORE_PATTERNS = [r"www\\.", r"http[s]?://", r"FEMA\\.gov", r"un\\.org", r"For more UN videos", r"For more information"]
VEHICLE_MAKES = r"(ford|chevy|chevrolet|gmc|ram|dodge|toyota|honda|nissan|jeep|bmw|mercedes|benz|kia|hyundai|subaru|vw|volkswagen|audi|lexus|infiniti|acura|mazda|buick|cadillac|chrysler|lincoln|volvo|porsche|jaguar|land rover|mini|mitsubishi|tesla)"
VEHICLE_TYPES = r"(car|truck|suv|van|pickup|tow ?truck|box ?truck)"
LOC_HINTS = r"(nashville|tennessee|i[- ]?\d+|hwy|highway|exit|rd|road|st|street|ave|avenue|blvd|pike|tn-?\d+)"
ISSUE_HINTS = r"(flat|tire|battery|dead|won'?t start|no start|engine|overheat|overheating|accident|lock(ed)? out|fuel|gas|out of gas|tow|winch|ditch|stuck)"
        FRAME_BYTES = 160  # 20 ms at 8k
    ACKS_ENABLED = os.getenv("ACKS_ENABLED", "1") == "1"
    ACK_DELAY_MS = int(os.getenv("ACK_DELAY_MS", "420"))
    ACK_MAX_DURATION_S = float(os.getenv("ACK_MAX_DURATION_S", "0.7"))
    ACK_MIN_CONTENT_CHARS = int(os.getenv("ACK_MIN_CONTENT_CHARS", "3"))
    ACK_PROMPT_DELAY_MS = int(os.getenv("ACK_PROMPT_DELAY_MS", "900"))  # delay after ack before main prompt
    SPEED_MIN = float(os.getenv("SPEED_MIN", "0.78"))
    SPEED_MAX = float(os.getenv("SPEED_MAX", "0.98"))

