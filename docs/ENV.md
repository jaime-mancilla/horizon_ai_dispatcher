# Horizon AI Dispatcher — Runtime ENV

## Required
- `OPENAI_API_KEY`
- `ELEVENLABS_API_KEY`
- `ELEVENLABS_VOICE_ID`
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_PHONE_NUMBER`

## Audio / Model
- `ELEVENLABS_OUTPUT_FORMAT` — `pcm_16000` (recommended) or `wav`
- `LLM_MODEL` — reserved for future agent (e.g., `gpt-5`)
- `WHISPER_PROMPT` — domain hint for transcription
- `STT_CHUNK_S` — Whisper chunk seconds (default `2.5`)

## TTS (playback)
- `TTS_GAIN_DB` — loudness boost (default `12` dB)
- `TTS_PAD_PRE_MS` — μ-law silence before speech (default `250` ms)
- `TTS_PAD_POST_MS` — μ-law silence after speech (default `300` ms)
- `TTS_MIN_GAP_MS` — minimum gap between bot replies (default `3500` ms)
- `TTS_BEEP_ON_CONNECT` — `1` to send a short tone on connect (debug)

## STT / VAD
- `VAD_RMS_THRESHOLD` — treat audio as “quiet” below this RMS (default `500`)
- `VAD_QUIET_MS` — quiet duration before speaking (default `300` ms)
