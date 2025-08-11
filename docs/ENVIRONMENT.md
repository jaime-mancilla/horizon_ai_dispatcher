# Environment Variables (Horizon AI Dispatcher)

> Keep real secrets in Render's Environment (not in Git). Commit only `.env.example` for reference.

## 1) API Keys (secrets)
- `OPENAI_API_KEY` – for Whisper
- `ELEVENLABS_API_KEY` – for TTS
- `ELEVENLABS_VOICE_ID` – chosen voice
- (Optional) `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` – only if you call Twilio REST from the app

## 2) Telephony / I/O
- `ELEVENLABS_OUTPUT_FORMAT` = `pcm_16000` | `wav`
- `TTS_BEEP_ON_CONNECT` = `1` to play a half‑second 880 Hz beep after connect

## 3) STT (Whisper)
- `STT_CHUNK_S` (default 1.4)
- `STT_MIN_MS` (default 600)
- `STT_FLUSH_ON_SILENCE` (1/0)
- `VAD_RMS_THRESHOLD` (default 700)
- `VAD_QUIET_MS` (default 250)
- `STT_RMS_GATE` (default 350) – skip too‑quiet chunks
- `STT_MIN_GAP_BETWEEN_CALLS_MS` (default 800)

## 4) TTS (levels & pacing)
- `TTS_GAIN_DB` (default 6)
- `TTS_PEAK_TARGET` (default 0.78)
- `TTS_PAD_PRE_MS`, `TTS_PAD_POST_MS` (600 / 500)
- `TTS_MIN_GAP_MS` (2400), `TTS_MIN_GAP_AFTER_BARGE_MS` (1200)

## 5) ElevenLabs voice quality
- `ELEVEN_LATENCY_MODE` (0–3, default 1)
- `ELEVEN_SPEED` (0.80–1.10, default 0.95)
- `ELEVEN_STABILITY` (0–1, default 0.5)
- `ELEVEN_SIMILARITY` (0–1, default 0.75)
- `ELEVEN_STYLE` (0–1, default 0.35)
- `ELEVEN_SPEAKER_BOOST` (1/0, default 1)

## 6) Turn-taking / barge-in
- `ENABLE_BARGE_IN` (1/0)
- `BARGE_GRACE_MS` (default 900)
- `FIRST_REPLY_NO_BARGE` (1/0, default 1)
- `FIRST_REPLY_WAIT_FOR_QUIET` (0/1)

## 7) Diagnostics
- `LOG_TTS_DEBUG` (1/0)
- `AUTO_TEST_TTS` (1/0)

## 8) Prompting
- `WHISPER_PROMPT` – domain hints to improve recognition
