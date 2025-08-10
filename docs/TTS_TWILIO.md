# TTS back to the caller (ulaw_8000) — 2025-08-10

This patch adds a minimal **talk-back** path:
- After the **first STT** result, we call **ElevenLabs** for **ulaw_8000** output,
  then **stream that audio back** to the caller over the same Twilio Media Stream.

## Env vars (Render → Environment)
- `ELEVENLABS_API_KEY`
- `ELEVENLABS_VOICE_ID` (a valid voice id)
- (already set) `OPENAI_API_KEY` for Whisper

## How it works
- We request `output_format="ulaw_8000"` so there is **no local transcoding**.
- We chunk the μ-law bytes into ~320-byte frames and send Twilio `event="media"` messages.
- Twilio buffers and plays them in order. Use a `mark` to signal end.

If you don't hear audio:
- Confirm your ElevenLabs plan supports **ulaw_8000**.
- Check logs for `[tts]` lines and HTTP status.
- Make sure your voice id is valid.
