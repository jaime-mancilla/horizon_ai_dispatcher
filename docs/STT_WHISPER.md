# Streaming STT (Whisper) — 2025-08-10

This patch adds a micro-batching STT pipeline:
- Twilio sends `audio/x-mulaw; rate=8000` frames over WebSocket.
- We decode mu-law → 16-bit linear PCM in Python (`audioop.ulaw2lin`).
- We buffer ~1s and send a small WAV to **OpenAI Whisper**.
- Recognized text is logged like: `[stt] your words here`.

## Setup
1. Add **OPENAI_API_KEY** to your env (Render → Environment tab).
2. `pip install -r requirements.stt.txt` (or add to your pyproject).

## Test
- Visit `/healthz` to confirm server is live.
- Call your Twilio number; watch **Render → Logs → Live tail** for `[stt] ...` lines.

## Notes
- We use `<Connect><Stream>` to prepare for bidirectional audio later.
- Next step: add **TTS** (ElevenLabs) and send mulaw/8000 audio back to Twilio via `event=media`.
