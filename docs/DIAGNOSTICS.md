# Diagnostic patch: LOG_TTS_DEBUG + AUTO_TEST_TTS

**New env (optional):**
- `LOG_TTS_DEBUG=1` — logs when audio is queued, dequeued, and streaming progress.
- `AUTO_TEST_TTS=1` — plays a short test message *right after the beep*, without waiting for STT.

**Expected logs when working:**
- `[tts] queueing len=…` → `[tts-worker] dequeued …` → `[tts-stream] start …` → `[tts-stream] frames=…` (every ~25 frames) → `[tts] streamed … ulaw bytes to Twilio (mark=tts-finished)`
