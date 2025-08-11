# Event-loop unblock patch

**What this fixes**
Your logs showed the 0.5s beep starting immediately but not finishing for ~22s,
while Whisper requests kept firing; later the TTS reply queued but didn't stream
before hangup. That means the event loop was being **blocked by synchronous Whisper
calls**, starving playback.

**Changes**
- Offload Whisper to a background thread via `asyncio.to_thread` (no loop blocking).
- Gate STT: skip too-quiet chunks and enforce a minimum gap between Whisper calls
  (`STT_RMS_GATE`, `STT_MIN_GAP_BETWEEN_CALLS_MS`).
- Stronger error handling in the TTS worker/streamer.
