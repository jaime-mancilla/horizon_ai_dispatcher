# Soft Barge & Ducking Patch

This patch makes barge-in sound natural:
- **Ducking:** while the caller speaks, the bot is reduced by `TTS_DUCK_DB` (default 6 dB).
- **Soft barge:** after grace, we **fade out** the bot over `TTS_BARGE_FADE_MS` (default 140 ms),
  then add `TTS_BARGE_POST_SILENCE_MS` (default 180 ms) of silence to avoid truncation pops.
- Keeps call recording bundle (`/debug/{CallSID}`) and conversation flow prompts.

## New ENV keys
```
# Soft-barge
TTS_BARGE_FADE_MS=140
TTS_BARGE_POST_SILENCE_MS=180
TTS_DUCK_DB=6
TTS_DUCK_DECAY_MS=800

# Existing (recommended)
BARGE_GRACE_MS=1500
TTS_MIN_GAP_AFTER_BARGE_MS=2200
TTS_PEAK_TARGET=0.65
VAD_RMS_THRESHOLD=260
STT_MIN_GAP_BETWEEN_CALLS_MS=1200
```
