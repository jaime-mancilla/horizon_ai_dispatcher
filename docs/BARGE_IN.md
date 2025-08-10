# Barge-in + Loudness Normalization + Faster First Reply

**New env (optional):**
- `ENABLE_BARGE_IN=1` (default) – abort playback when caller starts speaking.
- `TTS_PEAK_TARGET=0.82` – normalizes TTS peaks before μ-law.
- `TTS_MIN_GAP_AFTER_BARGE_MS=900` – shorter cooldown after an interrupted line.
- `STT_CHUNK_S=1.6`, `STT_MIN_MS=700`, `STT_FLUSH_ON_SILENCE=1` – earlier endpointing.
- Keep `TTS_GAIN_DB` (try 6–8 with hotter ElevenLabs voices).

**Behavior:**
- If user talks while TTS is playing, remaining audio is **dropped** immediately.
- Volume is normalized and padded to avoid clipping and chopped words.
- First reply happens sooner thanks to adaptive endpointing.
