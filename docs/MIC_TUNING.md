# BARGE_GRACE_MS + Ungated Replies

**Env (recommended):**
- `ENABLE_BARGE_IN=1`
- `BARGE_GRACE_MS=250` (adjust 200–350) – minimum playback time before barge-in can cancel
- `TTS_GAIN_DB=8` (tune ±2 dB), `TTS_PEAK_TARGET=0.82` – consistent loudness
- `STT_CHUNK_S=1.6`, `STT_MIN_MS=700`, `STT_FLUSH_ON_SILENCE=1` – earlier endpointing
- If barge-in cancels too easily, raise `VAD_RMS_THRESHOLD` to 700–800

**Behavior:**
- First reply always plays after the first transcript.
- Later replies are not gated on 'quiet' — barge-in handles interruptions.
- Volume normalized and μ-law padded to avoid clipping/chopped words.
