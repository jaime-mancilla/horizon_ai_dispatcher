# TTS → Twilio (clean playback)

This server:
- Requests **WAV** from ElevenLabs
- Converts to **8 kHz mono** 16‑bit PCM
- Encodes to **μ-law (PCMU)** and streams to Twilio as **20 ms frames**
- Paces frames in real time to avoid artifacts

## Env vars
- `ELEVENLABS_API_KEY`
- `ELEVENLABS_VOICE_ID`
- (optional) `ELEVENLABS_MODEL_ID` (defaults to `eleven_monolingual_v1`)

## Notes
- Keep `accept: audio/wav` when calling ElevenLabs; we convert locally.
- Twilio expects base64 μ-law frames in `{"event":"media","streamSid":"...","media":{"payload":"..."}}`.
- We include `track: "outbound"` and a `mark` at the end for clarity.
