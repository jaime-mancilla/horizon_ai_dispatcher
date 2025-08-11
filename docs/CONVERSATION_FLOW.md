# Conversational Flow v1 (HID)

Adds:
- ElevenLabs quality controls (`ELEVEN_LATENCY_MODE`, `ELEVEN_SPEED`, `ELEVEN_STABILITY`, `ELEVEN_SIMILARITY`, `ELEVEN_STYLE`, `ELEVEN_SPEAKER_BOOST`)
- First-reply barge protection (`FIRST_REPLY_NO_BARGE=1`)
- Slot-filling (vehicle → location → issue → urgency) with rotating rephrases
- Softer PSTN audio defaults and padding

See `docs/ENVIRONMENT.md` for full list of env vars.
