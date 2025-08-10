# TTS gain + beep patch

- Adds `TTS_GAIN_DB` (default **12 dB**) to boost TTS loudness before μ-law.
- Removes unused `track` field in outbound `media` messages (Twilio doesn't require it).
- Optional beep to validate audio path:
  - Set `TTS_BEEP_ON_CONNECT=1` to send a 700 ms 880 Hz tone immediately after `start`.
  - Logs will show `[beep] sending ...` and `[media] other event: mark (beep)` when Twilio has finished playing it.
