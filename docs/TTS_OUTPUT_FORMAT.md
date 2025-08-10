# ElevenLabs output format compatibility

Set one of:
- `ELEVENLABS_OUTPUT_FORMAT=wav` (recommended), or
- `ELEVENLABS_OUTPUT_FORMAT=pcm_16000`

The server requests `output_format` explicitly and converts audio to
μ-law @ 8 kHz, streaming 20 ms frames to Twilio.
