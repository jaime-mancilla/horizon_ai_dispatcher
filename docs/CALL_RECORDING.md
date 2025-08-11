# Call Recording & Debug Bundle

Capture both sides of the call + transcripts:
- `*.in.wav`  – caller audio (8k WAV)
- `*.out.wav` – assistant audio
- `*.stt.txt` – Whisper transcripts
- `*.tts.txt` – lines spoken by TTS

## Enable
Add in Render:
```
RECORD_CALL=1
RECORD_DIR=/tmp/recordings        # optional
TWILIO_RECORD_CALL=1              # optional: Twilio native recording
```
Deploy, make a call, then check logs for:
```
[record] saved: {'zip': '/tmp/recordings/CAxxxxxxxx.zip', ...}
```

## Download
- `https://<host>/debug/CAxxxxxxxx` → ZIP
- or individual files via `/recordings/CAxxxxxxxx.in.wav` etc.

Note: Free plans have ephemeral disk; download soon after each test.
