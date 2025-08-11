# HOTFIX — Restores TTSSpeaker + Recording Bundle

This file restores the missing `TTSSpeaker` class and merges:
- Conversation Flow v1 (slot filling + barge-in controls)
- Call recording (in/out WAV) + `stt.txt` and `tts.txt`
- Download endpoints: `/debug/{callSid}` and `/recordings/{file}`

## Env
```
RECORD_CALL=1
RECORD_DIR=/tmp/recordings
TWILIO_RECORD_CALL=1      # optional Twilio native recording
```
