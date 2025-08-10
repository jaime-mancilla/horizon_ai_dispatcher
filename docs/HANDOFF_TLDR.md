# Handoff TL;DR — 2025-08-10

**What changed:** Added FastAPI app (`app/server.py`), Twilio webhook stub, and docs for env vars.

**Next steps:** 
- Point your Twilio number's Voice webhook to `https://<your-ngrok-or-host>/twilio/voice` (POST).
- Replace TwiML <Say> with ElevenLabs audio via <Play> (requires hosting audio, or a media stream).

**Open questions:** 
- Do we prioritize a fast MVP using Twilio <Gather>/<Say>, or full media streams with ElevenLabs?
