# Twilio Setup (Track B)

## A. Point Voice Webhook to TwiML
- Set your phone number's **Voice & Fax → A CALL COMES IN** to:
  - **Webhook**: POST to `https://<your-domain>/twilio/voice`

The webhook returns TwiML that starts a **Media Stream**:

```xml
<Response>
  <Start>
    <Stream url="wss://<your-domain>/twilio/media" />
  </Start>
  <Say>Thanks for calling Horizon Road Rescue...</Say>
  <Pause length="60" />
</Response>
```

## B. Verify Media Streams are reaching your server
- Call your Twilio number and watch your app logs.
- You should see `start` then many `media` events, and finally `stop`.

## C. Next steps (BIDI audio & LLM)
- Feed `media.payload` frames into a small buffer → send to STT (e.g., Whisper API).
- Generate TTS via ElevenLabs and stream PCM 8kHz frames **back** to Twilio by sending WebSocket messages of type `media`.
- Add basic state tracking (vehicle, location, issue, urgency) and short confirmations.
