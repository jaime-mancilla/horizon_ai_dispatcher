# Environment Variables

| Name | Required | Description |
|---|---|---|
| OPENAI_API_KEY | yes | OpenAI API key (used by LLM and Whisper API) |
| WHISPER_API_KEY | yes | If separate from OpenAI key; else reuse OPENAI_API_KEY |
| ELEVENLABS_API_KEY | yes | ElevenLabs API key |
| ELEVENLABS_VOICE_ID | yes | ElevenLabs voice id to use |
| TWILIO_ACCOUNT_SID | yes (if using Twilio) | Twilio account SID |
| TWILIO_AUTH_TOKEN | yes (if using Twilio) | Twilio auth token |
| TWILIO_PHONE_NUMBER | yes (if using Twilio) | Your Twilio number in E.164 (+1...) |
| APP_ENV | no | development | staging | production |
