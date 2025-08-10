# Known issues

- When accounts are new, OpenAI Whisper can throttle with 429s. The code backs off for 20s and rate-limits to ~17 requests/minute by chunking to 3s and spacing requests by 3.5s minimum.
- Twilio normally closes the media websocket. We avoid double-close to prevent an ASGI crash.
