# Known issues & stability choices — 2025-08-10

- We **throttle Whisper** by batching ~2.0s and guarding with an `inflight` flag.
  This reduces 429s and load while keeping latency acceptable.
- We catch both `WebSocketDisconnect` and the Starlette `RuntimeError` raised when
  `receive()` is called after a disconnect, and avoid double-closing the socket.
- TwiML includes a short `<Say>` before `<Connect>` so callers hear something while
  the stream initializes.
