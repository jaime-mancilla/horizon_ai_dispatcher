# First-reply reliability
- This patch guarantees the **first** TTS reply will play after we get the first valid transcript.
- Subsequent replies are still debounced/deduped. You can re-enable a quiet gate by setting:
  - `FIRST_REPLY_WAIT_FOR_QUIET=1`
