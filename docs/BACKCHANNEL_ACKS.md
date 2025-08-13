# Back-channel Acks (Track B Patch v1)

**What it does:** When the caller says a short but meaningful clip (<~0.7 s), the bot gives a tiny ack (“Okay.”/“Got it.”) about ~420 ms later—unless already speaking. This reduces awkward pauses and makes turn‑taking feel more human.

**Toggles/env**

```
ACKS_ENABLED=1
ACK_DELAY_MS=420
ACK_MAX_DURATION_S=0.7
ACK_MIN_CONTENT_CHARS=3
ACK_PROMPT_DELAY_MS=900
```

**Notes**
- The main prompt is delayed by `ACK_DELAY_MS + ACK_PROMPT_DELAY_MS` to avoid the ack consuming the TTS cooldown window.
- Acks never throw; if anything fails they silently no‑op.
- Rollback = `ACKS_ENABLED=0` and redeploy.
