# Patch: Back‑channel Acks + Pacing (v1)

## Files changed
- `app/server.py` — adds back‑channel ack scheduler and per‑utterance ElevenLabs speed/style overrides.
- `docs/BACKCHANNEL_ACKS.md` — feature notes and env.
- `docs/ADAPTIVE_PACING.md` — pacing notes and env.

## Deploy
1. Commit the changes.
2. Render → **Clear build cache** → **Deploy**.
3. Hit `/healthz` → should be OK.
4. Make one 60–90 s test call with 2–3 overlaps.

## Ask from you
- Send the first ~40 log lines and the `/debug/<CallSID>.zip`.
- Tell me how the **feel** changed: “more attentive?”, “too chatty?”, “still ping‑pong?”

## Rollback
- Set `ACKS_ENABLED=0` and redeploy; the patch becomes passive.
