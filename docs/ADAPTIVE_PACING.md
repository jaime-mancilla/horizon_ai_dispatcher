# Sentiment‑Aware Pacing v0

**What it does:** Nudges ElevenLabs `speed` and `style` per utterance based on simple keywords in the last user clip.

- Urgent terms → a touch faster (`+0.03`) and slightly more energetic style (`0.35`).
- Frustrated terms → a touch slower (`−0.03`) and calmer style (`0.10`).
- Uncertain terms → slightly slower (`−0.02`) and gentle style (`0.20`).

**Env (bounds)**

```
SPEED_MIN=0.78
SPEED_MAX=0.98
```

**Rollback:** leave the code, just set `ELEVEN_SPEED` to your baseline (e.g., `0.86`) and the nudge will net out small, or remove the env bounds.
