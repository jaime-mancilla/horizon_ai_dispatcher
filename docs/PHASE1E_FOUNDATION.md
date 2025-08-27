
# Phase-1e Foundation Patch

**Goal:** Finish the clean foundation by centralizing config imports and audio helpers,
and ensuring STT is wired to the OpenAI client. Behavior is unchanged.

## What this patch does
- Adds imports in `app/server.py`:
  ```python
  import app.config as CFG
  from app.config import *
  from app.core.audio_utils import (
      rms as _rms, parse_wav as _parse_wav, to_wav as _to_wav,
      apply_gain_linear as _apply_gain_linear,
      peak_normalize as _peak_normalize,
      to_ulaw_8k_from_linear as _to_ulaw_8k_from_linear,
      silence_ulaw as _silence_ulaw,
  )
  ```
- Comments out any remaining `os.getenv` constant lines in `server.py`.
- Removes duplicate helper definitions from `server.py`.
- Ensures `stt_set_client(_openai_client)` is called once.

## How to apply
```bash
git checkout -b refactor/phase1e-foundation
unzip -o hid_refactor_phase1e_foundation.zip -d .
python scripts/apply_foundation_edits.py
bash scripts/sanity_check_phase1e.sh
git commit -am "Refactor Phase-1e: centralize config & audio helpers; wire STT client"
git push origin refactor/phase1e-foundation
```

## Deploy (Render)
- Switch service to `refactor/phase1e-foundation` → **Clear build cache → Deploy**.
- `GET /healthz` should be OK.
- Make one 60–90 s test call and share first ~40 log lines.

## Rollback
- A backup `app/server.py.bak_phase1e` is written. Restore it or revert the commit.
