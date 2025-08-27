# Phase‑1d Clean Sweep (Refactor branch)

This patch does **not** change behavior. It cleans the split so each module
owns exactly one responsibility and removes accidental duplication.

## What's included
- `app/core/audio_utils.py` — helpers only (rms, parse_wav, to_wav, apply_gain_linear,
  peak_normalize, to_ulaw_8k_from_linear, silence_ulaw).
- `app/config.py` — single source of env-backed constants. Import from here instead
  of redefining in `server.py`.
- `scripts/fix_core_split.py` — trims extra classes/functions from core files.
- `scripts/sanity_check_repo.sh` — quick checks for endpoints-in-core and duplicates.

## How to apply
```bash
git checkout -b refactor/phase1d-clean
unzip -o hid_refactor_phase1d_clean.zip -d .
python scripts/fix_core_split.py --dry-run
python scripts/fix_core_split.py
bash scripts/sanity_check_repo.sh
git add app scripts docs
git commit -m "Refactor Phase‑1d: clean core split, centralize config, dedupe helpers"
git push origin refactor/phase1d-clean
```

## Recommended manual tidy (server.py)
- Remove the big inline **Config** block and replace it with:
  ```python
  import app.config as CFG
  from app.config import *
  ```
- Remove any duplicate helper functions (`_apply_gain_linear`, `_parse_wav`,
  `_peak_normalize`, `_to_ulaw_8k_from_linear`) and import from audio_utils:
  ```python
  from app.core.audio_utils import (
      rms as _rms,
      parse_wav as _parse_wav,
      to_wav as _to_wav,
      apply_gain_linear as _apply_gain_linear,
      peak_normalize as _peak_normalize,
      to_ulaw_8k_from_linear as _to_ulaw_8k_from_linear,
      silence_ulaw as _silence_ulaw,
  )
  ```

## Deploy (Render)
- Switch service to this branch → **Clear build cache → Deploy**.
- `GET /healthz` should return OK.
- Make a 60–90 s test call; if anything regresses, revert the commit.

## Rollback
Every trimmed file is backed up with `.bak_phase1d`. You can restore those
or simply revert the commit.
