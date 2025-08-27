#!/usr/bin/env bash
set -euo pipefail
echo "== Phase-1e sanity: config centralization =="
if grep -nE '^\s*[A-Z_]{3,}\s*=\s*.*os\.getenv\(' app/server.py; then
  echo "WARN: Found env-backed constants still in app/server.py"
else
  echo "OK: server.py uses app.config only"
fi
echo "== Phase-1e sanity: audio helpers shouldn't be defined in server.py =="
if grep -nE 'def\s+(_?apply_gain_linear|_?parse_wav|_?peak_normalize|_?to_ulaw_8k_from_linear|_?silence_ulaw)\s*\(' app/server.py; then
  echo "WARN: Found helper definitions in server.py (should import from app.core.audio_utils)"
else
  echo "OK: server.py imports audio helpers"
fi
