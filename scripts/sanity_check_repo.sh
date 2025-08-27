#!/usr/bin/env bash
    set -euo pipefail
    echo "== Sanity: endpoints must NOT be in app/core =="?
    if grep -RnoE '@app\.(get|post)|def\s+twilio_voice_webhook|def\s+twilio_media_ws|def\s+healthz' app/core; then
      echo "ERROR: HTTP endpoints found inside app/core (should be in app/server.py)"
      exit 1
    else
      echo "OK: no HTTP endpoints under app/core"
    fi

    echo "== Sanity: one class per module (Recorder, STTBuffer, TTSSpeaker, DialogState) =="
    for c in Recorder STTBuffer TTSSpeaker DialogState; do
      echo -n "$c defs: "
      grep -Rno "class\\s\\+$c\\b" app | wc -l
    done

    echo "== Sanity: duplicate helper defs that belong in audio_utils? =="
    grep -RnoE 'def\\s+(_?apply_gain_linear|_?parse_wav|_?peak_normalize|_?to_ulaw_8k_from_linear)\\b' app | sed 's/^/  /'

    echo "== Sanity: server.py should import config rather than duplicate it =="
    if grep -nE '^\\s*[A-Z_]+\\s*=\\s*os\\.getenv' app/server.py; then
      echo "NOTE: Found env constants directly in server.py (consider importing from app.config)"
    else
      echo "OK: server.py defers to app.config"
    fi
