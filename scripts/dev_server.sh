#!/usr/bin/env bash
set -euo pipefail
export APP_ENV=${APP_ENV:-development}
uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload
