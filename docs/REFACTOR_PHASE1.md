# Refactor Phase 1 - Module Split (non-behavioral)
This patch moves major classes out of `app/server.py` into `app/core/`:
- `Recorder` -> `app/core/recorder.py`
- `STTBuffer` -> `app/core/stt.py` (with `set_openai_client()` setter)
- `TTSSpeaker` -> `app/core/tts.py`
- `DialogState` -> `app/core/dialog.py`
- audio helpers -> `app/core/audio_utils.py`

`app/server.py` now imports these and wires the OpenAI client via the setter.

Behavior should be unchanged. Rollback by restoring previous `app/server.py`.
