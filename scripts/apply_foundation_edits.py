
#!/usr/bin/env python3
"""
Phase-1e: Centralize config imports & audio helpers in server.py, remove duplicates.

What this script does (idempotent):
  1) Inserts imports for app.config and app.core.audio_utils.
  2) Comments out env-backed constant assignments in server.py (moved to app/config.py).
  3) Removes duplicate helper functions (_apply_gain_linear, _parse_wav, _peak_normalize, _to_ulaw_8k_from_linear).
  4) Ensures STT set_openai_client() is called once after _openai_client is created.
  5) Writes a backup: app/server.py.bak_phase1e

Run from repo root:
    python scripts/apply_foundation_edits.py
"""
import re, pathlib, sys

SERVER = pathlib.Path("app/server.py")
if not SERVER.exists():
    print("ERROR: app/server.py not found. Run from repo root.")
    sys.exit(1)

text = SERVER.read_text(encoding="utf-8")
original = text

# 1) Ensure imports
import_block = """
# === Phase-1e: centralized config & audio helpers ===
try:
    import app.config as CFG  # central config
    from app.config import *  # expose constants unchanged
    from app.core.audio_utils import (
        rms as _rms,
        parse_wav as _parse_wav,
        to_wav as _to_wav,
        apply_gain_linear as _apply_gain_linear,
        peak_normalize as _peak_normalize,
        to_ulaw_8k_from_linear as _to_ulaw_8k_from_linear,
        silence_ulaw as _silence_ulaw,
    )
except Exception as _e:
    # If imports fail during tooling, don't crash the script.
    pass
# === end Phase-1e ===
""" + "\n\n"

if "app.core.audio_utils import" not in text or "app.config as CFG" not in text:
    # insert after last top-level import block
    m = list(re.finditer(r'^(from\s+\S+\s+import\s+.+|import\s+\S+)\s*$', text, flags=re.M))
    insert_at = m[-1].end() if m else 0
    text = text[:insert_at] + "\n\n" + import_block + text[insert_at:]
    print("[edit] inserted Phase-1e imports")

# 2) Comment out env-backed constants
def comment_env_lines(s):
    out = []
    for line in s.splitlines(True):
        if re.match(r'^\s*[A-Z_]{3,}\s*=\s*.*os\.getenv\(', line):
            out.append("# PHASE1E moved-to-config: " + line)
        else:
            out.append(line)
    return "".join(out)
text = comment_env_lines(text)

# 3) Remove duplicate helper defs in server.py
helper_names = [
    "_apply_gain_linear", "apply_gain_linear",
    "_parse_wav", "parse_wav",
    "_peak_normalize", "peak_normalize",
    "_to_ulaw_8k_from_linear", "to_ulaw_8k_from_linear",
    "_silence_ulaw", "silence_ulaw"
]
for name in helper_names:
    # remove top-level 'def name(...): ...' block
    pattern = re.compile(
        r'^\s*def\s+' + re.escape(name) + r'\s*\([^)]*\):\s*\n'  # signature
        r'(?:\s+.*\n)+'                                          # body
        , flags=re.M)
    # Remove only first occurrence per helper to avoid over-stripping
    text_new = pattern.sub("", text, count=1)
    if text_new != text:
        text = text_new
        print(f"[edit] removed duplicate helper def: {name}")

# 4) Ensure set_openai_client is called
if "set_openai_client" not in text:
    # add import alias
    text = re.sub(
        r'^(from\s+app\.core\.stt\s+import\s+STTBuffer.*)$',
        r'\1\nfrom app.core.stt import set_openai_client as stt_set_client',
        text, flags=re.M
    )
    if "stt_set_client" not in text:
        # If the above pattern didn't match (import line is different), append a new import
        m = list(re.finditer(r'^(from\s+\S+\s+import\s+.+|import\s+\S+)\s*$', text, flags=re.M))
        insert_at = m[-1].end() if m else 0
        text = text[:insert_at] + "\nfrom app.core.stt import set_openai_client as stt_set_client\n" + text[insert_at:]
        print("[edit] inserted stt_set_client import")

if "_openai_client" in text and "stt_set_client(" not in text:
    # place the call right after _openai_client assignment
    text = re.sub(
        r'(_openai_client\s*=\s*.*\n)',
        r'\1# Phase-1e: wire OpenAI client into STT\nstt_set_client(_openai_client)\n',
        text, count=1
    )
    print("[edit] added stt_set_client(_openai_client)")

# Write backup and new file
if text != original:
    SERVER.with_suffix(SERVER.suffix + ".bak_phase1e").write_text(original, encoding="utf-8")
    SERVER.write_text(text, encoding="utf-8")
    print("Done. Backup written to app/server.py.bak_phase1e")
else:
    print("No edits necessary (already clean).")
