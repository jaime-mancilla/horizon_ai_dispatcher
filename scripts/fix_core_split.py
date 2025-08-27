#!/usr/bin/env python3
import argparse, re, os, pathlib

TARGETS = {
    "app/core/audio_utils.py": {"keep_classes": [], "keep_funcs": [
        "rms","parse_wav","to_wav","apply_gain_linear","peak_normalize","to_ulaw_8k_from_linear","silence_ulaw"
    ]},
    "app/core/recorder.py":    {"keep_classes": ["Recorder"], "keep_funcs": []},
    "app/core/stt.py":         {"keep_classes": ["STTBuffer"], "keep_funcs": ["set_openai_client"]},
    "app/core/tts.py":         {"keep_classes": ["TTSSpeaker"], "keep_funcs": []},
    "app/core/dialog.py":      {"keep_classes": ["DialogState"], "keep_funcs": []},
}

DEF_RE = re.compile(r'^(def|class)\\s+([A-Za-z_][A-Za-z_0-9]*)\\b')
TOPLEVEL_RE = re.compile(r'^(def|class)\\s+')

def split_blocks(text):
    lines = text.splitlines(keepends=True)
    n = len(lines); i = 0; blocks = []
    while i < n:
        m = DEF_RE.match(lines[i])
        if m:
            kind, name = m.group(1), m.group(2)
            start = i; i += 1
            while i < n and not TOPLEVEL_RE.match(lines[i]):
                i += 1
            end = i
            blocks.append((kind, name, start, end))
        else:
            i += 1
    return lines, blocks

def filter_file(path, keep_classes, keep_funcs, dry_run=False):
    p = pathlib.Path(path)
    if not p.exists():
        print(f"[SKIP] {path} not found")
        return False
    original = p.read_text(encoding="utf-8")
    lines, blocks = split_blocks(original)
    to_remove = []
    for kind, name, start, end in blocks:
        if (kind == "class" and name in keep_classes) or (kind == "def" and name in keep_funcs):
            continue
        to_remove.append((start, end, kind, name))
    if not to_remove:
        print(f"[OK] {path}: nothing to trim")
        return False

    preview = ", ".join([f"{k} {n}" for _,_,k,n in to_remove[:8]])
    if len(to_remove) > 8: preview += ", …"
    print(f"[TRIM] {path}: removing {len(to_remove)} blocks -> {preview}")
    if dry_run: return True

    out = []; i = 0; rm = 0
    while i < len(lines):
        if rm < len(to_remove) and i == to_remove[rm][0]:
            i = to_remove[rm][1]; rm += 1
        else:
            out.append(lines[i]); i += 1
    p.with_suffix(p.suffix + ".bak_phase1d").write_text(original, encoding="utf-8")
    p.write_text("".join(out), encoding="utf-8")
    return True

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    changed = False
    for path, allow in TARGETS.items():
        c = filter_file(path, allow["keep_classes"], allow["keep_funcs"], dry_run=args.dry_run)
        changed = changed or c
    print("\\nDone.", "(dry run)" if args.dry_run else "")
    if not changed:
        print("No changes were necessary.")
