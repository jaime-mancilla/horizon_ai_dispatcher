# Recorder extracted from server.py
import os, base64, json, logging, time, zipfile
from io import BytesIO
import wave, audioop, contextlib
from pathlib import Path
from app.config import *

class Recorder:
    def __init__(self, call_sid: str):
        self.call_sid = call_sid or f"call-{int(time.time())}"
        self.in_ulaw = bytearray()
        self.out_ulaw = bytearray()
        self.stt_lines = []
        self.tts_lines = []

    def add_in_ulaw_b64(self, b64: str):
        try:
            self.in_ulaw.extend(base64.b64decode(b64))
        except Exception:
            pass

    def add_out_ulaw_chunk(self, ulaw: bytes):
        self.out_ulaw.extend(ulaw)

    def add_stt(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.stt_lines.append(f"[{ts}] {text}")

    def add_tts(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.tts_lines.append(f"[{ts}] {text}")

    def finalize_files(self) -> dict:
        def ulaw_to_pcm16(ulaw: bytes) -> bytes:
            try:
                return audioop.ulaw2lin(ulaw, 2)
            except Exception:
                return b""

        in_pcm = ulaw_to_pcm16(bytes(self.in_ulaw))
        out_pcm = ulaw_to_pcm16(bytes(self.out_ulaw))

        base = Path(RECORD_DIR) / self.call_sid
        in_path = base.with_suffix(".in.wav")
        out_path = base.with_suffix(".out.wav")
        stt_path = base.with_suffix(".stt.txt")
        tts_path = base.with_suffix(".tts.txt")
        bundle_path = base.with_suffix(".zip")

        _to_wav(in_path, in_pcm, 8000)
        _to_wav(out_path, out_pcm, 8000)
        with open(stt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.stt_lines))
        with open(tts_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.tts_lines))

        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in (in_path, out_path, stt_path, tts_path):
                zf.write(p, arcname=p.name)

        return {"in_wav": str(in_path), "out_wav": str(out_path),
                "stt": str(stt_path), "tts": str(tts_path), "zip": str(bundle_path)}

# ---------- STT (non-blocking) ----------
IGNORE_PATTERNS = [r"www\\.", r"http[s]?://", r"FEMA\\.gov", r"un\\.org", r"For more UN videos", r"For more information"]


