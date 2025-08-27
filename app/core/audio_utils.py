"""Audio helpers for Horizon AI Dispatcher (utilities only).

This module intentionally contains **no** FastAPI endpoints and **no** classes.
It provides low-level helpers used by Recorder/STT/TTS.
"""
from io import BytesIO
from pathlib import Path
import audioop, wave

# ---- simple metrics ----

def rms(pcm16: bytes) -> float:
    """RMS of 16‑bit mono PCM bytes (0..32767)."""
    try:
        return audioop.rms(pcm16, 2)
    except Exception:
        return 0.0

# ---- WAV helpers ----

def parse_wav(data: bytes):
    """Return (sampwidth, samplerate, nchannels, frames_bytes)."""
    with wave.open(BytesIO(data), "rb") as wf:
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        ch = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
    return sw, sr, ch, frames

def to_wav(path: Path, pcm16: bytes, sr: int = 8000):
    path = Path(path)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(pcm16)

# ---- transforms ----

def apply_gain_linear(frames: bytes, sampwidth: int, gain_db: float) -> bytes:
    """Apply linear gain (dB) to PCM frames of width `sampwidth`."""
    if not gain_db:
        return frames
    try:
        factor = pow(10.0, gain_db / 20.0)
        return audioop.mul(frames, sampwidth, factor)
    except Exception:
        return frames

def peak_normalize(pcm16: bytes, target_ratio: float) -> bytes:
    """Scale PCM16 so peak matches target_ratio of full scale."""
    if not pcm16:
        return pcm16
    peak = 1
    n = len(pcm16)
    # find peak
    for i in range(0, n, 2):
        s = int.from_bytes(pcm16[i:i+2], "little", signed=True)
        if abs(s) > peak: peak = abs(s)
    max_allowed = int(32767 * target_ratio)
    if peak <= 0 or peak == max_allowed:
        return pcm16
    scale = max_allowed / float(peak)
    out = bytearray(n)
    for i in range(0, n, 2):
        s = int.from_bytes(pcm16[i:i+2], "little", signed=True)
        s = int(s * scale)
        if s > 32767: s = 32767
        if s < -32768: s = -32768
        out[i:i+2] = s.to_bytes(2, "little", signed=True)
    return bytes(out)

def to_ulaw_8k_from_linear(frames: bytes, sw: int, nch: int, gain_db: float = 0.0) -> bytes:
    """Convert linear PCM -> μ‑law 8k mono, with optional gain."""
    if not frames:
        return b""
    try:
        # mixdown to mono if needed
        if nch == 2:
            frames = audioop.tomono(frames, sw, 0.5, 0.5)
            nch = 1
        sr = 8000
        # resample to 8k if needed
        if sr and sr != 8000:
            frames, _ = audioop.ratecv(frames, sw, 1, sr, 8000, None)
        # normalize/gain
        frames = peak_normalize(frames, 0.65)  # conservative
        if gain_db:
            frames = apply_gain_linear(frames, sw, gain_db)
        # μ‑law
        return audioop.lin2ulaw(frames, sw)
    except Exception:
        # last resort: return silence of same duration?
        # here we choose empty to avoid noise
        return b""

# ---- convenience ----

def silence_ulaw(ms: int) -> bytes:
    samples = int(8000 * ms / 1000)
    return bytes([0xFF]) * samples
