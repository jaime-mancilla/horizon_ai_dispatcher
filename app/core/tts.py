# Auto-generated split module from server.py (modularization phase 1)
import os, asyncio, base64, json, logging, time, math, re
from io import BytesIO
import wave, audioop, contextlib
from pathlib import Path
import httpx
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, FileResponse
# OpenAI client is set via setter when needed

from .audio_utils import _rms, convert_elevenlabs_to_ulaw8k, _apply_gain_linear, _peak_normalize, _parse_wav, _to_ulaw_8k_from_linear
class TTSSpeaker:
    def __init__(self, ws: WebSocket, stream_sid: str, recorder: Recorder | None):
        self.ws = ws
        self.stream_sid = stream_sid
        self.q = asyncio.Queue()
        self._task = asyncio.create_task(self._worker())
        self._last_reply_at = 0.0
        self._last_hash = None
        self.playing = False
        self.cancel_event = asyncio.Event()
        self.after_barge = False
        self.recorder = recorder

        # ducking
        self._duck_until = 0.0  # monotonic seconds

    def duck_for(self, ms: int):
        self._duck_until = max(self._duck_until, time.monotonic() + (ms / 1000.0))

    async def _worker(self):
        try:
            while True:
                ulaw, mark_name, protect, plain_text = await self.q.get()
                if LOG_TTS_DEBUG:
                    log.info(f"[tts-worker] dequeued len={len(ulaw)} mark={mark_name} protect={protect} qsize={self.q.qsize()}")
                await self._stream_ulaw(ulaw, mark_name, protect)
                self.q.task_done()
        except asyncio.CancelledError:
            if LOG_TTS_DEBUG:
                log.info("[tts-worker] cancelled")

    def _apply_duck_and_fade(self, chunk: bytes, fade_gain):
        # Apply ducking and optional fade multiplier to a 20 ms μ-law frame.
        need_process = (fade_gain is not None) or (time.monotonic() < self._duck_until)
        if not need_process:
            return chunk
        lin = audioop.ulaw2lin(chunk, 2)
        if time.monotonic() < self._duck_until:
            duck_factor = pow(10.0, -TTS_DUCK_DB / 20.0)
            try:
                lin = audioop.mul(lin, 2, duck_factor)
            except Exception:
                pass
        if fade_gain is not None:
            try:
                lin = audioop.mul(lin, 2, float(fade_gain))
            except Exception:
                pass
        return audioop.lin2ulaw(lin, 2)

    async def _stream_ulaw(self, ulaw: bytes, mark_name: str, protect: bool):
        FRAME_BYTES = 160  # 20 ms at 8k
        total = 0
        self.playing = True
        self.cancel_event.clear()
        frames_played = 0

        # Precompute fade steps if cancel_event gets set
        fade_frames_total = max(1, int(TTS_BARGE_FADE_MS / 20))
        post_silence_frames = int(TTS_BARGE_POST_SILENCE_MS / 20)

        while frames_played * FRAME_BYTES < len(ulaw):
            start = frames_played * FRAME_BYTES
            end = start + FRAME_BYTES
            raw = ulaw[start:end]

            # Should we start fading?
            fade_gain = None
            played_ms = frames_played * 20
            if self.cancel_event.is_set() and ENABLE_BARGE_IN and not protect and played_ms >= BARGE_GRACE_MS:
                # linear fade from 1.0 to 0.0 across fade_frames_total frames
                idx = min(frames_played, fade_frames_total - 1)
                gain = max(0.0, 1.0 - (idx / float(fade_frames_total)))
                fade_gain = gain

            chunk = self._apply_duck_and_fade(raw, fade_gain)

            if self.recorder and RECORD_CALL:
                self.recorder.add_out_ulaw_chunk(chunk)

            payload = base64.b64encode(chunk).decode("ascii")
            msg = {"event": "media", "streamSid": self.stream_sid, "media": {"payload": payload}}
            await self.ws.send_text(json.dumps(msg))
            total += len(chunk)
            frames_played += 1

            # If we finished fading after barge, stop early and pad silence
            if fade_gain is not None and frames_played >= fade_frames_total:
                if post_silence_frames > 0:
                    pad = _silence_ulaw(post_silence_frames * 20)
                    payload = base64.b64encode(pad).decode("ascii")
                    await self.ws.send_text(json.dumps({"event": "media", "streamSid": self.stream_sid, "media": {"payload": payload}}))
                    total += len(pad)
                self.playing = False
                self.after_barge = True
                log.info(f"[tts] soft-barged after {total} bytes (fade {TTS_BARGE_FADE_MS} ms, pad {TTS_BARGE_POST_SILENCE_MS} ms)")
                return

            await asyncio.sleep(0.02)

        self.playing = False
        await self.ws.send_text(json.dumps({"event": "mark", "streamSid": self.stream_sid, "mark": {"name": mark_name}}))
        log.info(f"[tts] streamed {total} ulaw bytes to Twilio (mark={mark_name})")

    async def enqueue_text(self, text: str, protect: bool = False):
        now = time.monotonic()
        min_gap = TTS_MIN_GAP_AFTER_BARGE_MS if self.after_barge else TTS_MIN_GAP_MS
        if (now - self._last_reply_at) * 1000 < min_gap:
            log.info("[tts] suppressed (cooldown)"); return
        h = hash(text.strip().lower())
        if h == self._last_hash:
            log.info("[tts] suppressed (dedupe)"); return
        self._last_hash = h; self._last_reply_at = now

        data, fmt = await elevenlabs_tts_bytes(text)
        ulaw = convert_elevenlabs_to_ulaw8k(data, fmt, gain_db=TTS_GAIN_DB)
        if self.recorder and RECORD_CALL:
            self.recorder.add_tts(text)
        await self.q.put((ulaw, "tts-finished", protect, text))

    async def beep(self):
        if os.getenv("TTS_BEEP_ON_CONNECT", "0") != "1": return
        sr = 8000; n = int(sr * 0.5); amp = int(32767 * 0.20)
        pcm = bytearray()
        for i in range(n):
            s = int(amp * math.sin(2 * math.pi * 880.0 * (i / sr)))
            pcm.extend(s.to_bytes(2, byteorder="little", signed=True))
        ulaw = audioop.lin2ulaw(bytes(pcm), 2)
        await self.q.put((ulaw, "beep", False, ""))

    async def close(self):
        self._task.cancel()
        with contextlib.suppress(Exception):
            await self._task

# ---------- WebSocket ----------
@app.websocket("/twilio/media")
async def twilio_media_ws(ws: WebSocket):
    await ws.accept(subprotocol="audio")
    frames = 0
    speaker = None
    state = DialogState()
    recorder = None
    call_sid = None

    def on_text(text: str, pcm: bytes):
        nonlocal state, speaker, recorder
        state.update_from_text(text)
        need = state.need()

        if need is not None:
            prompt = state.next_prompt()
        else:
            prompt = "Thanks. I can get a truck headed your way. What’s a good callback number in case we get disconnected?"

        protect_first = (not state.first_reply_sent) and FIRST_REPLY_NO_BARGE
        state.first_reply_sent = True
        asyncio.create_task(speaker.enqueue_text(prompt, protect=protect_first))

    stt = STTBuffer(on_text=on_text)  # set recorder after 'start'
    log.info("WS accepted (subprotocol=audio): /twilio/media")

    try:
        while True:
            try:
                message = await ws.receive()
            except WebSocketDisconnect:
                log.info("WS disconnect (peer)"); break
            except RuntimeError as e:
                log.info(f"WS receive after disconnect: {e}"); break

            if message["type"] == "websocket.receive":
                payload_text = message.get("text") or (message.get("bytes") or b"").decode("utf-8","ignore")
                if not payload_text: continue
                try:
                    data = json.loads(payload_text)
                except Exception:
                    continue

                event = data.get("event")
                if event == "start":
                    stream_sid = data.get("start", {}).get("streamSid")
                    call_sid = data.get("start", {}).get("callSid") or stream_sid
                    log.info(f"[media] start callSid={call_sid} streamSid={stream_sid}")
                    recorder = Recorder(call_sid) if RECORD_CALL else None
                    stt.recorder = recorder
                    speaker = TTSSpeaker(ws, stream_sid, recorder)
                    asyncio.create_task(speaker.beep())
                    if AUTO_TEST_TTS:
                        asyncio.create_task(speaker.enqueue_text("System check. You should hear this test phrase.", protect=True))
                elif event == "media":
                    payload_b64 = data.get("media", {}).get("payload", "")
                    # Soft-duck when caller speaks; only hard-barge after grace
                    if speaker and speaker.playing and ENABLE_BARGE_IN:
                        try:
                            ulaw = base64.b64decode(payload_b64)
                            lin = audioop.ulaw2lin(ulaw, BYTES_PER_SAMPLE)
                            if _rms(lin) > VAD_RMS_THRESHOLD:
                                speaker.duck_for(TTS_DUCK_DECAY_MS)
                                speaker.cancel_event.set()
                        except Exception:
                            pass
                    stt.add_ulaw_b64(payload_b64)
                    frames += 1
                    if frames % 25 == 0:
                        log.info(f"[media] frames={frames}")
                elif event == "mark":
                    pass
                elif event == "stop":
                    log.info(f"[media] stop after frames={frames}")
                    stt.finish(); break
    finally:
        if speaker: await speaker.close()
        if recorder and RECORD_CALL:
            paths = recorder.finalize_files()
            log.info(f"[record] saved: {paths}")
        try:
            if ws.application_state == WebSocketState.CONNECTED:
                await ws.close()
        except Exception:
            pass
        log.info("WS closed")

# ---------- Artifact download endpoints ----------
@app.get("/recordings/{name}")
async def get_recording(name: str):
    base = Path(RECORD_DIR) / name
    if not base.exists() or not base.is_file():
        for suf in (".in.wav", ".out.wav", ".zip", ".stt.txt", ".tts.txt"):
            cand = base.with_suffix(suf)
            if cand.exists():
                base = cand; break
    if not base.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(base))

@app.get("/debug/{call_id}")
async def get_debug_bundle(call_id: str):
    base = Path(RECORD_DIR) / call_id
    zip_path = base.with_suffix(".zip")
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(zip_path))
