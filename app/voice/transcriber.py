# app/voice/transcriber.py

import asyncio
import tempfile
import aiofiles
import aiohttp
from app.utils.config_loader import load_config

config = load_config()
WHISPER_API_KEY = config["whisper"]["api_key"]
WHISPER_API_URL = config["whisper"]["api_url"]  # e.g., https://api.whisper.com/v1/audio/transcriptions

class Transcriber:
    async def transcribe(self, audio_bytes: bytes) -> str:
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('file', audio_bytes, filename="input.wav", content_type='audio/wav')
            data.add_field('model', 'whisper-1')
            data.add_field('language', 'en')

            headers = {
                "Authorization": f"Bearer {WHISPER_API_KEY}"
            }

            async with session.post(WHISPER_API_URL, headers=headers, data=data) as resp:
                if resp.status != 200:
                    raise Exception(f"Whisper API error: {resp.status} - {await resp.text()}")
                result = await resp.json()
                return result.get("text", "")
