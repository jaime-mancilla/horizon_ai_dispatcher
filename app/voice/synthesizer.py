# app/voice/synthesizer.py

import aiohttp
import asyncio
from app.utils.config_loader import load_config

config = load_config()
ELEVEN_API_KEY = config["elevenlabs"]["api_key"]
VOICE_ID = config["elevenlabs"]["voice_id"]
MODEL_ID = config["elevenlabs"]["model"]

class Synthesizer:
    def __init__(self):
        self.api_url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

    async def synthesize(self, text: str) -> bytes:
        headers = {
            "xi-api-key": ELEVEN_API_KEY,
            "Content-Type": "application/json"
        }

        payload = {
            "text": text,
            "model_id": MODEL_ID,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    raise Exception(f"ElevenLabs error: {resp.status} - {await resp.text()}")
                return await resp.read()
