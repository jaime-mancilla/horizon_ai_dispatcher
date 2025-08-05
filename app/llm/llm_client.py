# app/llm/llm_client.py
import aiohttp
import asyncio
from app.utils.config_loader import load_config

config = load_config()
OPENAI_API_KEY = config["openai"]["api_key"]
SYSTEM_PROMPT = config["llm"]["system_prompt"]

class LLMClient:
    def __init__(self):
        self.api_url = "https://api.openai.com/v1/chat/completions"

    async def get_response(self, user_input: str) -> str:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gpt-4",  # or "gpt-3.5-turbo"
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            "temperature": 0.7
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    raise Exception(f"LLM API error {resp.status}: {await resp.text()}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"].strip()

