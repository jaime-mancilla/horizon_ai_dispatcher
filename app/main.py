# app/main.py
import asyncio
from app.voice.transcriber import Transcriber
from app.voice.synthesizer import Synthesizer
from app.llm.agent import LLMDispatcher

async def main():
    print("Starting Horizon AI Dispatcher...")

    transcriber = Transcriber()
    synthesizer = Synthesizer()
    dispatcher = LLMDispatcher(transcriber, synthesizer)

    await dispatcher.run()

if __name__ == "__main__":
    asyncio.run(main())
