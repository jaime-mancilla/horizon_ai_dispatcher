# app/llm/agent.py

class LLMDispatcher:
    def __init__(self, transcriber, synthesizer):
        self.transcriber = transcriber
        self.synthesizer = synthesizer

    async def run(self):
        print("[LLMDispatcher] Dispatcher is running.")
        # Placeholder: async loop for handling interaction
        input_audio = b"fake audio input"
        text = await self.transcriber.transcribe(input_audio)
        print(f"[LLMDispatcher] Got text: {text}")
        output_audio = await self.synthesizer.synthesize("Hello, how can I help you?")
        print(f"[LLMDispatcher] Synthesized response of {len(output_audio)} bytes.")
