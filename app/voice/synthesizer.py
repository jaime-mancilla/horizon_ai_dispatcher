# app/voice/synthesizer.py

class Synthesizer:
    def __init__(self):
        # Placeholder: later we'll connect ElevenLabs API with voice config
        pass

    async def synthesize(self, text: str) -> bytes:
        """
        Synthesizes speech from given text.
        Placeholder for ElevenLabs API integration.
        """
        print(f"[Synthesizer] Synthesizing audio for: '{text}'")
        return b"FAKEAUDIOBYTES"
