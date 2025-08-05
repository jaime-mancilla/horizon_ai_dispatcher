# app/voice/transcriber.py

class Transcriber:
    def __init__(self):
        # Placeholder: later we'll initialize Whisper or a remote API
        pass

    async def transcribe(self, audio_chunk: bytes) -> str:
        """
        Transcribes audio chunk to text.
        Placeholder for Whisper / custom transcription service.
        """
        print("[Transcriber] Received audio chunk for transcription.")
        return "Transcribed text here"
