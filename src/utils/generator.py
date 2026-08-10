import io
import logging
import time
import requests
import soundfile as sf
from pathlib import Path
from .config import settings


class VoiceGenerator:
    """
    A class to manage voice generation using Pocket TTS remote streaming.
    """

    def __init__(self, models_dir=None, voices_dir=None):
        """
        Initializes the VoiceGenerator for Pocket TTS.
        """
        self._initialized = False
        self.pocket_tts_url = None
        self.pocket_tts_voice = "default"

    def initialize(self, pocket_tts_url="http://host.docker.internal:49112/", pocket_tts_voice="default", **kwargs):
        """
        Initializes Pocket TTS remote streaming client.
        """
        self.pocket_tts_url = pocket_tts_url.rstrip("/")
        self.pocket_tts_voice = pocket_tts_voice
        self._initialized = True
        return f"Loaded Pocket TTS remote streaming client: {self.pocket_tts_url}"

    def is_initialized(self):
        """
        Checks if generator is initialized.
        """
        return self._initialized and bool(self.pocket_tts_url)

    def generate(
        self,
        text,
        speed=1.0,
        return_chunks=False,
        max_retries=None,
        **kwargs
    ):
        """
        Generates speech from given text via Pocket TTS HTTP streaming with retry logic.
        """
        if not self.is_initialized():
            raise RuntimeError("Pocket TTS not initialized. Call initialize() first.")

        text = text.strip()
        if not text:
            return (None, []) if not return_chunks else ([], [])

        if max_retries is None:
            max_retries = getattr(settings, "TTS_MAX_RETRIES", 3)

        payload = {
            "input": text,
            "model": "pocket-tts",
        }
        if self.pocket_tts_voice:
            payload["voice"] = self.pocket_tts_voice

        endpoint = f"{self.pocket_tts_url}/v1/audio/speech"

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                res = requests.post(endpoint, json=payload, stream=True, timeout=10)
                res.raise_for_status()

                audio_bytes = bytearray()
                for chunk in res.iter_content(chunk_size=4096):
                    if chunk:
                        audio_bytes.extend(chunk)

                if not audio_bytes:
                    raise ValueError("Received empty audio response from Pocket TTS server")

                audio_data, _ = sf.read(io.BytesIO(audio_bytes), dtype='float32')
                return (audio_data, []) if not return_chunks else ([audio_data], [])
            except Exception as e:
                last_error = e
                logging.warning(f"Pocket TTS attempt {attempt}/{max_retries} failed: {str(e)}")
                if attempt < max_retries:
                    time.sleep(0.5)

        logging.error(f"Pocket TTS generation failed after {max_retries} retries: {str(last_error)}")
        raise ValueError(f"Error in Pocket TTS generation after {max_retries} attempts: {str(last_error)}")


