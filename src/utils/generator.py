import io
import logging
import struct
import time
import requests
import numpy as np
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
        self.streaming = True

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
        stream=True,
        **kwargs
    ):
        """
        Generates speech from given text via Pocket TTS HTTP streaming with retry logic.

        When stream=True, returns a generator of float32 mono segments so audio can
        start playing before the whole sentence has been synthesized.
        """
        if not self.is_initialized():
            raise RuntimeError("Pocket TTS not initialized. Call initialize() first.")

        text = text.strip()
        if not text:
            return iter(()) if stream else ((None, []) if not return_chunks else ([], []))

        if max_retries is None:
            max_retries = getattr(settings, "TTS_MAX_RETRIES", 3)

        payload = {
            "input": text,
            "model": "pocket-tts",
            "response_format": "pcm",
        }
        if self.pocket_tts_voice:
            payload["voice"] = self.pocket_tts_voice

        endpoint = f"{self.pocket_tts_url}/v1/audio/speech"

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                res = requests.post(endpoint, json=payload, stream=True, timeout=10)
                res.raise_for_status()

                if stream:
                    return self._stream_audio(res)

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

    def _stream_audio(self, response):
        """Yield float32 mono segments as PCM samples arrive from the HTTP stream.

        Handles both a RIFF/WAVE response (what the local Pocket TTS shim returns for
        response_format=pcm) and a bare int16 PCM stream. Playback rate is 24 kHz.
        """
        buf = b""
        channels = 1
        bytes_per_sample = 2
        data_offset = None
        for raw in response.iter_content(chunk_size=16384):
            if not raw:
                continue
            buf += raw
            if data_offset is None:
                if buf.startswith(b"RIFF"):
                    info = _wav_header_info(buf)
                    if info is None:
                        continue
                    channels, bytes_per_sample, data_offset = info
                else:
                    data_offset = 0  # raw PCM stream, no header
                buf = buf[data_offset:]
            frame = channels * bytes_per_sample
            end = len(buf) // frame * frame
            if not end:
                continue
            samples = np.frombuffer(buf[:end], dtype="<i2")
            buf = buf[end:]
            audio = samples.astype(np.float32) / 32768.0
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
            yield audio


def _wav_header_info(header: bytes):
    """Parse a RIFF/WAVE header prefix once the data chunk is fully buffered.

    Returns (channels, bytes_per_sample, data_offset) or None if more header bytes
    are needed.
    """
    if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        return None
    pos = 12
    channels = 1
    bytes_per_sample = 2
    while pos + 8 <= len(header):
        cid = header[pos:pos + 4]
        size = struct.unpack("<I", header[pos + 4:pos + 8])[0]
        if cid == b"fmt ":
            if pos + 24 > len(header):
                return None
            audio_format, channels, _rate, _block, _bits, bits = struct.unpack(
                "<HHIIHH", header[pos + 8:pos + 24]
            )
            if audio_format != 1:
                return None
            bytes_per_sample = bits // 8
        elif cid == b"data":
            return channels, bytes_per_sample, pos + 8
        pos += 8 + size + (size % 2)
    return None



