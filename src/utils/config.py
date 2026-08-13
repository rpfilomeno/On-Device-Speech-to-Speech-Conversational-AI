from pathlib import Path
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from dotenv import load_dotenv
load_dotenv()

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Settings class to manage application configurations."""

    BASE_DIR: Path = Path(__file__).parent.parent.parent
    MODELS_DIR: Path = BASE_DIR / "data" / "models"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    RECORDINGS_DIR: Path = BASE_DIR / "recordings"

    POCKET_TTS_URL: str = Field(default="http://host.docker.internal:49112/")
    POCKET_TTS_VOICE: str = Field(default="Jelai.wav")
    SPEED: float = Field(default=1.0)
    TTS_MAX_RETRIES: int = Field(default=3)
    HUGGINGFACE_TOKEN: Optional[str] = Field(default="")

    LM_STUDIO_URL: str = Field(...)
    DEFAULT_SYSTEM_PROMPT: str = Field(...)
    LLM_MODEL: str = Field(...)
    NUM_THREADS: int = Field(default=2)
    MAX_TOKENS: int = Field(default=512)
    LLM_TEMPERATURE: float = Field(default=0.7)

    QDRANT_HOST: Optional[str] = Field(default="")
    QDRANT_COLLECTION: str = Field(default="conversation_memory")
    EMBEDDING_MODEL: str = Field(default="text-embedding-nomic-embed-text-v1.5")

    WHISPER_MODEL_ID: str = Field(default="openai/whisper-tiny.en")
    WHISPER_MODEL_DIR: str = Field(default="data/models/whisper-tiny.en")

    VAD_MODEL_ID: str = Field(default="pyannote/segmentation-3.0")
    VAD_MODEL_DIR: str = Field(default="data/models/segmentation-3.0")
    VAD_MIN_DURATION_ON: float = Field(default=0.1)
    VAD_MIN_DURATION_OFF: float = Field(default=0.1)

    CHUNK: int = Field(default=1024)
    FORMAT: str = Field(default="pyaudio.paFloat32")
    CHANNELS: int = Field(default=1)
    RATE: int = Field(default=16000)
    RECORD_DURATION: int = Field(default=5)
    SILENCE_THRESHOLD: float = Field(default=0.01)
    INTERRUPTION_THRESHOLD: float = Field(default=0.005)
    MAX_SILENCE_DURATION: int = Field(default=1)
    SPEECH_CHECK_TIMEOUT: float = Field(default=0.1)
    SPEECH_CHECK_THRESHOLD: float = Field(default=0.005)
    ROLLING_BUFFER_TIME: float = Field(default=0.5)
    TARGET_SIZE: int = Field(default=15)
    FIRST_SENTENCE_SIZE: int = Field(default=3)
    PLAYBACK_DELAY: float = Field(default=0.005)
    MIC_DEVICE: str = Field(default="")
    SPEAKER_DEVICE: str = Field(default="")
    LOG_TTS_CHUNKS: bool = Field(default=True)
    LOG_TWITCH_CHATS: bool = Field(default=True)

    TWITCH_CLIENT_ID: Optional[str] = Field(default="")
    TWITCH_CLIENT_SECRET: Optional[str] = Field(default="")
    TWITCH_CLIENT_CHANNEL: Optional[str] = Field(default="")
    TWITCH_MAX_CHAT_SIZE: int = Field(default=50)
    TWITCH_MAX_CHAT_AGE: float = Field(default=180.0)

    MAX_IDLE_TIME: float = Field(default=60.0)
    TWITCH_CHAT_PROMPT: str = Field(
        default="You noticed there are messages and events from twitch chat, try to respond or react to them: {TWITCH_CHATS_AND_EVENTS}"
    )
    IDLE_PROMPTS: str = Field(
        default="Make a joke, Tell what you dreamt last night, Tell us an unforgettable memory, What is the fact-of-the-day, Tell us thinf you plan to do someday"
    )

    def get_idle_prompts_list(self) -> list[str]:
        """Parses IDLE_PROMPTS string into a list of individual prompt strings."""
        if not self.IDLE_PROMPTS:
            return ["Tell us something interesting!"]
        return [p.strip() for p in self.IDLE_PROMPTS.split(",") if p.strip()]

    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()


def configure_logging():
    """Configure logging to suppress all logs"""
    import logging
    import warnings

    warnings.filterwarnings("ignore")

    logging.getLogger().setLevel(logging.ERROR)

    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("whisper").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("pyannote").setLevel(logging.ERROR)
    logging.getLogger("sounddevice").setLevel(logging.ERROR)
    logging.getLogger("soundfile").setLevel(logging.ERROR)
    logging.getLogger("uvicorn").setLevel(logging.ERROR)
    logging.getLogger("fastapi").setLevel(logging.ERROR)


configure_logging()
