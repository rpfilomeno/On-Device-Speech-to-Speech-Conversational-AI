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

    POCKET_TTS_URL: str = Field(
        default="http://host.docker.internal:49112/", env="POCKET_TTS_URL"
    )
    POCKET_TTS_VOICE: str = Field(default="Jelai.wav", env="POCKET_TTS_VOICE")
    SPEED: float = Field(default=1.0, env="SPEED")
    HUGGINGFACE_TOKEN: Optional[str] = Field(default="", env="HUGGINGFACE_TOKEN")

    LM_STUDIO_URL: str = Field(..., env="LM_STUDIO_URL")
    DEFAULT_SYSTEM_PROMPT: str = Field(..., env="DEFAULT_SYSTEM_PROMPT")
    LLM_MODEL: str = Field(..., env="LLM_MODEL")
    NUM_THREADS: int = Field(default=2, env="NUM_THREADS")
    MAX_TOKENS: int = Field(default=512, env="MAX_TOKENS")
    LLM_TEMPERATURE: float = Field(default=0.7, env="LMM_TEMPERATURE")

    WHISPER_MODEL: str = Field(default="openai/whisper-tiny.en", env="WHISPER_MODEL")

    VAD_MODEL: str = Field(default="pyannote/segmentation-3.0", env="VAD_MODEL")
    VAD_MIN_DURATION_ON: float = Field(default=0.1, env="VAD_MIN_DURATION_ON")
    VAD_MIN_DURATION_OFF: float = Field(default=0.1, env="VAD_MIN_DURATION_OFF")

    CHUNK: int = Field(default=1024, env="CHUNK")
    FORMAT: str = Field(default="pyaudio.paFloat32", env="FORMAT")
    CHANNELS: int = Field(default=1, env="CHANNELS")
    RATE: int = Field(default=16000, env="RATE")
    RECORD_DURATION: int = Field(default=5, env="RECORD_DURATION")
    SILENCE_THRESHOLD: float = Field(default=0.01, env="SILENCE_THRESHOLD")
    INTERRUPTION_THRESHOLD: float = Field(default=0.005, env="INTERRUPTION_THRESHOLD")
    MAX_SILENCE_DURATION: int = Field(default=1, env="MAX_SILENCE_DURATION")
    SPEECH_CHECK_TIMEOUT: float = Field(default=0.1, env="SPEECH_CHECK_TIMEOUT")
    SPEECH_CHECK_THRESHOLD: float = Field(default=0.005, env="SPEECH_CHECK_THRESHOLD")
    ROLLING_BUFFER_TIME: float = Field(default=0.5, env="ROLLING_BUFFER_TIME")
    TARGET_SIZE: int = Field(default=15, env="TARGET_SIZE")
    FIRST_SENTENCE_SIZE: int = Field(default=3, env="FIRST_SENTENCE_SIZE")
    PLAYBACK_DELAY: float = Field(default=0.005, env="PLAYBACK_DELAY")

    ENABLE_AEC: bool = Field(default=True, env="ENABLE_AEC")
    AEC_FILTER_LENGTH: int = Field(default=512, env="AEC_FILTER_LENGTH")
    AEC_MU: float = Field(default=0.1, env="AEC_MU")

    ENABLE_NOISE_SUPPRESSION: bool = Field(default=True, env="ENABLE_NOISE_SUPPRESSION")
    ENABLE_AGC: bool = Field(default=True, env="ENABLE_AGC")
    AGC_TARGET_DB: float = Field(default=-16.0, env="AGC_TARGET_DB")

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
