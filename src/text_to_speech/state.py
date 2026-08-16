"""Shared mutable state for the text-to-speech app.

Holds the event bus, event flags, timing info, idle coordination, and chat-file
helpers used by the pipeline thread and the TUI. Import this module by reference
(`from . import state; state.X = ...`) whenever a function must write to one of
these names, so both threads always see the same objects.
"""

import os
import queue
import threading
import time

from src.utils.config import settings

# config.py merges SPEAKER_DEVICE from settings.json over the env value at
# import; this variant must not pick up that app-level device selection, so
# reset to the .env/default value (empty = system default).
settings.SPEAKER_DEVICE = os.environ.get("SPEAKER_DEVICE", "") or ""
settings.setup_directories()

# ---- Event bus: pipeline thread(s) -> TUI main thread ----
_event_queue: queue.Queue = queue.Queue()
# TUI -> pipeline thread: typed text to send to the bot
text_input_queue: queue.Queue = queue.Queue()
shutdown_event = threading.Event()
# slash-command signals (TUI -> pipeline): /stop, /pause, /now
interrupt_event = threading.Event()
pause_event = threading.Event()
now_event = threading.Event()
# /memory on/off: Qdrant long-term memory on/off (RAM fallback when off); pipeline thread switches
memory_request_queue: queue.Queue = queue.Queue()
memory_status: dict[str, str | bool] = {"enabled": False, "backend": "RAM"}
# /new: reset the current chat session (clear LLM history); pipeline thread applies it
new_chat_event = threading.Event()

timing_info: dict[str, float | None] = {
    "vad_start": None,
    "transcription_start": None,
    "llm_first_token": None,
    "audio_queued": None,
    "first_audio_play": None,
    "playback_start": None,
    "end": None,
    "transcription_duration": None,
}


def emit(kind: str, *payload):
    """Post an event to the TUI's event queue (thread-safe)."""
    _event_queue.put((kind, *payload))


def _append_chat_file(name: str, text: str):
    """Append a line to one of the turn files (auto-created if missing)."""
    try:
        with open(settings.OUTPUT_DIR / name, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass


def _clear_chat_files():
    """Empty both turn files at the end of a spoken turn (files are kept, not deleted)."""
    try:
        for name in ("in.txt", "out.txt"):
            open(settings.OUTPUT_DIR / name, "w", encoding="utf-8").close()
    except Exception:
        pass


_last_idle_prompt = ""

# Idle-mode and typing-pause coordination, shared between the pipeline thread
# and the TUI. pipeline_last_activity is the authoritative idle countdown clock.
pipeline_last_activity: float = time.time()
last_typing_activity: float = 0.0
typing_pause_start: float | None = None
idle_mode: bool = False
_TYPING_PAUSE_SECONDS = 5.0


def _idle_elapsed() -> float:
    """Seconds since the last real activity, excluding any active typing pause."""
    now = time.time()
    if typing_pause_start is not None and now - last_typing_activity < _TYPING_PAUSE_SECONDS:
        return max(0.0, typing_pause_start - pipeline_last_activity)
    return now - pipeline_last_activity
