import gc
import logging
import random
import queue
import re
import sys
import threading
import time
import traceback
from typing import cast
import numpy as np
import requests
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from src.utils.config import settings, save_settings, save_device_settings, log_error
from src.utils import (
    VoiceGenerator,
    get_ai_response,
    init_vad_pipeline,
    init_whisper_model,
    detect_speech_segments,
    record_continuous_audio,
    transcribe_audio,
    list_audio_devices,
    twitch_collector,
    twitch_bot_manager,
)
from src.utils.speech import TurnAudioPlayer, classify_barge
from src.utils.audio_queue import AudioGenerationQueue
from src.utils.llm import parse_stream_chunk
from src.utils.memory import Memory, MemoryWorker, RamMemory
from src.utils.text_chunker import TextChunker

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, OptionList, ProgressBar, RichLog, Static

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
# /voice on/off: VAD + voice transcription listening (off by default)
voice_event = threading.Event()
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


class _LogStream:
    """Captures sys.stdout/sys.stderr so every print() (from pipeline, TTS,
    Twitch bot) and traceback lands in the TUI's log panel instead of being
    printed on top of the screen."""

    def __init__(self):
        self._buf = ""

    def write(self, text: str):
        # never raise / never touch the real terminal: any bad write is dropped.
        # Only complete (newline-terminated) lines are emitted -- partial lines
        # stay buffered so they can never be emitted twice.
        try:
            # keep only the tail after the last \r so tqdm progress bars collapse
            # into their final line instead of spamming the log panel
            if "\r" in text:
                text = text.rsplit("\r", 1)[-1]
            self._buf += text
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.strip()
                if line:
                    emit("log", line)
        except Exception:
            pass

    def flush(self):
        try:
            if self._buf.strip():
                emit("log", self._buf.strip())
            self._buf = ""
        except Exception:
            pass


_log_stream = _LogStream()


class _LogHandler(logging.Handler):
    """Routes every logging record (PocketTTS retries, audio stats, ...) to the
    TUI log panel instead of the raw stderr stream."""

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            if msg.strip():
                emit("log", msg)
        except Exception:
            pass


def _excepthook(exc_type, exc_value, exc_tb):
    try:
        emit("error", f"Unhandled {exc_type.__name__}: {exc_value}")
        for line in traceback.format_exception(exc_type, exc_value, exc_tb):
            emit("log", line.rstrip())
    except Exception:
        pass


def _thread_excepthook(args):
    try:
        emit("error", f"Unhandled {args.exc_type.__name__} in thread '{args.thread.name}': {args.exc_value}")
        for line in traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback):
            emit("log", line.rstrip())
    except Exception:
        pass


_orig_excepthook = sys.excepthook
_orig_thread_excepthook = threading.excepthook


def _install_error_routing():
    """Redirect all error output (prints, logging, unhandled exceptions) into
    the TUI event queue so nothing can be written over the screen."""
    sys.stdout = _log_stream
    sys.stderr = _log_stream
    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook
    # Replace any handler created at import time (e.g. audio_queue's
    # logging.basicConfig bound to the original stderr) with one that feeds
    # the log panel. Also prevents later basicConfig() calls from adding more.
    logger = logging.getLogger()
    handler = _LogHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers[:] = [handler]


def _restore_error_routing():
    if isinstance(sys.stdout, _LogStream):
        sys.stdout = sys.__stdout__
    if isinstance(sys.stderr, _LogStream):
        sys.stderr = sys.__stderr__
    sys.excepthook = _orig_excepthook
    threading.excepthook = _orig_thread_excepthook


_TRIM_GRACE_RETRIES = 2  # transient failures don't shrink context; trim only after this many consecutive retries
_MAX_LLM_RETRIES = 5  # after this many blank responses, start a new session


def _trim_history(messages: list) -> bool:
    """Drop a random contiguous block of the completed turns (a random 20-50%
    of the middle, at a random position), keeping the system prompt and the
    current user message, so each retry sends a smaller context. Returns True
    if anything was dropped."""
    middle_len = len(messages) - 2
    if middle_len < 2:
        return False
    n_drop = max(1, int(middle_len * random.uniform(0.2, 0.5)))
    start = random.randint(0, middle_len - n_drop)
    del messages[1 + start:1 + start + n_drop]
    return True


def _retry_llm(messages: list, retry: int, reason: str) -> bool:
    """Retry a failed LLM call. The first few failures (server busy, model
    loading) retry as-is and let the history grow; only later failures start
    trimming older turns so the context can actually shrink. Returns True if a
    retry is worth another attempt."""
    if retry >= _TRIM_GRACE_RETRIES:
        if not _trim_history(messages):
            return False
        emit("log", f"{reason} - trimmed history and retrying.")
    else:
        emit("log", f"{reason} - retrying without trimming history.")
    return True


def process_input(
    session: requests.Session,
    user_input: str,
    messages: list,
    generator: VoiceGenerator,
    speed: float,
    memory: MemoryWorker | None = None,
    whisper_processor=None,
    whisper_model=None,
    vad_pipeline=None,
) -> tuple[bool, np.ndarray | None]:
    """Processes user input, generates a response, and handles audio output.

    Args:
        session (requests.Session): The requests session to use.
        user_input (str): The user's input text.
        messages (list): The list of messages to send to the LLM.
        generator (VoiceGenerator): The voice generator object.
        speed (float): The playback speed.
        memory (Memory, optional): Long-term vector memory to recall from / write to.
        whisper_processor: Whisper processor used to gate barge candidates.
        whisper_model: Whisper model used to gate barge candidates.
        vad_pipeline: VAD used to strip non-speech from barge candidates.

    Returns:
        tuple[bool, None]: A tuple containing a boolean indicating if the process was interrupted and None.
    """
    global timing_info
    timing_info = {k: None for k in timing_info}
    timing_info["vad_start"] = time.perf_counter()

    messages.append({"role": "user", "content": user_input})
    emit("status", "THINKING")

    memory_block = None
    if memory is not None:
        try:
            recalled = memory.search(user_input)
            if recalled:
                memory_block = "\n".join(f"- {m}" for m in recalled)
        except Exception as e:
            log_error(e)
            emit("log", f"Memory recall failed: {e}")

    def llm_messages_for(msgs: list) -> list:
        if memory_block is None:
            return msgs
        llm = list(msgs)
        llm[0] = {
            "role": "system",
            "content": msgs[0]["content"]
            + "\n\nRelevant memories from past conversations:\n"
            + memory_block,
        }
        return llm

    llm_messages = llm_messages_for(messages)

    emit("turn_start", user_input)
    interrupt_event.clear()
    start_time = time.time()
    audio_queue: AudioGenerationQueue | None = None
    interrupted = False
    interrupt_data: np.ndarray | None = None
    try:
        audio_queue = AudioGenerationQueue(generator, speed)
        audio_queue.start()

        def worker_runner():
            nonlocal interrupted, interrupt_data
            was_int, int_data = audio_playback_worker(
                audio_queue, whisper_processor, whisper_model, vad_pipeline
            )
            interrupted = was_int
            interrupt_data = int_data

        playback_thread = threading.Thread(target=worker_runner)
        playback_thread.daemon = True
        playback_thread.start()

        retry = 0
        session_reset = False
        bot_text = ""
        while True:
            response_stream = get_ai_response(
                session=session,
                messages=llm_messages,
                llm_model=settings.LLM_MODEL,
                llm_url=settings.LM_STUDIO_URL,
                max_tokens=settings.MAX_TOKENS,
                stream=True,
            )

            if not response_stream:
                if not _retry_llm(messages, retry, "LLM request failed"):
                    emit("log", "Failed to get AI response stream.")
                    break
                retry += 1
                llm_messages = llm_messages_for(messages)
                continue

            chunker = TextChunker()
            complete_response = []
            for chunk in response_stream:
                if interrupt_event.is_set():
                    emit("log", "[Command] Stop: stream aborted.")
                    break
                data = parse_stream_chunk(chunk)
                if not data or "choices" not in data:
                    continue

                choice = data["choices"][0]
                if "delta" in choice and "content" in choice["delta"]:
                    content = choice["delta"]["content"]
                    if content:
                        if not timing_info["llm_first_token"]:
                            timing_info["llm_first_token"] = time.perf_counter()
                        emit("bot_token", content)
                        chunker.current_text.append(content)

                        text = "".join(chunker.current_text)
                        settings.TARGET_SIZE = _adaptive_target_words()
                        if chunker.should_process(text):
                            if not timing_info["audio_queued"]:
                                timing_info["audio_queued"] = time.perf_counter()
                            remaining = chunker.process(text, audio_queue)
                            chunker.current_text = [remaining] if remaining else []
                            processed_len = len(text) - len(remaining)
                            if processed_len > 0:
                                complete_response.append(text[:processed_len])

                if choice.get("finish_reason") == "stop":
                    break

            settings.TARGET_SIZE = _adaptive_target_words()
            final_flushed = chunker.flush(audio_queue)
            if final_flushed:
                complete_response.append(final_flushed)

            bot_text = " ".join(" ".join(complete_response).split())
            if bot_text:
                break

            if retry >= _MAX_LLM_RETRIES and not session_reset:
                messages[:] = [
                    {"role": "system", "content": settings.DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                ]
                session_reset = True
                retry = 0
                emit("log", "[Chat] LLM blank after history trim — starting a new session.")
                llm_messages = llm_messages_for(messages)
                continue

            if not _retry_llm(messages, retry, "LLM returned empty (context too long?)"):
                emit("log", "Failed to get a response after trimming history.")
                break
            retry += 1
            llm_messages = llm_messages_for(messages)

        if bot_text:
            messages.append({"role": "assistant", "content": bot_text})
        print()

        audio_queue.stop()
        playback_thread.join()
        _clear_chat_files()

        timing_info["end"] = time.perf_counter()
        print_timing_chart(timing_info)
        if bot_text:
            emit("turn", user_input, bot_text)
        if memory is not None and bot_text:
            try:
                memory.store("user", user_input)
                memory.store("assistant", bot_text)
            except Exception as e:
                log_error(e)
                emit("log", f"Memory store failed: {e}")
        emit("status", "LISTENING")
        return interrupted, interrupt_data

    except Exception as e:
        log_error(e)
        emit("log", f"Error during streaming: {str(e)}")
        if audio_queue is not None:
            audio_queue.stop()
        return False, None


# Adaptive chunking: when TTS synthesis falls behind the player, shrink the
# words-per-chunk target so sentences synthesize faster (fewer words = shorter
# audio = less time behind), and poll fast so the next chunk is grabbed the
# instant it's ready. The chunk size eases the same way the old playback-speed
# recovery did: smooth descent, cubic ease-out back to full size.
_TARGET_MIN = 6
_TARGET_RAMP_START = 0.3
_TARGET_RAMP_END = 2.5
_BEHIND_EMA_ALPHA = 0.3
_BEHIND_DECAY = 0.8
_POLL_LAZY_DELAY = 0.05
_POLL_BEHIND_THRESHOLD = 0.5

# Persistent across turns (until app exit): a turn that starts slow already
# carries the previous adaptation instead of re-learning the first gap.
_behind_ema = 0.0
_EASE_ROUNDS = 5
_smoothed_target: float | None = None
_ease_rounds_left = 0
_ease_start_value = 0.0


def _target_for(behind_ema: float) -> float:
    """Raw words-per-chunk target for a given behind-average."""
    full = float(settings.TARGET_SIZE)
    if behind_ema <= _TARGET_RAMP_START:
        return full
    if behind_ema >= _TARGET_RAMP_END:
        return float(_TARGET_MIN)
    t = (behind_ema - _TARGET_RAMP_START) / (_TARGET_RAMP_END - _TARGET_RAMP_START)
    return full - t * (full - _TARGET_MIN)


def _ease_to(current: float, target: float) -> float:
    """Ease `current` toward `target`: smooth descent, 5-round cubic ease-out
    on recovery (the scheme the playback-speed recovery used)."""
    global _ease_rounds_left, _ease_start_value
    if abs(target - current) < 0.05:
        _ease_rounds_left = 0
        return target
    if target < current:
        _ease_rounds_left = 0
        return current + 0.5 * (target - current)
    if _ease_rounds_left == 0:
        _ease_start_value = current
        _ease_rounds_left = _EASE_ROUNDS
    _ease_rounds_left = max(0, _ease_rounds_left - 1)
    progress = 1.0 - _ease_rounds_left / _EASE_ROUNDS
    eased = 1.0 - (1.0 - progress) ** 3
    return _ease_start_value + (target - _ease_start_value) * eased


def _adaptive_target_words() -> int:
    """Current eased words-per-chunk target, driven by the behind-average."""
    global _smoothed_target
    target = _target_for(_behind_ema)
    if _smoothed_target is None:
        _smoothed_target = target
    else:
        _smoothed_target = _ease_to(_smoothed_target, target)
    return int(round(_smoothed_target))


def _adaptive_poll_delay(behind_ema: float) -> float:
    """Poll interval: minimal when behind (grab audio ASAP), lazy when ahead."""
    if behind_ema > _POLL_BEHIND_THRESHOLD:
        return settings.PLAYBACK_DELAY
    return _POLL_LAZY_DELAY


# Runtime jitter learner: bucket per-turn jitter (inter-chunk gap stddev) by the
# words-per-chunk target in effect, then persist the best-scoring target size to
# settings.json so future sessions start already-tuned.
_jitter_samples: dict[int, list[float]] = {}
_MIN_JITTER_SAMPLES = 3


def _best_target_size() -> int | None:
    """Target size with the lowest mean jitter, among buckets with enough samples."""
    best = None
    for size, samples in _jitter_samples.items():
        if len(samples) < _MIN_JITTER_SAMPLES:
            continue
        mean = sum(samples) / len(samples)
        if best is None or mean < best[1]:
            best = (size, mean)
    return best[0] if best else None


def _record_jitter(target_size: int, jitter_ms: float) -> int | None:
    """Feed one turn's jitter into the learner; returns the current best target size."""
    samples = _jitter_samples.setdefault(target_size, [])
    samples.append(jitter_ms)
    del samples[:-30]
    return _best_target_size()


# PLAYBACK_DELAY tuning: round-robin one candidate per turn until every candidate
# has enough jitter samples, then settle on the lowest-mean one and persist it.
_DELAY_CANDIDATES = [0.005, 0.01, 0.02, 0.05]
_delay_samples: dict[float, list[float]] = {}
_delay_explore_index = 0
_delay_settled = False


def _record_delay_jitter(jitter_ms: float) -> float | None:
    """Record jitter under the current PLAYBACK_DELAY and advance to the next
    candidate. Once every candidate has enough samples, returns the best
    (lowest mean jitter) delay to adopt, else None."""
    global _delay_explore_index, _delay_settled
    if _delay_settled:
        return None
    if settings.PLAYBACK_DELAY not in _DELAY_CANDIDATES:
        settings.PLAYBACK_DELAY = _DELAY_CANDIDATES[0]
    samples = _delay_samples.setdefault(settings.PLAYBACK_DELAY, [])
    samples.append(jitter_ms)
    del samples[:-30]
    if len(_delay_samples) == len(_DELAY_CANDIDATES) and all(
        len(v) >= _MIN_JITTER_SAMPLES for v in _delay_samples.values()
    ):
        _delay_settled = True
        return min(
            _delay_samples, key=lambda k: sum(_delay_samples[k]) / len(_delay_samples[k])
        )
    _delay_explore_index += 1
    settings.PLAYBACK_DELAY = _DELAY_CANDIDATES[
        _delay_explore_index % len(_DELAY_CANDIDATES)
    ]
    return None


def audio_playback_worker(
    audio_queue, whisper_processor=None, whisper_model=None, vad_pipeline=None
) -> tuple[bool, np.ndarray | None]:
    """Manages audio playback in a separate thread, handling interruptions.

    One TurnAudioPlayer stays open for the whole turn: gapless chunk playback,
    continuous barge-in monitoring, and a single stop/clear path.

    Args:
        audio_queue (AudioGenerationQueue): The audio queue object.
        whisper_processor: Whisper processor used to gate barge candidates.
        whisper_model: Whisper model used to gate barge candidates.
        vad_pipeline: VAD used to strip non-speech from barge candidates.

    Returns:
        tuple[bool, None]: A tuple containing a boolean indicating if the playback was interrupted and the interrupt audio data.
    """
    global timing_info, _behind_ema
    was_interrupted = False
    interrupt_audio = None

    played_any = False
    wait_start = None
    gaps: list[float] = []

    try:
        player = TurnAudioPlayer(
            stop_events=[interrupt_event, pause_event],
            monitor_input=voice_event.is_set(),
        )
    except Exception as e:
        log_error(e)
        emit("log", f"Error opening audio player: {str(e)}")
        return False, None

    try:
        while True:
            if interrupt_event.is_set():
                interrupt_event.clear()
                emit("log", "[Command] Stop: playback interrupted, clearing queues.")
                player.stop()
                audio_queue.clear_queues()
                break
            if pause_event.is_set():
                audio_queue.clear_queues()
                player.stop()
                time.sleep(settings.PLAYBACK_DELAY)
                continue
            if player.is_interrupted():
                if pause_event.is_set():
                    emit("log", "[Command] Pause: playback suspended.")
                    player.stop()
                    audio_queue.clear_queues()
                    break
                if interrupt_event.is_set():
                    interrupt_event.clear()
                    emit("log", "[Command] Stop: playback interrupted, clearing queues.")
                    player.stop()
                    audio_queue.clear_queues()
                    break
                # Barge candidate: hold playback, capture the full phrase, then decide.
                player.pause()
                emit("log", "[TTS] Voice detected — checking for barge intent...")
                player.wait_for_quiet()
                capture = player.take_capture()
                if capture is not None and whisper_processor is not None:
                    text = ""
                    try:
                        if vad_pipeline is not None:
                            seg = detect_speech_segments(vad_pipeline, capture)
                            if seg is not None:
                                text = transcribe_audio(
                                    whisper_processor, whisper_model, seg
                                )
                        else:
                            text = transcribe_audio(
                                whisper_processor, whisper_model, capture
                            )
                    except Exception as e:
                        log_error(e)
                        text = ""
                    reason = classify_barge(text)
                    if reason:
                        was_interrupted = True
                        if reason == "turn":
                            interrupt_audio = capture
                            emit("log", f"[TTS Interrupted] '{text}' — halting, using as next input.")
                        else:
                            emit("log", f"[TTS Interrupted] '{text}' — halting playback.")
                        player.stop()
                        audio_queue.clear_queues()
                        break
                    emit("log", f"[TTS] '{text}' — not a barge command, resuming.")
                player.resume()
                continue

            audio_data, sentence, is_first = audio_queue.get_next_audio()
            if audio_data is not None:
                if wait_start is not None:
                    gap = time.time() - wait_start
                    gaps.append(gap)
                    _behind_ema += _BEHIND_EMA_ALPHA * (gap - _behind_ema)
                    wait_start = None
                else:
                    _behind_ema *= _BEHIND_DECAY
                played_any = True
                if not timing_info["first_audio_play"]:
                    timing_info["first_audio_play"] = time.perf_counter()
                    emit("status", "SPEAKING")

                if is_first:
                    emit("bot_spoken", sentence)
                    _append_chat_file("out.txt", sentence)
                if settings.LOG_TTS_CHUNKS:
                    emit("log", f"[TTS Playing] {sentence!r}")
                player.push(audio_data)
            else:
                if played_any and wait_start is None and not player.is_playing():
                    wait_start = time.time()
                time.sleep(_adaptive_poll_delay(_behind_ema))

            if (
                not audio_queue.is_running
                and audio_queue.sentence_queue.empty()
                and audio_queue.audio_queue.empty()
            ):
                player.flush()
                break

    except Exception as e:
        log_error(e)
        emit("log", f"Error in audio playback: {str(e)}")
    finally:
        player.stop()

    save_settings(
        TARGET_SIZE=round(settings.TARGET_SIZE),
        PLAYBACK_DELAY=settings.PLAYBACK_DELAY,
    )

    if gaps:
        mean = sum(gaps) / len(gaps)
        stddev = (sum((g - mean) ** 2 for g in gaps) / len(gaps)) ** 0.5
        emit(
            "log",
            f"[Playback] jitter (inter-chunk gap): stddev {stddev * 1000:.0f}ms, "
            f"mean {mean * 1000:.0f}ms, max {max(gaps) * 1000:.0f}ms over {len(gaps)} gap(s).",
        )
        if len(gaps) >= 2:
            used_target = round(settings.TARGET_SIZE)
            used_delay = settings.PLAYBACK_DELAY
            best_target = _record_jitter(used_target, stddev * 1000)
            best_delay = _record_delay_jitter(stddev * 1000)
            if best_target is not None and best_target != used_target:
                settings.TARGET_SIZE = best_target
                save_settings(TARGET_SIZE=best_target)
                emit(
                    "log",
                    f"[Jitter Stats] Best TARGET_SIZE is now {best_target} "
                    f"(lowest mean jitter); saved to settings.json.",
                )
            if best_delay is not None:
                settings.PLAYBACK_DELAY = best_delay
                save_settings(PLAYBACK_DELAY=best_delay)
                emit(
                    "log",
                    f"[Jitter Stats] Best PLAYBACK_DELAY is now {best_delay}s "
                    f"(lowest mean jitter); saved to settings.json.",
                )

    return was_interrupted, interrupt_audio


def init_memory(mode: str) -> tuple[MemoryWorker | None, str]:
    """Build a memory backend. mode='off' -> None (no embedding calls); mode='on' -> Qdrant (RAM fallback). Returns (worker, label)."""
    if mode != "on":
        return None, "off"
    try:
        if not settings.QDRANT_HOST:
            raise RuntimeError("QDRANT_HOST not set")
        backend = Memory(
            settings.QDRANT_HOST,
            settings.LM_STUDIO_URL,
            settings.EMBEDDING_MODEL,
            collection=settings.QDRANT_COLLECTION,
        )
        backend.check()
        label = f"Qdrant ({settings.QDRANT_HOST})"
    except Exception as e:
        log_error(e)
        backend = RamMemory(settings.LM_STUDIO_URL, settings.EMBEDDING_MODEL)
        label = f"RAM (Qdrant unavailable: {e})"
    return MemoryWorker(backend), label


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


def pipeline_main():
    """Runs the voice-loop pipeline in a background thread; reports to the TUI."""
    global _last_idle_prompt, pipeline_last_activity, typing_pause_start, idle_mode
    try:
        emit("status", "INITIALIZING")
        whisper_processor = whisper_model = None
        vad_pipeline = None
        emit("log", "Voice input is OFF — VAD/Whisper models load on /voice on.")
        emit("log", "Initializing voice generator (Pocket TTS remote streaming)...")
        generator = VoiceGenerator()
        result = generator.initialize(
            pocket_tts_url=settings.POCKET_TTS_URL,
            pocket_tts_voice=settings.POCKET_TTS_VOICE,
        )
        emit("log", result)
        speed = settings.SPEED

        session = requests.Session()
        messages = [{"role": "system", "content": settings.DEFAULT_SYSTEM_PROMPT}]

        memory, label = init_memory("off")
        memory_status.update(enabled=False, backend=label)
        emit("log", f"Long-term memory: {label} (use /memory on for Qdrant).")

        if settings.TWITCH_CLIENT_CHANNEL:
            emit("log", f"Starting Twitch chat collector for channel: {settings.TWITCH_CLIENT_CHANNEL}...")
            twitch_bot_manager.start(settings.TWITCH_CLIENT_CHANNEL)

        try:
            emit("log", "Warming up the LLM...")
            response_stream = get_ai_response(
                session=session,
                messages=[
                    {"role": "system", "content": settings.DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": "Hi!"},
                ],
                llm_model=settings.LLM_MODEL,
                llm_url=settings.LM_STUDIO_URL,
                max_tokens=settings.MAX_TOKENS,
                stream=False,
            )
            if not response_stream:
                emit("log", "Failed to initialize the AI model!")
        except requests.RequestException as e:
            log_error(e)
            emit("log", f"Warmup failed: {str(e)}")

        emit("status", "LISTENING")
        emit("log", "=== Ready. Voice input is OFF (type /voice on to enable). Typing below triggers a response. ===")
        pipeline_last_activity = time.time()
        was_paused = False

        while not shutdown_event.is_set():
            if pause_event.is_set():
                if not was_paused:
                    emit("status", "PAUSED")
                    emit("log", "[Command] Paused: voice and idle countdown suspended. /play to resume.")
                was_paused = True
                time.sleep(0.1)
                continue
            if was_paused:
                was_paused = False
                pipeline_last_activity = time.time()
                idle_mode = False
                typing_pause_start = None
                emit("status", "LISTENING")
                emit("log", "[Command] Resumed.")

            try:
                text = text_input_queue.get_nowait()
            except queue.Empty:
                text = None

            if text is not None:
                idle_mode = False
                typing_pause_start = None
                pipeline_last_activity = time.time()
                process_input(session, text, messages, generator, speed, memory=memory,
                              whisper_processor=whisper_processor, whisper_model=whisper_model,
                              vad_pipeline=vad_pipeline)
                pipeline_last_activity = time.time()
                continue

            try:
                mem_cmd = memory_request_queue.get_nowait()
            except queue.Empty:
                mem_cmd = None
            if mem_cmd == "on":
                memory, label = init_memory("on")
                memory_status.update(enabled=True, backend=label)
                emit("log", f"[Memory] Long-term memory: {label}.")
            elif mem_cmd == "off":
                memory, label = init_memory("off")
                memory_status.update(enabled=False, backend=label)
                emit("log", f"[Memory] Long-term memory: {label}.")

            if new_chat_event.is_set():
                new_chat_event.clear()
                messages = [{"role": "system", "content": settings.DEFAULT_SYSTEM_PROMPT}]
                emit("log", "[Chat] New session started — LLM history cleared.")

            if not voice_event.is_set():
                audio_data = None
                time.sleep(0.1)
                if vad_pipeline is not None:
                    emit("log", "[Voice] VAD/Whisper models unloaded (use /voice on to reload).")
                    whisper_processor = whisper_model = None
                    vad_pipeline = None
                    gc.collect()
            else:
                if vad_pipeline is None:
                    emit("log", "Initializing Whisper model...")
                    whisper_processor, whisper_model = init_whisper_model(
                        settings.WHISPER_MODEL_ID, settings.WHISPER_MODEL_DIR, hf_token=settings.HUGGINGFACE_TOKEN
                    )
                    emit("log", "Initializing Voice Activity Detection...")
                    vad_pipeline = init_vad_pipeline(settings.HUGGINGFACE_TOKEN)
                audio_data = record_continuous_audio(max_wait=1.0)
            if audio_data is not None:
                speech_segments = detect_speech_segments(vad_pipeline, audio_data)

                if speech_segments is not None:
                    pipeline_last_activity = time.time()
                    typing_pause_start = None
                    idle_mode = False
                    emit("activity")
                    emit("status", "TRANSCRIBING")
                    timing_info["transcription_start"] = time.perf_counter()

                    user_input = transcribe_audio(
                        whisper_processor, whisper_model, speech_segments
                    )

                    timing_info["transcription_duration"] = (
                        time.perf_counter() - timing_info["transcription_start"]
                    )
                    if user_input.strip():
                        emit("transcript", "voice", user_input)
                        was_interrupted, speech_data = process_input(
                            session, user_input, messages, generator, speed, memory=memory,
                            whisper_processor=whisper_processor, whisper_model=whisper_model,
                            vad_pipeline=vad_pipeline,
                        )
                        pipeline_last_activity = time.time()
                        emit("activity")

                        if was_interrupted and speech_data is not None:
                            speech_segments = detect_speech_segments(
                                vad_pipeline, speech_data
                            )
                            if speech_segments is not None:
                                emit("log", "Transcribing interrupted speech...")
                                emit("status", "TRANSCRIBING")
                                user_input = transcribe_audio(
                                    whisper_processor,
                                    whisper_model,
                                    speech_segments,
                                )
                                if user_input.strip():
                                    emit("transcript", "voice", user_input)
                                    process_input(
                                        session,
                                        user_input,
                                        messages,
                                        generator,
                                        speed,
                                        memory=memory,
                                        whisper_processor=whisper_processor,
                                        whisper_model=whisper_model,
                                        vad_pipeline=vad_pipeline,
                                    )
                                    pipeline_last_activity = time.time()
                                    emit("activity")
                    else:
                        emit("log", "No clear speech detected, please try again.")
                else:
                    emit("log", "No clear speech detected, please try again.")
            else:
                # Check idle condition if no voice input was detected
                if interrupt_event.is_set():
                    interrupt_event.clear()
                    emit("log", "[Command] Stop: nothing in progress.")

                now = time.time()
                # Close a finished typing pause (fold the paused span out of the countdown)
                if typing_pause_start is not None and now - last_typing_activity >= _TYPING_PAUSE_SECONDS:
                    pipeline_last_activity += now - typing_pause_start
                    typing_pause_start = None
                # Typing activity: exit idle mode and suspend the countdown for 5s
                typing_active = now - last_typing_activity < _TYPING_PAUSE_SECONDS
                if typing_active:
                    if idle_mode:
                        idle_mode = False
                        emit("status", "LISTENING")
                        emit("log", "[Idle] Typing detected — exiting idle mode.")
                    if typing_pause_start is None:
                        typing_pause_start = now

                idle_elapsed = _idle_elapsed()
                if (
                    idle_mode
                    or (not typing_active and idle_elapsed >= settings.MAX_IDLE_TIME)
                    or now_event.is_set()
                ):
                    if not idle_mode:
                        idle_mode = True
                        now_event.clear()
                        emit("status", "IDLE")
                        emit("log", f"[Idle Trigger] No activity for {idle_elapsed:.1f}s (MAX_IDLE_TIME={settings.MAX_IDLE_TIME}s).")
                    else:
                        now_event.clear()

                    # Check if recent Twitch events/messages are available
                    twitch_events = twitch_collector.get_recent_events(
                        max_size=settings.TWITCH_MAX_CHAT_SIZE,
                        max_age=settings.TWITCH_MAX_CHAT_AGE,
                    )

                    if twitch_events:
                        events_summary = "\n".join(twitch_events)
                        prompt_text = settings.TWITCH_CHAT_PROMPT.format(
                            TWITCH_CHATS_AND_EVENTS=events_summary
                        )
                        emit("log", f"[Twitch Idle Event] Responding to {len(twitch_events)} collected Twitch event(s)...")
                        emit("transcript", "idle", prompt_text)
                    else:
                        idle_prompts = settings.get_idle_prompts_list()
                        choices = [p for p in idle_prompts if p != _last_idle_prompt] or idle_prompts
                        prompt_text = random.choice(choices)
                        _last_idle_prompt = prompt_text
                        emit("log", f"[Random Idle Event] Picked prompt: '{prompt_text}'")
                        emit("transcript", "idle", prompt_text)

                    _, speech_data = process_input(session, prompt_text, messages, generator, speed, memory=memory,
                                                   whisper_processor=whisper_processor, whisper_model=whisper_model,
                                                   vad_pipeline=vad_pipeline)
                    if speech_data is not None:
                        idle_mode = False
                        emit("status", "LISTENING")
                        emit("log", "[Idle] Voice interrupt — exiting idle mode.")
                        speech_segments = detect_speech_segments(vad_pipeline, speech_data)
                        if speech_segments is not None:
                            emit("log", "Transcribing interrupted speech...")
                            emit("status", "TRANSCRIBING")
                            user_input = transcribe_audio(
                                whisper_processor, whisper_model, speech_segments
                            )
                            if user_input.strip():
                                emit("transcript", "voice", user_input)
                                process_input(
                                    session, user_input, messages, generator, speed, memory=memory,
                                    whisper_processor=whisper_processor, whisper_model=whisper_model,
                                    vad_pipeline=vad_pipeline,
                                )
                                pipeline_last_activity = time.time()
                                typing_pause_start = None
                                emit("activity")
                    else:
                        pipeline_last_activity = time.time()
                        typing_pause_start = None
                        emit("activity")
                    if idle_mode:
                        emit("status", "IDLE")

                if session is not None:
                    session.headers.update({"Connection": "keep-alive"})
                    if hasattr(session, "connection_pool"):
                        session.connection_pool.clear()

    except Exception as e:
        log_error(e)
        emit("error", f"{type(e).__name__}: {str(e)}")
        emit("log", traceback.format_exc())


def print_timing_chart(metrics):
    """Prints timing chart from global metrics"""
    base_time = metrics["vad_start"]
    events = [
        ("User stopped speaking", metrics["vad_start"]),
        ("VAD started", metrics["vad_start"]),
        ("Transcription started", metrics["transcription_start"]),
        ("LLM first token", metrics["llm_first_token"]),
        ("Audio queued", metrics["audio_queued"]),
        ("First audio played", metrics["first_audio_play"]),
        ("Playback started", metrics["playback_start"]),
        ("End-to-end response", metrics["end"]),
    ]

    print("\nTiming Chart:")
    print(f"{'Event':<25} | {'Time (s)':>9} | {'Δ+':>6}")
    print("-" * 45)

    prev_time = base_time
    for name, t in events:
        if t is None:
            continue
        elapsed = t - base_time
        delta = t - prev_time
        print(f"{name:<25} | {elapsed:9.2f} | {delta:6.2f}")
        prev_time = t


# Slash commands: name -> (handler method, description). Drives both the
# completion menu and the /help panel.
SLASH_COMMANDS = [
    ("/quit", "Quit the application (asks for confirmation)"),
    ("/clear", "Clear the chat display"),
    ("/stop", "Interrupt the current playback / response"),
    ("/now", "Trigger the idle event immediately"),
    ("/idle", "Enter idle mode now (same as /now)"),
    ("/pause", "Suspend voice output and the idle countdown"),
    ("/play", "Resume from pause"),
    ("/slap", "Erase queued Twitch messages, or /slap @user for just theirs"),
    ("/config", "Pick the microphone and speaker devices"),
    ("/voice", "Show voice-input status; /voice on|off enables/disables VAD + transcription"),
    ("/memory", "Show memory status; /memory on|off toggles Qdrant long-term memory (RAM fallback)"),
    ("/new", "Start a new chat session (clears LLM history)"),
    ("/help", "Show this list of slash commands"),
]


def _twitch_usernames() -> list[str]:
    """Usernames currently present in the queued Twitch chat events."""
    names = set()
    for e in twitch_collector.snapshot(max_size=500):
        m = re.match(r"\[Chat\] (\S+):", e["text"])
        if m:
            names.add(m.group(1))
    return sorted(names)


class ChatInput(Input):
    """Text input that sends lines to the pipeline thread, with slash-command
    support: typing / shows a completion menu (Tab/arrows/Enter like opencode)."""

    BINDINGS = [
        Binding("escape", "unfocus", "Unfocus input"),
        Binding("up", "suggestion(-1)", "Previous command", show=False),
        Binding("down", "suggestion(1)", "Next command", show=False),
        Binding("tab", "accept_suggestion", "Accept command", show=False),
    ]

    def on_mount(self):
        self._suggestion_matches: list = []
        self._suggestion_index = 0

    def action_unfocus(self):
        self._hide_suggestions()
        try:
            self.screen.set_focus(None)
        except Exception as e:
            log_error(e)
            emit("error", f"unfocus error: {type(e).__name__}: {e}")

    def on_input_changed(self, event: Input.Changed):
        global last_typing_activity
        last_typing_activity = time.time()
        if idle_mode:
            interrupt_event.set()
        value = event.value
        if value.startswith("/") and " " not in value:
            prefix = value[1:].lower()
            self._suggestion_matches = [
                c for c in SLASH_COMMANDS if c[0].startswith("/" + prefix)
            ]
            self._suggestion_index = 0
            self._render_suggestions()
        elif value.startswith("/slap ") and value.strip():
            raw = value[len("/slap "):].lstrip("@")
            usernames = [
                u for u in _twitch_usernames()
                if u.lower().startswith(raw.lower())
            ]
            self._suggestion_matches = [
                (f"/slap @{u}", f"erase queued chat from @{u}") for u in usernames
            ]
            self._suggestion_index = 0
            self._render_suggestions()
        else:
            self._hide_suggestions()

    def on_input_submitted(self, event: Input.Submitted):
        try:
            if self._suggestion_matches:
                cast("SpeechTUI", self.app)._run_command(
                    self._suggestion_matches[self._suggestion_index][0]
                )
                self.value = ""
                self._hide_suggestions()
                return
            text = event.value.strip()
            if text.startswith("/"):
                cast("SpeechTUI", self.app)._run_command(text)
                self.value = ""
                self._hide_suggestions()
                return
            if text:
                text_input_queue.put(text)
                interrupt_event.set()
                emit("activity")
                emit("transcript", "text", text)
                self.value = ""
        except Exception as e:
            # never raise out of a message handler (Textual would panic the app)
            log_error(e)
            emit("error", f"text input error: {type(e).__name__}: {e}")

    def action_suggestion(self, delta: int):
        if self._suggestion_matches:
            self._suggestion_index = (
                self._suggestion_index + delta
            ) % len(self._suggestion_matches)
            self._render_suggestions()
        elif delta < 0:
            self.action_home()
        else:
            self.action_end()

    def action_accept_suggestion(self):
        if self._suggestion_matches:
            name = self._suggestion_matches[self._suggestion_index][0]
            self.value = name
            self.cursor_position = len(name)
        else:
            self.app.action_focus_next()

    def _suggestions_widget(self):
        return self.app.query_one("#cmd-suggestions", Static)

    def _render_suggestions(self):
        matches = self._suggestion_matches
        if not matches:
            self._hide_suggestions()
            return
        t = Text()
        for i, (name, desc) in enumerate(matches):
            selected = i == self._suggestion_index
            t.append(f"  {name:<8}", style="bold reverse" if selected else "")
            t.append(f"  {desc}", style="" if selected else "dim")
            if i < len(matches) - 1:
                t.append("\n")
        self._suggestions_widget().update(t)
        self._suggestions_widget().styles.display = "block"

    def _hide_suggestions(self):
        self._suggestion_matches = []
        try:
            self._suggestions_widget().styles.display = "none"
        except Exception:
            pass


class QuitConfirm(Screen):
    """Confirmation dialog shown before the app finally exits."""

    CSS = """
    QuitConfirm { align: center middle; }
    #quit-dialog {
        width: 52;
        height: auto;
        padding: 1 2;
        border: round $error;
        background: $panel;
    }
    #quit-dialog Static { height: 1; }
    #dialog-buttons { height: 3; align-horizontal: center; margin-top: 1; }
    #dialog-buttons Button { width: 16; margin: 0 1; }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="quit-dialog"):
            yield Static("Are you sure you want to quit?", classes="dialog-title")
            with Horizontal(id="dialog-buttons"):
                yield Button("Cancel", id="quit-cancel")
                yield Button("Quit", id="quit-yes", variant="error")
        yield Footer()

    def action_cancel(self):
        self.app.pop_screen()

    @on(Button.Pressed, "#quit-cancel")
    def _cancel(self):
        self.app.pop_screen()

    @on(Button.Pressed, "#quit-yes")
    def _quit(self):
        self.app.exit()


class HelpScreen(Screen):
    """Modal panel listing the available slash commands."""

    CSS = """
    HelpScreen { align: center middle; }
    #help-dialog {
        width: 62;
        height: auto;
        padding: 1 2;
        border: round $accent;
        background: $panel;
    }
    #help-title { text-style: bold; height: 1; }
    #help-list { height: auto; margin-top: 1; }
    #help-list Static { height: 1; }
    #help-hint { color: $text-muted; height: 1; margin-top: 1; }
    """

    BINDINGS = [
        Binding("escape", "close", "Dismiss"),
        Binding("q", "close", "Dismiss"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog"):
            yield Static("Slash Commands", id="help-title")
            with Vertical(id="help-list"):
                for name, desc in SLASH_COMMANDS:
                    yield Static(f"[bold #7fb4ff]{name:<8}[/]  {desc}")
            yield Static("Press Escape or q to close.", id="help-hint")
        yield Footer()

    def action_close(self):
        self.app.pop_screen()


class ConfigScreen(Screen):
    """Modal dialog to pick the microphone (input) and speaker (output) devices."""

    CSS = """
    ConfigScreen { align: center middle; }
    #config-dialog {
        width: 76;
        height: auto;
        padding: 1 2;
        border: round $accent;
        background: $panel;
    }
    #config-dialog OptionList { height: 8; margin-bottom: 1; }
    .config-label { text-style: bold; height: 1; }
    #config-hint { color: $text-muted; height: 1; margin-bottom: 1; }
    #config-actions { height: 3; align-horizontal: center; margin-top: 1; }
    #config-actions Button { width: 14; margin: 0 1; }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        inputs, outputs = list_audio_devices()
        with Vertical(id="config-dialog"):
            yield Static("Audio Devices", id="help-title")
            yield Static("Arrows pick a device, Save applies. Empty list = system default.", id="config-hint")
            yield Static("Microphone:", classes="config-label")
            yield OptionList(*inputs, id="mic-list")
            yield Static("Speaker:", classes="config-label")
            yield OptionList(*outputs, id="spk-list")
            with Horizontal(id="config-actions"):
                yield Button("Save", id="config-save", variant="success")
                yield Button("Cancel", id="config-cancel")
        yield Footer()

    def on_mount(self):
        # children aren't queryable in on_mount for pushed screens; defer a frame
        self.call_after_refresh(self._preselect)

    def _preselect(self):
        try:
            for widget_id, current in (("#mic-list", settings.MIC_DEVICE), ("#spk-list", settings.SPEAKER_DEVICE)):
                ol = self.query_one(widget_id, OptionList)
                if current:
                    for i, opt in enumerate(ol.options):
                        if opt.prompt == current:
                            ol.highlighted = i
                            break
        except Exception:
            pass

    def _selected(self, widget_id) -> str:
        ol = self.query_one(widget_id, OptionList)
        idx = ol.highlighted
        if idx is None or not ol.options or idx >= len(ol.options):
            return ""
        return str(ol.get_option_at_index(idx).prompt)

    def action_cancel(self):
        self.app.pop_screen()

    @on(Button.Pressed, "#config-cancel")
    def _cancel(self):
        self.app.pop_screen()

    @on(Button.Pressed, "#config-save")
    def _save(self):
        mic = self._selected("#mic-list")
        spk = self._selected("#spk-list")
        settings.MIC_DEVICE = mic
        settings.SPEAKER_DEVICE = spk
        save_device_settings(mic, spk)
        self.app.pop_screen()
        cast("SpeechTUI", self.app)._bot_reply(
            f"Audio devices set: mic '{mic or 'default'}', speaker '{spk or 'default'}'."
        )


# mIRC's classic per-nick color palette (hex for the 16-color table).
_IRC_NICK_COLORS = [
    "#FF0000",  # red
    "#0000FC",  # light blue
    "#00FC00",  # bright green
    "#FC7F00",  # orange
    "#009300",  # green
    "#FFFF00",  # yellow
    "#FF00FF",  # pink
    "#9C009C",  # purple
    "#7F0000",  # maroon
    "#00FFFF",  # aqua
    "#009393",  # teal
    "#7F7F7F",  # grey
    "#00007F",  # blue
]


def _irc_nick_color(nick: str) -> str:
    """Deterministically assign an mIRC nick color so each user keeps theirs."""
    return _IRC_NICK_COLORS[sum(ord(c) for c in nick) % len(_IRC_NICK_COLORS)]


class SpeechTUI(App):
    TITLE = "On-Device Speech-to-Speech AI"

    BINDINGS = [
        Binding("t", "focus_input", "Text input"),
    ]

    CSS = """
    #menu-bar { dock: top; height: auto; background: $panel; }
    #menu-row { height: 1; }
    #menu-system { height: 1; min-width: 10; }
    .menu-title { height: 1; color: $text-muted; padding: 0 1; }
    #system-menu { display: none; height: auto; border: round $accent; background: $panel; }
    #system-menu.-open { display: block; }
    #system-quit { width: 20; height: 1; }
    #main { height: 1fr; }
    #middle { height: 1fr; }
    #chat-scroll {
        width: 3fr;
        border: round $accent;
        overflow-x: hidden;
    }
    #chat-scroll > Static {
        width: 100%;
        text-wrap: wrap;
    }
    #right { width: 1fr; }
    #twitch-log { height: 2fr; border: round $accent; }
    #idle-pane { height: 1fr; border: round $accent; }
    #idle-pane Label { padding: 0 1; }
    #status-text { text-style: bold; height: 1; }
    #idle-bar { height: 1; margin: 0 1 1 1; }
    #last-activity { color: $text-muted; height: 1; }
    #status-bar { dock: bottom; height: 1; color: $text; background: $panel; padding: 0 1; }
    #text-input { dock: bottom; height: 3; }
    #cmd-suggestions {
        display: none;
        height: auto;
        max-height: 6;
        padding: 0 1;
        background: $panel;
        border: round $accent;
        margin: 0 1 1 1;
    }
    """

    _STATUS_COLORS = {
        "INITIALIZING": "cyan",
        "LISTENING": "green",
        "TRANSCRIBING": "yellow",
        "THINKING": "magenta",
        "SPEAKING": "blue",
        "IDLE": "red",
        "PAUSED": "grey",
    }

    def __init__(self):
        super().__init__()
        self.status = "STARTING"
        self.twitch_cache = None
        self._twitch_count = 0
        self._live_static = None
        self._stream_buf = ""
        self._spoken_buf = ""
        self._notice_seen = set()

    def compose(self) -> ComposeResult:
        with Vertical(id="menu-bar"):
            with Horizontal(id="menu-row"):
                yield Button("System", id="menu-system")
                yield Static("On-Device Speech-to-Speech AI", classes="menu-title")
            with Vertical(id="system-menu"):
                yield Button("(Q)uit", id="system-quit")
        yield Header(show_clock=True)
        with Vertical(id="main"):
            with Horizontal(id="middle"):
                yield VerticalScroll(id="chat-scroll")
                with Vertical(id="right"):
                    yield self._rich_log("twitch-log", "Twitch chat (pending)", wrap=True)
                    with Vertical(id="idle-pane"):
                        yield Label(id="status-text")
                        yield ProgressBar(total=settings.MAX_IDLE_TIME, show_eta=False, show_percentage=False, id="idle-bar")
                        yield Label(id="idle-count")
                        yield Label(id="last-activity")
            yield Static(id="status-bar")
            yield Static(id="cmd-suggestions")
            yield ChatInput(id="text-input", placeholder="Type a message for the bot, or / for commands. Enter to send.")
        yield Footer()

    def _rich_log(self, widget_id, title, max_lines=200, wrap=False):
        w = RichLog(id=widget_id, markup=True, max_lines=max_lines, wrap=wrap)
        w.border_title = title
        return w

    def on_mount(self):
        self.chat: VerticalScroll = self.query_one("#chat-scroll", VerticalScroll)
        self.twitch_log: RichLog = self.query_one("#twitch-log", RichLog)
        self.status_bar: Static = self.query_one("#status-bar", Static)
        self.status_text: Label = self.query_one("#status-text", Label)
        self.idle_bar: ProgressBar = self.query_one("#idle-bar", ProgressBar)
        self.idle_count: Label = self.query_one("#idle-count", Label)
        self.last_activity_label: Label = self.query_one("#last-activity", Label)

        _install_error_routing()
        self.set_interval(0.1, self._poll_events)
        self.set_interval(0.25, self._update_status)
        threading.Thread(target=pipeline_main, daemon=True).start()
        self._set_status("INITIALIZING")

    def on_unmount(self):
        shutdown_event.set()
        _restore_error_routing()

    def action_focus_input(self):
        self.query_one("#text-input", Input).focus()

    @on(Button.Pressed, "#menu-system")
    def _toggle_menu(self):
        self.query_one("#system-menu").toggle_class("-open")

    @on(Button.Pressed, "#system-quit")
    def _open_quit_confirm(self):
        self.query_one("#system-menu").remove_class("-open")
        self.push_screen(QuitConfirm())

    # ---- slash commands ----

    def _run_command(self, raw: str):
        try:
            parts = raw.strip().split(None, 1)
            name = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            handler = {
                "/quit": self._cmd_quit,
                "/clear": self._cmd_clear,
                "/stop": self._cmd_stop,
                "/now": self._cmd_now,
                "/idle": self._cmd_idle,
                "/pause": self._cmd_pause,
                "/play": self._cmd_play,
                "/slap": self._cmd_slap,
                "/config": self._cmd_config,
                "/voice": self._cmd_voice,
                "/memory": self._cmd_memory,
                "/new": self._cmd_new,
                "/help": self._cmd_help,
            }.get(name)
            if handler:
                handler(arg)
            else:
                self._notice(f"Unknown command: {raw}", color="bold red")
        except Exception as e:
            log_error(e)
            self._log_tui_error("command", e)

    def _cmd_quit(self, arg=""):
        self.query_one("#system-menu").remove_class("-open")
        self.push_screen(QuitConfirm())

    def _cmd_clear(self, arg=""):
        self.chat.remove_children()
        self._reset_live()
        self._notice_seen = set()
        self._bot_reply("Chat cleared.")

    def _cmd_stop(self, arg=""):
        interrupt_event.set()
        self._bot_reply("Stopped. I've silenced my output.")

    def _cmd_now(self, arg=""):
        now_event.set()
        self._bot_reply("Idle countdown set to zero — I'll start talking now.")

    def _cmd_idle(self, arg=""):
        now_event.set()
        self._bot_reply("Entering idle mode — I'll start talking now.")

    def _cmd_pause(self, arg=""):
        pause_event.set()
        self._set_status("PAUSED")
        self._bot_reply("Paused. My voice and the idle countdown are suspended. Type /play to resume.")

    def _cmd_play(self, arg=""):
        if pause_event.is_set():
            pause_event.clear()
            self._bot_reply("Resumed. I'm listening again.")
        else:
            self._bot_reply("I wasn't paused.")

    def _cmd_slap(self, arg=""):
        arg = arg.strip()
        if not arg:
            n = twitch_collector.clear_all()
            self._bot_reply(f"Slapped the chat — erased {n} queued Twitch event(s).")
        else:
            username = arg.lstrip("@").strip().lower()
            n = twitch_collector.clear_by_user(username)
            if n:
                self._bot_reply(f"Slapped @{username} — erased {n} queued message(s).")
            else:
                self._bot_reply(f"@{username} has no queued messages to slap.")

    def _cmd_config(self, arg=""):
        self.push_screen(ConfigScreen())

    def _cmd_voice(self, arg=""):
        arg = arg.strip().lower()
        if arg == "on":
            voice_event.set()
            self._bot_reply("Voice input enabled: VAD and voice transcription are ON.")
        elif arg == "off":
            voice_event.clear()
            self._bot_reply("Voice input disabled: VAD and voice transcription are OFF.")
        else:
            state = "ON" if voice_event.is_set() else "OFF"
            self._bot_reply(f"VAD and voice transcription are currently {state}.")

    def _cmd_memory(self, arg=""):
        arg = arg.strip().lower()
        if arg == "on":
            if memory_status["enabled"]:
                self._bot_reply(f"Qdrant long-term memory is already ON ({memory_status['backend']}).")
            else:
                memory_request_queue.put("on")
                self._bot_reply("Switching to Qdrant long-term memory...")
        elif arg == "off":
            if not memory_status["enabled"]:
                self._bot_reply("Long-term memory is already OFF.")
            else:
                memory_request_queue.put("off")
                self._bot_reply("Switching off long-term memory...")
        else:
            if memory_status["enabled"]:
                self._bot_reply(f"Qdrant long-term memory is ON ({memory_status['backend']}).")
            else:
                self._bot_reply("Long-term memory is OFF.")

    def _cmd_new(self, arg=""):
        new_chat_event.set()
        self._bot_reply("Starting a new chat session — history cleared.")

    def _cmd_help(self, arg=""):
        self.push_screen(HelpScreen())

    # ---- main chat window ----

    def _chat_line(self, renderable, classes=""):
        """Append a line to the main chat window and scroll to the bottom."""
        line = Static(renderable, classes=classes)
        self.chat.mount(line)
        self.chat.scroll_end(animate=False)
        return line

    def _notice(self, text, color=""):
        """mIRC-style notice line in the main window."""
        t = Text()
        t.append("*** ", style="bold cyan")
        t.append(text, style=color)
        return self._chat_line(t, classes="notice")

    def _bot_reply(self, text: str):
        """Bot's own chat line, e.g. an acknowledgment to a slash command."""
        t = Text()
        t.append("Bot: ", style="bold #7fb4ff")
        t.append(text)
        return self._chat_line(t)

    def _clear_notices(self):
        for w in list(self.chat.query(".notice")):
            w.remove()

    def _reset_live(self):
        self._live_static = None
        self._stream_buf = ""
        self._spoken_buf = ""

    def _render_live(self):
        """Re-render the streaming bot line: spoken words normal, the rest grey."""
        if self._live_static is None:
            self._live_static = self._chat_line("")
        words = self._stream_buf.split()
        spoken_count = len(self._spoken_buf.split())
        spoken_part = " ".join(words[:spoken_count])
        unspoken_part = " ".join(words[spoken_count:])
        t = Text()
        t.append("Bot: ", style="bold #7fb4ff")
        if spoken_part:
            t.append(spoken_part)
            if unspoken_part:
                t.append(" " + unspoken_part, style="dim")
        elif unspoken_part:
            t.append(unspoken_part, style="dim")
        self._live_static.update(t)
        self.chat.scroll_end(animate=False)

    def _finalize_live(self, bot_text):
        """Replace the streaming line with the completed response."""
        if self._live_static is not None:
            self._live_static.remove()
            self._live_static = None
        if bot_text:
            bot_text = " ".join(bot_text.split())
            t = Text()
            t.append("Bot: ", style="bold #7fb4ff")
            t.append(bot_text)
            self._chat_line(t)

    # ---- status footer ----

    def _set_status(self, status: str):
        self.status = status
        self._update_status()

    def _timing_summary(self) -> str:
        t = timing_info
        if t["vad_start"] is None:
            return ""
        bits = []
        if t["transcription_duration"]:
            bits.append(f"trans {t['transcription_duration']:.1f}s")
        if t["llm_first_token"]:
            bits.append(f"llm {t['llm_first_token'] - t['vad_start']:.1f}s")
        if t["first_audio_play"]:
            bits.append(f"audio {t['first_audio_play'] - t['vad_start']:.1f}s")
        if t["end"]:
            bits.append(f"ttl {t['end'] - t['vad_start']:.1f}s")
        return "  ".join(bits)

    def _update_status(self):
        try:
            left = f"[bold {self._STATUS_COLORS.get(self.status, 'white')}]{self.status}[/]"
            if self.status == "LISTENING":
                remaining = max(0.0, settings.MAX_IDLE_TIME - _idle_elapsed())
                mid = f"idle [bold]{remaining:5.1f}s[/]"
            else:
                mid = f"in turn: {self.status}"
            parts = [left, mid, f"twitch [bold]{self._twitch_count}[/] pending"]
            timing = self._timing_summary()
            if timing:
                parts.append(timing)
            self.status_bar.update("  │  ".join(parts))
        except Exception as e:
            log_error(e)
            self._log_tui_error("status bar", e)
        self._update_idle_pane()

    def _update_idle_pane(self):
        try:
            self.status_text.update(f"[bold {self._STATUS_COLORS.get(self.status, 'white')}]{self.status}[/]")
            if self.status == "LISTENING":
                remaining = max(0.0, settings.MAX_IDLE_TIME - _idle_elapsed())
                self.idle_count.update(f"idle in [bold]{remaining:5.1f}s[/]")
                self.idle_bar.update(progress=remaining, total=settings.MAX_IDLE_TIME)
            else:
                self.idle_count.update(f"in turn: {self.status}")
                self.idle_bar.update(progress=settings.MAX_IDLE_TIME, total=settings.MAX_IDLE_TIME)
            self.last_activity_label.update(
                f"Last activity: {time.strftime('%H:%M:%S', time.localtime(pipeline_last_activity))}"
            )
        except Exception as e:
            log_error(e)
            self._log_tui_error("idle pane", e)

    # ---- event handling ----

    def _poll_events(self):
        try:
            while True:
                kind, *payload = _event_queue.get_nowait()
                self._handle(kind, *payload)
        except queue.Empty:
            pass
        try:
            self._update_twitch()
        except Exception as e:
            log_error(e)
            self._log_tui_error("twitch panel", e)

    def _log_tui_error(self, context: str, error: Exception):
        """Write an exception to the chat window as a notice; must never raise
        (a raised exception here would make Textual panic and tear down the TUI)."""
        try:
            self._notice(f"{context}: {type(error).__name__}: {error}", color="bold red")
            for line in traceback.format_exc().splitlines():
                self._notice(line, color="dim")
        except Exception:
            pass

    def _handle(self, kind: str, *payload):
        try:
            if kind == "status":
                self._set_status(payload[0])
            elif kind == "log":
                self._log_notice(payload[0])
            elif kind == "transcript":
                source, text = payload
                t = Text()
                t.append(f"You ({source}): ", style="bold #7fd8a4")
                t.append(text)
                self._chat_line(t)
                _append_chat_file("in.txt", text)
            elif kind == "bot_token":
                self._stream_buf += payload[0]
                self._render_live()
            elif kind == "bot_spoken":
                self._spoken_buf = (self._spoken_buf + " " + payload[0]).strip()
                self._render_live()
            elif kind == "turn":
                _, bot_text = payload
                self._finalize_live(bot_text)
                self._reset_live()
            elif kind == "turn_start":
                self._clear_notices()
                self._notice_seen = set()
                self._finalize_live("")
                self._reset_live()
            elif kind == "error":
                self._notice(payload[0], color="bold red")
        except Exception as e:
            log_error(e)
            self._log_tui_error("event handler", e)

    def _log_notice(self, line: str):
        if line in self._notice_seen:
            return
        self._notice_seen.add(line)
        if line.startswith("[TTS") or line.startswith("[Twitch"):
            color = "dim"
        elif line.startswith("[Idle") or line.startswith("[Random"):
            color = "yellow"
        elif "Error" in line or "Traceback" in line:
            color = "bold red"
        else:
            color = ""
        self._notice(line, color=color)

    def _update_twitch(self):
        events = twitch_collector.snapshot(max_size=200)
        prev_count = self._twitch_count
        self._twitch_count = len(events)
        self.twitch_log.border_title = f"Twitch chat ({len(events)} pending)"
        if self._twitch_count != prev_count:
            self._update_status()
        rendered = [self._twitch_line(e) for e in events]
        key = repr(rendered)
        if key != self.twitch_cache:
            self.twitch_cache = key
            self.twitch_log.clear()
            if rendered:
                for line in rendered:
                    self.twitch_log.write(line)
            else:
                self.twitch_log.write("[dim]No pending chat events[/]")

    def _twitch_line(self, event: dict) -> Text:
        """Build an mIRC-styled line: dim timestamp, per-nick colored name."""
        t = Text()
        t.append(f"[{time.strftime('%H:%M:%S', time.localtime(event['timestamp']))}] ", style="dim")
        m = re.match(r"\[Chat\] (\S+): (.*)$", event["text"], re.S)
        if m:
            nick, message = m.groups()
            color = _irc_nick_color(nick)
            t.append(f"<{nick}> ", style=f"bold {color}")
            t.append(message)
        else:
            t.append(event["text"], style="dim")
        return t


def main():
    SpeechTUI().run()


if __name__ == "__main__":
    main()
