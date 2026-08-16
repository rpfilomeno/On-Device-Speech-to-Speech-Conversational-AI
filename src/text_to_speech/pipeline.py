"""Text-driven pipeline: LLM streaming, chunked TTS playback, memory, and the
idle/Twitch auto-talk loop. Runs in a background thread and reports to the TUI
through the state event bus."""

import queue
import random
import threading
import time
import traceback

import numpy as np
import requests

from src.utils import (
    VoiceGenerator,
    get_ai_response,
    twitch_collector,
    twitch_bot_manager,
)
from src.utils.audio_queue import AudioGenerationQueue
from src.utils.config import save_settings, log_error, settings
from src.utils.llm import parse_stream_chunk, fetch_context_window
from src.utils.memory import Memory, MemoryWorker, RamMemory
from src.utils.speech import TurnAudioPlayer
from src.utils.text_chunker import TextChunker

from . import pacing
from . import state
from . import history


def process_input(
    session: requests.Session,
    user_input: str,
    messages: list,
    generator: VoiceGenerator,
    speed: float,
    memory: MemoryWorker | None = None,
) -> tuple[bool, np.ndarray | None]:
    """Processes user input, generates a response, and handles audio output.

    Args:
        session (requests.Session): The requests session to use.
        user_input (str): The user's input text.
        messages (list): The list of messages to send to the LLM.
        generator (VoiceGenerator): The voice generator object.
        speed (float): The playback speed.
        memory (Memory, optional): Long-term vector memory to recall from / write to.

    Returns:
        tuple[bool, None]: A tuple containing a boolean indicating if the process was interrupted and None.
    """
    state.timing_info = {k: None for k in state.timing_info}
    state.timing_info["vad_start"] = time.perf_counter()

    messages.append({"role": "user", "content": user_input})
    state.emit("status", "THINKING")

    memory_block = None
    if memory is not None:
        try:
            recalled = memory.search(user_input)
            if recalled:
                memory_block = "\n".join(f"- {m}" for m in recalled)
        except Exception as e:
            log_error(e)
            state.emit("log", f"Memory recall failed: {e}")

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

    if history.trim_history_to_budget(messages):
        state.emit("log", "[Chat] Trimmed history to fit within context budget.")
        llm_messages = llm_messages_for(messages)

    state.emit("turn_start", user_input)
    state.interrupt_event.clear()
    start_time = time.time()
    audio_queue: AudioGenerationQueue | None = None
    interrupted = False
    interrupt_data: np.ndarray | None = None
    try:
        audio_queue = AudioGenerationQueue(generator, speed)
        audio_queue.start()

        def worker_runner():
            nonlocal interrupted, interrupt_data
            was_int, int_data = audio_playback_worker(audio_queue)
            interrupted = was_int
            interrupt_data = int_data

        playback_thread = threading.Thread(target=worker_runner)
        playback_thread.daemon = True
        playback_thread.start()

        retry = 0
        session_reset = False
        bot_text = ""
        while True:
            prompt_tokens = sum(history.estimate_tokens(m.get("content", "")) for m in llm_messages)
            max_tokens = min(settings.MAX_TOKENS, max(1, settings.CONTEXT_WINDOW - prompt_tokens))
            response_stream = get_ai_response(
                session=session,
                messages=llm_messages,
                llm_model=settings.LLM_MODEL,
                llm_url=settings.LM_STUDIO_URL,
                max_tokens=max_tokens,
                stream=True,
            )

            if not response_stream:
                if not history.retry_llm(messages, retry, "LLM request failed"):
                    state.emit("log", "Failed to get AI response stream.")
                    break
                retry += 1
                llm_messages = llm_messages_for(messages)
                continue

            chunker = TextChunker()
            complete_response = []
            for chunk in response_stream:
                if state.interrupt_event.is_set():
                    state.emit("log", "[Command] Stop: stream aborted.")
                    break
                data = parse_stream_chunk(chunk)
                if not data or "choices" not in data:
                    continue

                choice = data["choices"][0]
                if "delta" in choice and "content" in choice["delta"]:
                    content = choice["delta"]["content"]
                    if content:
                        if not state.timing_info["llm_first_token"]:
                            state.timing_info["llm_first_token"] = time.perf_counter()
                        state.emit("bot_token", content)
                        chunker.current_text.append(content)

                        text = "".join(chunker.current_text)
                        settings.TARGET_SIZE = pacing.adaptive_target_words()
                        if chunker.should_process(text):
                            if not state.timing_info["audio_queued"]:
                                state.timing_info["audio_queued"] = time.perf_counter()
                            remaining = chunker.process(text, audio_queue)
                            chunker.current_text = [remaining] if remaining else []
                            processed_len = len(text) - len(remaining)
                            if processed_len > 0:
                                complete_response.append(text[:processed_len])

                if choice.get("finish_reason") == "stop":
                    break

            settings.TARGET_SIZE = pacing.adaptive_target_words()
            final_flushed = chunker.flush(audio_queue)
            if final_flushed:
                complete_response.append(final_flushed)

            bot_text = " ".join(" ".join(complete_response).split())
            if bot_text:
                break

            if retry >= history._MAX_LLM_RETRIES and not session_reset:
                messages[:] = [
                    {"role": "system", "content": settings.DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                ]
                session_reset = True
                retry = 0
                state.emit("log", "[Chat] LLM blank after history trim — starting a new session.")
                llm_messages = llm_messages_for(messages)
                continue

            if not history.retry_llm(messages, retry, "LLM returned empty (context too long?)"):
                state.emit("log", "Failed to get a response after trimming history.")
                break
            retry += 1
            llm_messages = llm_messages_for(messages)

        if bot_text:
            messages.append({"role": "assistant", "content": bot_text})
        print()

        audio_queue.stop()
        playback_thread.join()
        state._clear_chat_files()

        state.timing_info["end"] = time.perf_counter()
        print_timing_chart(state.timing_info)
        if bot_text:
            state.emit("turn", user_input, bot_text)
        if memory is not None and bot_text:
            try:
                memory.store("user", user_input)
                memory.store("assistant", bot_text)
            except Exception as e:
                log_error(e)
                state.emit("log", f"Memory store failed: {e}")
        state.emit("status", "LISTENING")
        return interrupted, interrupt_data

    except Exception as e:
        log_error(e)
        state.emit("log", f"Error during streaming: {str(e)}")
        if audio_queue is not None:
            audio_queue.stop()
        return False, None


def audio_playback_worker(audio_queue) -> tuple[bool, np.ndarray | None]:
    """Manages audio playback in a separate thread, handling interruptions.

    One TurnAudioPlayer stays open for the whole turn: gapless chunk playback
    and a single stop/clear path. No mic is opened (text-only variant).

    Args:
        audio_queue (AudioGenerationQueue): The audio queue object.

    Returns:
        tuple[bool, None]: A tuple containing a boolean indicating if the playback was interrupted and the interrupt audio data.
    """
    was_interrupted = False
    interrupt_audio = None

    played_any = False
    wait_start = None
    gaps: list[float] = []

    try:
        player = TurnAudioPlayer(
            stop_events=[state.interrupt_event, state.pause_event],
            monitor_input=False,
        )
    except Exception as e:
        log_error(e)
        state.emit("log", f"Error opening audio player: {str(e)}")
        return False, None

    try:
        while True:
            if state.interrupt_event.is_set():
                state.interrupt_event.clear()
                state.emit("log", "[Command] Stop: playback interrupted, clearing queues.")
                player.stop()
                audio_queue.clear_queues()
                break
            if state.pause_event.is_set():
                audio_queue.clear_queues()
                player.stop()
                time.sleep(settings.PLAYBACK_DELAY)
                continue

            audio_data, sentence, is_first = audio_queue.get_next_audio()
            if audio_data is not None:
                if wait_start is not None:
                    gap = time.time() - wait_start
                    gaps.append(gap)
                    pacing._behind_ema += pacing._BEHIND_EMA_ALPHA * (gap - pacing._behind_ema)
                    wait_start = None
                else:
                    pacing._behind_ema *= pacing._BEHIND_DECAY
                played_any = True
                if not state.timing_info["first_audio_play"]:
                    state.timing_info["first_audio_play"] = time.perf_counter()
                    state.emit("status", "SPEAKING")

                if is_first:
                    state.emit("bot_spoken", sentence)
                    state._append_chat_file("out.txt", sentence)
                if settings.LOG_TTS_CHUNKS:
                    state.emit("log", f"[TTS Playing] {sentence!r}")
                player.push(audio_data)
            else:
                if played_any and wait_start is None and not player.is_playing():
                    wait_start = time.time()
                time.sleep(pacing.adaptive_poll_delay(pacing._behind_ema))

            if (
                not audio_queue.is_running
                and audio_queue.sentence_queue.empty()
                and audio_queue.audio_queue.empty()
            ):
                player.flush()
                break

    except Exception as e:
        log_error(e)
        state.emit("log", f"Error in audio playback: {str(e)}")
    finally:
        player.stop()

    save_settings(
        TARGET_SIZE=round(settings.TARGET_SIZE),
        PLAYBACK_DELAY=settings.PLAYBACK_DELAY,
    )

    if gaps:
        mean = sum(gaps) / len(gaps)
        stddev = (sum((g - mean) ** 2 for g in gaps) / len(gaps)) ** 0.5
        state.emit(
            "log",
            f"[Playback] jitter (inter-chunk gap): stddev {stddev * 1000:.0f}ms, "
            f"mean {mean * 1000:.0f}ms, max {max(gaps) * 1000:.0f}ms over {len(gaps)} gap(s).",
        )
        if len(gaps) >= 2:
            used_target = round(settings.TARGET_SIZE)
            used_delay = settings.PLAYBACK_DELAY
            best_target = pacing.record_jitter(used_target, stddev * 1000)
            best_delay = pacing.record_delay_jitter(stddev * 1000)
            if best_target is not None and best_target != used_target:
                settings.TARGET_SIZE = best_target
                save_settings(TARGET_SIZE=best_target)
                state.emit(
                    "log",
                    f"[Jitter Stats] Best TARGET_SIZE is now {best_target} "
                    f"(lowest mean jitter); saved to settings.json.",
                )
            if best_delay is not None:
                settings.PLAYBACK_DELAY = best_delay
                save_settings(PLAYBACK_DELAY=best_delay)
                state.emit(
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


def pipeline_main():
    """Runs the text-driven pipeline in a background thread; reports to the TUI."""
    try:
        state.emit("status", "INITIALIZING")
        state.emit("log", "Initializing voice generator (Pocket TTS remote streaming)...")
        generator = VoiceGenerator()
        result = generator.initialize(
            pocket_tts_url=settings.POCKET_TTS_URL,
            pocket_tts_voice=settings.POCKET_TTS_VOICE,
        )
        state.emit("log", result)
        speed = settings.SPEED

        session = requests.Session()
        messages = [{"role": "system", "content": settings.DEFAULT_SYSTEM_PROMPT}]

        memory, label = init_memory("off")
        state.memory_status.update(enabled=False, backend=label)
        state.emit("log", f"Long-term memory: {label} (use /memory on for Qdrant).")

        if settings.TWITCH_CLIENT_CHANNEL:
            state.emit("log", f"Starting Twitch chat collector for channel: {settings.TWITCH_CLIENT_CHANNEL}...")
            twitch_bot_manager.start(settings.TWITCH_CLIENT_CHANNEL)

        detected_ctx = fetch_context_window(session, settings.LLM_MODEL, settings.LM_STUDIO_URL)
        if detected_ctx:
            settings.CONTEXT_WINDOW = detected_ctx
            state.emit("log", f"LLM context window: {settings.CONTEXT_WINDOW} tokens "
                              f"(prompt kept ≤ {int(settings.CONTEXT_WINDOW * settings.CONTEXT_TRIM_RATIO)}).")
        else:
            state.emit("log", f"Could not detect context window; using fallback {settings.CONTEXT_WINDOW}.")

        try:
            state.emit("log", "Warming up the LLM...")
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
                state.emit("log", "Failed to initialize the AI model!")
        except requests.RequestException as e:
            log_error(e)
            state.emit("log", f"Warmup failed: {str(e)}")

        state.emit("status", "LISTENING")
        state.emit("log", "=== Ready. Type below to chat (spoken reply via TTS). ===")
        state.pipeline_last_activity = time.time()
        was_paused = False

        while not state.shutdown_event.is_set():
            if state.pause_event.is_set():
                if not was_paused:
                    state.emit("status", "PAUSED")
                    state.emit("log", "[Command] Paused: voice output and idle countdown suspended. /play to resume.")
                was_paused = True
                time.sleep(0.1)
                continue
            if was_paused:
                was_paused = False
                state.pipeline_last_activity = time.time()
                state.idle_mode = False
                state.typing_pause_start = None
                state.emit("status", "LISTENING")
                state.emit("log", "[Command] Resumed.")

            try:
                text = state.text_input_queue.get_nowait()
            except queue.Empty:
                text = None

            if text is not None:
                state.idle_mode = False
                state.typing_pause_start = None
                state.pipeline_last_activity = time.time()
                process_input(session, text, messages, generator, speed, memory=memory)
                state.pipeline_last_activity = time.time()
                continue

            try:
                mem_cmd = state.memory_request_queue.get_nowait()
            except queue.Empty:
                mem_cmd = None
            if mem_cmd == "on":
                memory, label = init_memory("on")
                state.memory_status.update(enabled=True, backend=label)
                state.emit("log", f"[Memory] Long-term memory: {label}.")
            elif mem_cmd == "off":
                memory, label = init_memory("off")
                state.memory_status.update(enabled=False, backend=label)
                state.emit("log", f"[Memory] Long-term memory: {label}.")

            if state.new_chat_event.is_set():
                state.new_chat_event.clear()
                messages = [{"role": "system", "content": settings.DEFAULT_SYSTEM_PROMPT}]
                state.emit("log", "[Chat] New session started — LLM history cleared.")

            if state.interrupt_event.is_set():
                state.interrupt_event.clear()
                state.emit("log", "[Command] Stop: nothing in progress.")

            now = time.time()
            # Close a finished typing pause (fold the paused span out of the countdown)
            if state.typing_pause_start is not None and now - state.last_typing_activity >= state._TYPING_PAUSE_SECONDS:
                state.pipeline_last_activity += now - state.typing_pause_start
                state.typing_pause_start = None
            # Typing activity: exit idle mode and suspend the countdown for 5s
            typing_active = now - state.last_typing_activity < state._TYPING_PAUSE_SECONDS
            if typing_active:
                if state.idle_mode:
                    state.idle_mode = False
                    state.emit("status", "LISTENING")
                    state.emit("log", "[Idle] Typing detected — exiting idle mode.")
                if state.typing_pause_start is None:
                    state.typing_pause_start = now

            idle_elapsed = state._idle_elapsed()
            if (
                state.idle_mode
                or (not typing_active and idle_elapsed >= settings.MAX_IDLE_TIME)
                or state.now_event.is_set()
            ):
                if not state.idle_mode:
                    state.idle_mode = True
                    state.now_event.clear()
                    state.emit("status", "IDLE")
                    state.emit("log", f"[Idle Trigger] No activity for {idle_elapsed:.1f}s (MAX_IDLE_TIME={settings.MAX_IDLE_TIME}s).")
                else:
                    state.now_event.clear()

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
                    state.emit("log", f"[Twitch Idle Event] Responding to {len(twitch_events)} collected Twitch event(s)...")
                    state.emit("transcript", "idle", prompt_text)
                else:
                    idle_prompts = settings.get_idle_prompts_list()
                    choices = [p for p in idle_prompts if p != state._last_idle_prompt] or idle_prompts
                    prompt_text = random.choice(choices)
                    state._last_idle_prompt = prompt_text
                    state.emit("log", f"[Random Idle Event] Picked prompt: '{prompt_text}'")
                    state.emit("transcript", "idle", prompt_text)

                process_input(session, prompt_text, messages, generator, speed, memory=memory)
                state.pipeline_last_activity = time.time()
                state.typing_pause_start = None
                state.emit("activity")
                if state.idle_mode:
                    state.emit("status", "IDLE")

            if session is not None:
                session.headers.update({"Connection": "keep-alive"})
                if hasattr(session, "connection_pool"):
                    session.connection_pool.clear()

    except Exception as e:
        log_error(e)
        state.emit("error", f"{type(e).__name__}: {str(e)}")
        state.emit("log", traceback.format_exc())


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
