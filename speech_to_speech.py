import logging
import random
import queue
import sys
import threading
import time
import traceback
import requests
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from src.utils.config import settings
from src.utils import (
    VoiceGenerator,
    get_ai_response,
    play_audio_with_interrupt,
    init_vad_pipeline,
    init_whisper_model,
    detect_speech_segments,
    record_continuous_audio,
    check_for_speech,
    transcribe_audio,
    twitch_collector,
    twitch_bot_manager,
)
from src.utils.audio_queue import AudioGenerationQueue
from src.utils.llm import parse_stream_chunk
from src.utils.text_chunker import TextChunker

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, Label, ProgressBar, RichLog

settings.setup_directories()

# ---- Event bus: pipeline thread(s) -> TUI main thread ----
_event_queue: queue.Queue = queue.Queue()
# TUI -> pipeline thread: typed text to send to the bot
text_input_queue: queue.Queue = queue.Queue()
shutdown_event = threading.Event()

timing_info = {
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


class _LogStream:
    """Captures sys.stdout/sys.stderr so every print() (from pipeline, TTS,
    Twitch bot) and traceback lands in the TUI's log panel instead of being
    printed on top of the screen."""

    def __init__(self):
        self._buf = ""
        self._last_flush = 0.0

    def write(self, text: str):
        # never raise / never touch the real terminal: any bad write is dropped
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
            # show long partial lines (e.g. LLM token streaming) roughly twice a second
            now = time.time()
            if self._buf.strip() and now - self._last_flush >= 0.5:
                self._last_flush = now
                emit("log", self._buf.strip())
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


def process_input(
    session: requests.Session,
    user_input: str,
    messages: list,
    generator: VoiceGenerator,
    speed: float,
) -> tuple[bool, None]:
    """Processes user input, generates a response, and handles audio output.

    Args:
        session (requests.Session): The requests session to use.
        user_input (str): The user's input text.
        messages (list): The list of messages to send to the LLM.
        generator (VoiceGenerator): The voice generator object.
        speed (float): The playback speed.

    Returns:
        tuple[bool, None]: A tuple containing a boolean indicating if the process was interrupted and None.
    """
    global timing_info
    timing_info = {k: None for k in timing_info}
    timing_info["vad_start"] = time.perf_counter()

    messages.append({"role": "user", "content": user_input})
    emit("status", "THINKING")
    emit("turn_start", user_input)
    start_time = time.time()
    try:
        response_stream = get_ai_response(
            session=session,
            messages=messages,
            llm_model=settings.LLM_MODEL,
            llm_url=settings.LM_STUDIO_URL,
            max_tokens=settings.MAX_TOKENS,
            stream=True,
        )

        if not response_stream:
            emit("log", "Failed to get AI response stream.")
            return False, None

        audio_queue = AudioGenerationQueue(generator, speed)
        audio_queue.start()
        chunker = TextChunker()
        complete_response = []

        playback_result = {"interrupted": False, "data": None}

        def worker_runner():
            was_int, int_data = audio_playback_worker(audio_queue)
            playback_result["interrupted"] = was_int
            playback_result["data"] = int_data

        playback_thread = threading.Thread(target=worker_runner)
        playback_thread.daemon = True
        playback_thread.start()

        for chunk in response_stream:
            data = parse_stream_chunk(chunk)
            if not data or "choices" not in data:
                continue

            choice = data["choices"][0]
            if "delta" in choice and "content" in choice["delta"]:
                content = choice["delta"]["content"]
                if content:
                    if not timing_info["llm_first_token"]:
                        timing_info["llm_first_token"] = time.perf_counter()
                    print(content, end="", flush=True)
                    chunker.current_text.append(content)

                    text = "".join(chunker.current_text)
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

        final_flushed = chunker.flush(audio_queue)
        if final_flushed:
            complete_response.append(final_flushed)

        messages.append({"role": "assistant", "content": " ".join(complete_response).strip()})
        print()

        audio_queue.stop()
        playback_thread.join()

        timing_info["end"] = time.perf_counter()
        print_timing_chart(timing_info)
        bot_text = " ".join(complete_response).strip()
        if bot_text:
            emit("turn", user_input, bot_text)
        emit("status", "LISTENING")
        return playback_result["interrupted"], playback_result["data"]

    except Exception as e:
        emit("log", f"Error during streaming: {str(e)}")
        if "audio_queue" in locals():
            audio_queue.stop()
        return False, None


def audio_playback_worker(audio_queue) -> tuple[bool, None]:
    """Manages audio playback in a separate thread, handling interruptions.

    Args:
        audio_queue (AudioGenerationQueue): The audio queue object.

    Returns:
        tuple[bool, None]: A tuple containing a boolean indicating if the playback was interrupted and the interrupt audio data.
    """
    global timing_info
    was_interrupted = False
    interrupt_audio = None

    try:
        while True:
            if audio_queue.audio_queue.empty():
                speech_detected, audio_data = check_for_speech()
                if speech_detected:
                    was_interrupted = True
                    interrupt_audio = audio_data
                    emit("log", "[TTS Interrupted] Speech detected from microphone! Clearing audio queues...")
                    audio_queue.clear_queues()
                    break

            audio_data, sentence = audio_queue.get_next_audio()
            if audio_data is not None:
                if not timing_info["first_audio_play"]:
                    timing_info["first_audio_play"] = time.perf_counter()
                    emit("status", "SPEAKING")

                if settings.LOG_TTS_CHUNKS:
                    emit("log", f"[TTS Playing] {sentence!r}")
                was_interrupted, interrupt_data = play_audio_with_interrupt(audio_data)
                if was_interrupted:
                    interrupt_audio = interrupt_data
                    emit("log", "[TTS Interrupted] Interrupted during playback! Clearing audio queues...")
                    audio_queue.clear_queues()
                    break
            else:
                time.sleep(settings.PLAYBACK_DELAY)

            if (
                not audio_queue.is_running
                and audio_queue.sentence_queue.empty()
                and audio_queue.audio_queue.empty()
            ):
                break

    except Exception as e:
        emit("log", f"Error in audio playback: {str(e)}")

    return was_interrupted, interrupt_audio


def pipeline_main():
    """Runs the voice-loop pipeline in a background thread; reports to the TUI."""
    try:
        emit("status", "INITIALIZING")
        emit("log", "Initializing Whisper model...")
        whisper_processor, whisper_model = init_whisper_model(
            settings.WHISPER_MODEL_ID, settings.WHISPER_MODEL_DIR, hf_token=settings.HUGGINGFACE_TOKEN
        )
        emit("log", "Initializing Voice Activity Detection...")
        vad_pipeline = init_vad_pipeline(settings.HUGGINGFACE_TOKEN)
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
            emit("log", f"Warmup failed: {str(e)}")

        emit("status", "LISTENING")
        emit("log", "=== Ready. Speaking to me (or type below) triggers a response. ===")
        last_activity_time = time.time()

        while not shutdown_event.is_set():
            try:
                text = text_input_queue.get_nowait()
            except queue.Empty:
                text = None

            if text is not None:
                last_activity_time = time.time()
                process_input(session, text, messages, generator, speed)
                last_activity_time = time.time()
                continue

            audio_data = record_continuous_audio(max_wait=1.0)
            if audio_data is not None:
                speech_segments = detect_speech_segments(vad_pipeline, audio_data)

                if speech_segments is not None:
                    last_activity_time = time.time()
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
                            session, user_input, messages, generator, speed
                        )
                        last_activity_time = time.time()
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
                                    )
                                    last_activity_time = time.time()
                                    emit("activity")
                    else:
                        emit("log", "No clear speech detected, please try again.")
                else:
                    emit("log", "No clear speech detected, please try again.")
            else:
                # Check idle condition if no voice input was detected
                idle_elapsed = time.time() - last_activity_time
                if idle_elapsed >= settings.MAX_IDLE_TIME:
                    emit("status", "IDLE")
                    emit("log", f"[Idle Trigger] No activity for {idle_elapsed:.1f}s (MAX_IDLE_TIME={settings.MAX_IDLE_TIME}s).")

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
                    else:
                        idle_prompts = settings.get_idle_prompts_list()
                        prompt_text = random.choice(idle_prompts)
                        emit("log", f"[Random Idle Event] Picked prompt: '{prompt_text}'")

                    process_input(session, prompt_text, messages, generator, speed)
                    last_activity_time = time.time()
                    emit("activity")

                if session is not None:
                    session.headers.update({"Connection": "keep-alive"})
                    if hasattr(session, "connection_pool"):
                        session.connection_pool.clear()

    except Exception as e:
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


class ChatInput(Input):
    """Text input that sends lines to the pipeline thread."""

    BINDINGS = [
        Binding("escape", "unfocus", "Unfocus input"),
        Binding("ctrl+q", "exit_app", "Quit"),
    ]

    def action_unfocus(self):
        try:
            self.screen.set_focus(None)
        except Exception as e:
            emit("error", f"unfocus error: {type(e).__name__}: {e}")

    def action_exit_app(self):
        self.app.exit()

    def on_input_submitted(self, event: Input.Submitted):
        try:
            text = event.value.strip()
            if text:
                text_input_queue.put(text)
                emit("activity")
                emit("transcript", "text", text)
                self.value = ""
        except Exception as e:
            # never raise out of a message handler (Textual would panic the app)
            emit("error", f"text input error: {type(e).__name__}: {e}")


class SpeechTUI(App):
    TITLE = "On-Device Speech-to-Speech AI"

    BINDINGS = [
        Binding("q", "exit_app", "Quit"),
        Binding("t", "focus_input", "Text input"),
    ]

    CSS = """
    #main { height: 1fr; }
    #row1 { height: 2fr; }
    #row2 { height: 2fr; }
    #idle-panel, #twitch-log, #transcript-log, #history-log {
        width: 1fr;
        border: round $accent;
    }
    #log { height: 3fr; border: round $primary; }
    #state-label { height: 1; text-style: bold; padding: 0 1; }
    #idle-count { height: 1; padding: 0 1; }
    #last-activity { height: 1; color: $text-muted; padding: 0 1; }
    #idle-bar { height: 1; margin: 0 1 1 1; }
    #idle-tip { height: 2; color: $text-muted; padding: 0 1; }
    #text-input { height: 3; }
    """

    def __init__(self):
        super().__init__()
        self.status = "STARTING"
        self.last_activity = time.time()
        self.twitch_cache = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="main"):
            with Horizontal(id="row1"):
                with Vertical(id="idle-panel"):
                    yield Label("STATUS / IDLE COUNTDOWN", id="state-label")
                    yield Label(f"[bold]{self.status}[/]", id="status-text")
                    yield ProgressBar(total=settings.MAX_IDLE_TIME, show_eta=False, show_percentage=False, id="idle-bar")
                    yield Label(id="idle-count")
                    yield Label(id="last-activity")
                    yield Label(f"Idle trigger at MAX_IDLE_TIME={settings.MAX_IDLE_TIME:.0f}s of silence", id="idle-tip")
                yield self._rich_log("twitch-log", "Twitch chat (pending)")
            with Horizontal(id="row2"):
                yield self._rich_log("transcript-log", "Transcriptions")
                yield self._rich_log("history-log", "Conversation history")
            yield self._rich_log("log", "Log", id="log", max_lines=2000)
            yield ChatInput(id="text-input", placeholder="Type a message for the bot, Enter to send (Ctrl+Q quits)...")
        yield Footer()

    def _rich_log(self, widget_id, title, id=None, max_lines=200):
        w = RichLog(id=id or widget_id, markup=True, max_lines=max_lines)
        w.border_title = title
        return w

    def on_mount(self):
        self.state_label: Label = self.query_one("#state-label", Label)
        self.status_text: Label = self.query_one("#status-text", Label)
        self.idle_bar: ProgressBar = self.query_one("#idle-bar", ProgressBar)
        self.idle_count: Label = self.query_one("#idle-count", Label)
        self.last_activity_label: Label = self.query_one("#last-activity", Label)
        self.twitch_log: RichLog = self.query_one("#twitch-log", RichLog)
        self.transcript_log: RichLog = self.query_one("#transcript-log", RichLog)
        self.history_log: RichLog = self.query_one("#history-log", RichLog)
        self.log_panel: RichLog = self.query_one("#log", RichLog)

        _install_error_routing()
        self.set_interval(0.1, self._poll_events)
        self.set_interval(0.25, self._update_idle)
        threading.Thread(target=pipeline_main, daemon=True).start()
        self._set_status("INITIALIZING")

    def on_unmount(self):
        shutdown_event.set()
        _restore_error_routing()

    def action_focus_input(self):
        self.query_one("#text-input", Input).focus()

    def action_exit_app(self):
        self.exit()

    def _set_status(self, status: str):
        self.status = status
        colors = {
            "INITIALIZING": "cyan",
            "LISTENING": "green",
            "TRANSCRIBING": "yellow",
            "THINKING": "magenta",
            "SPEAKING": "blue",
            "IDLE": "red",
        }
        color = colors.get(status, "white")
        self.status_text.update(f"[bold {color}]{status}[/]")

    def _touch(self):
        self.last_activity = time.time()

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
            self._log_tui_error("twitch panel", e)

    def _log_tui_error(self, context: str, error: Exception):
        """Write an exception to the log panel; must never raise (a raised
        exception here would make Textual panic and tear down the TUI)."""
        try:
            self.log_panel.write(f"[bold red]{context}: {type(error).__name__}: {error}[/]")
            for line in traceback.format_exc().splitlines():
                self.log_panel.write(f"[dim]{line}[/]")
        except Exception:
            pass

    def _handle(self, kind: str, *payload):
        try:
            if kind == "status":
                self._set_status(payload[0])
                self._touch()
            elif kind == "log":
                line = payload[0]
                if line.startswith("[TTS") or line.startswith("[Twitch"):
                    self.log_panel.write(f"[dim]{line}[/]")
                elif line.startswith("[Idle") or line.startswith("[Random"):
                    self.log_panel.write(f"[yellow]{line}[/]")
                elif "Error" in line or "Traceback" in line or line.startswith("Traceback"):
                    self.log_panel.write(f"[bold red]{line}[/]")
                else:
                    self.log_panel.write(line)
            elif kind == "transcript":
                source, text = payload
                self.transcript_log.write(f"[bold #7fd8a4]You ({source}):[/] {text}")
                self._touch()
            elif kind == "turn":
                user_text, bot_text = payload
                self.history_log.write(f"[bold #ffa657]You:[/] {user_text}")
                self.history_log.write(f"[bold #7fb4ff]Bot:[/] {bot_text}")
                self._touch()
            elif kind == "turn_start":
                self._touch()
            elif kind == "activity":
                self._touch()
            elif kind == "error":
                self.log_panel.write(f"[bold red]{payload[0]}[/]")
        except Exception as e:
            self._log_tui_error("event handler", e)

    def _update_idle(self):
        try:
            if self.status == "LISTENING":
                remaining = max(0.0, settings.MAX_IDLE_TIME - (time.time() - self.last_activity))
                self.idle_count.update(f"[bold]{remaining:5.1f}s[/] until idle trigger")
                self.idle_bar.update(progress=remaining, total=settings.MAX_IDLE_TIME)
            else:
                self.idle_count.update(f"in turn: {self.status}")
                self.idle_bar.update(progress=settings.MAX_IDLE_TIME, total=settings.MAX_IDLE_TIME)
            self.last_activity_label.update(
                f"Last activity: {time.strftime('%H:%M:%S', time.localtime(self.last_activity))}"
            )
        except Exception as e:
            self._log_tui_error("idle panel", e)

    def _update_twitch(self):
        events = twitch_collector.snapshot(max_size=200)
        rendered = [
            f"[{time.strftime('%H:%M:%S', time.localtime(e['timestamp']))}] {e['text']}"
            for e in events
        ]
        key = "\n".join(rendered)
        if key != self.twitch_cache:
            self.twitch_cache = key
            self.twitch_log.clear()
            if rendered:
                for line in rendered:
                    self.twitch_log.write(line)
            else:
                self.twitch_log.write("[dim]No pending chat events[/]")


def main():
    SpeechTUI().run()


if __name__ == "__main__":
    main()
