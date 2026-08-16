"""Textual TUI for the text-to-speech app: chat window, input, slash commands,
idle/twitch panels, and the streaming bot-line renderer."""

import queue
import re
import threading
import time
import traceback
from typing import cast

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, ProgressBar, RichLog, Static

from src.utils import twitch_collector
from src.utils.config import log_error, settings

from . import logging as err_logging
from . import pipeline
from . import state

# Slash commands: name -> (handler method, description). Drives both the
# completion menu and the /help panel.
SLASH_COMMANDS = [
    ("/quit", "Quit the application (asks for confirmation)"),
    ("/clear", "Clear the chat display"),
    ("/stop", "Interrupt the current playback / response"),
    ("/now", "Trigger the idle event immediately"),
    ("/idle", "Enter idle mode now; /idle off disables+resets the countdown, /idle on re-enables it"),
    ("/pause", "Suspend voice output and the idle countdown"),
    ("/play", "Resume from pause"),
    ("/slap", "Erase queued Twitch messages, or /slap @user for just theirs"),
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
            state.emit("error", f"unfocus error: {type(e).__name__}: {e}")

    def on_input_changed(self, event: Input.Changed):
        state.last_typing_activity = time.time()
        if state.idle_mode:
            state.interrupt_event.set()
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
                cast("TextSpeechTUI", self.app)._run_command(
                    self._suggestion_matches[self._suggestion_index][0]
                )
                self.value = ""
                self._hide_suggestions()
                return
            text = event.value.strip()
            if text.startswith("/"):
                cast("TextSpeechTUI", self.app)._run_command(text)
                self.value = ""
                self._hide_suggestions()
                return
            if text:
                state.text_input_queue.put(text)
                state.interrupt_event.set()
                state.emit("activity")
                state.emit("transcript", "text", text)
                self.value = ""
        except Exception as e:
            # never raise out of a message handler (Textual would panic the app)
            log_error(e)
            state.emit("error", f"text input error: {type(e).__name__}: {e}")

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


class TextSpeechTUI(App):
    TITLE = "On-Device Text-to-Speech AI"

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
    #llm-rate { color: $text-muted; height: 1; }
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
                yield Static("On-Device Text-to-Speech AI", classes="menu-title")
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
                        yield Label(id="llm-rate")
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
        self.llm_rate_label: Label = self.query_one("#llm-rate", Label)

        err_logging.install_error_routing()
        self.set_interval(0.1, self._poll_events)
        self.set_interval(0.25, self._update_status)
        threading.Thread(target=pipeline.pipeline_main, daemon=True).start()
        self._set_status("INITIALIZING")

    def on_unmount(self):
        state.shutdown_event.set()
        err_logging.restore_error_routing()

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
        state.interrupt_event.set()
        self._bot_reply("Stopped. I've silenced my output.")

    def _cmd_now(self, arg=""):
        state.now_event.set()
        self._bot_reply("Idle countdown set to zero — I'll start talking now.")

    def _cmd_idle(self, arg=""):
        arg = arg.strip().lower()
        if arg == "off":
            state.idle_enabled = False
            state.pipeline_last_activity = time.time()
            state.idle_mode = False
            state.now_event.clear()
            state.emit("status", "LISTENING")
            self._bot_reply("Idle countdown disabled and reset. Type /idle on to re-enable.")
        elif arg == "on":
            state.idle_enabled = True
            state.pipeline_last_activity = time.time()
            state.now_event.clear()
            self._bot_reply("Idle countdown enabled.")
        else:
            state.now_event.set()
            self._bot_reply("Entering idle mode — I'll start talking now.")

    def _cmd_pause(self, arg=""):
        state.pause_event.set()
        self._set_status("PAUSED")
        self._bot_reply("Paused. My voice and the idle countdown are suspended. Type /play to resume.")

    def _cmd_play(self, arg=""):
        if state.pause_event.is_set():
            state.pause_event.clear()
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

    def _cmd_memory(self, arg=""):
        arg = arg.strip().lower()
        if arg == "on":
            if state.memory_status["enabled"]:
                self._bot_reply(f"Qdrant long-term memory is already ON ({state.memory_status['backend']}).")
            else:
                state.memory_request_queue.put("on")
                self._bot_reply("Switching to Qdrant long-term memory...")
        elif arg == "off":
            if not state.memory_status["enabled"]:
                self._bot_reply("Long-term memory is already OFF.")
            else:
                state.memory_request_queue.put("off")
                self._bot_reply("Switching off long-term memory...")
        else:
            if state.memory_status["enabled"]:
                self._bot_reply(f"Qdrant long-term memory is ON ({state.memory_status['backend']}).")
            else:
                self._bot_reply("Long-term memory is OFF.")

    def _cmd_new(self, arg=""):
        state.new_chat_event.set()
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
        t = state.timing_info
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
                if state.idle_enabled:
                    remaining = max(0.0, settings.MAX_IDLE_TIME - state._idle_elapsed())
                    mid = f"idle [bold]{remaining:5.1f}s[/]"
                else:
                    mid = "idle [bold]off[/]"
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
                if state.idle_enabled:
                    remaining = max(0.0, settings.MAX_IDLE_TIME - state._idle_elapsed())
                    self.idle_count.update(f"idle in [bold]{remaining:5.1f}s[/]")
                    self.idle_bar.update(progress=remaining, total=settings.MAX_IDLE_TIME)
                else:
                    self.idle_count.update("idle [bold]off[/]")
                    self.idle_bar.update(progress=0, total=settings.MAX_IDLE_TIME)
            else:
                self.idle_count.update(f"in turn: {self.status}")
                self.idle_bar.update(progress=settings.MAX_IDLE_TIME, total=settings.MAX_IDLE_TIME)
            self.last_activity_label.update(
                f"Last activity: {state._time_ago(state.pipeline_last_activity)}"
            )
            self.llm_rate_label.update(f"LLM: [bold]{state.llm_tokens_per_sec():.1f}[/] tok/s")
        except Exception as e:
            log_error(e)
            self._log_tui_error("idle pane", e)

    # ---- event handling ----

    def _poll_events(self):
        try:
            while True:
                kind, *payload = state._event_queue.get_nowait()
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
                if source != "twitch":
                    state._append_chat_file("in.txt", text)
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
