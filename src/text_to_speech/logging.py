"""Error routing: redirect prints, logging, and unhandled exceptions into the
TUI's event queue so nothing is written over the Textual screen."""

import logging
import sys
import threading
import traceback

from . import state


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
                    state.emit("log", line)
        except Exception:
            pass

    def flush(self):
        try:
            if self._buf.strip():
                state.emit("log", self._buf.strip())
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
                state.emit("log", msg)
        except Exception:
            pass


def _excepthook(exc_type, exc_value, exc_tb):
    try:
        state.emit("error", f"Unhandled {exc_type.__name__}: {exc_value}")
        for line in traceback.format_exception(exc_type, exc_value, exc_tb):
            state.emit("log", line.rstrip())
    except Exception:
        pass


def _thread_excepthook(args):
    try:
        state.emit("error", f"Unhandled {args.exc_type.__name__} in thread '{args.thread.name}': {args.exc_value}")
        for line in traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback):
            state.emit("log", line.rstrip())
    except Exception:
        pass


_orig_excepthook = sys.excepthook
_orig_thread_excepthook = threading.excepthook


def install_error_routing():
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


def restore_error_routing():
    if isinstance(sys.stdout, _LogStream):
        sys.stdout = sys.__stdout__
    if isinstance(sys.stderr, _LogStream):
        sys.stderr = sys.__stderr__
    sys.excepthook = _orig_excepthook
    threading.excepthook = _orig_thread_excepthook
