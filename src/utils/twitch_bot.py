import threading
import time
from collections import deque
import twitch
from src.utils.config import settings


class TwitchEventCollector:
    """Thread-safe event collector for Twitch chat messages."""

    def __init__(self):
        self._lock = threading.Lock()
        self._events = deque()

    def add_event(self, event_str: str):
        with self._lock:
            self._events.append({
                "timestamp": time.time(),
                "text": event_str
            })
        if settings.LOG_TWITCH_CHATS:
            print(f"\n[Twitch Event Collected] {event_str}")

    def get_recent_events(self, max_size: int, max_age: float) -> list[str]:
        """Returns collected events not older than max_age seconds, up to max_size count."""
        now = time.time()
        with self._lock:
            # Filter by age
            valid_events = [e for e in self._events if (now - e["timestamp"]) <= max_age]
            # Take up to max_size most recent events
            recent = valid_events[-max_size:]
            # Clear returned events from buffer to prevent reprocessing
            self._events = deque([e for e in valid_events if e not in recent])
            return [e["text"] for e in recent]

    def has_recent_events(self, max_size: int, max_age: float) -> bool:
        now = time.time()
        with self._lock:
            valid_events = [e for e in self._events if (now - e["timestamp"]) <= max_age]
            return len(valid_events) > 0


class TwitchBotManager:
    """Manages the twitch-python chat bot background thread."""

    def __init__(self, collector: TwitchEventCollector):
        self.collector = collector
        self.thread = None
        self.chat = None

    def start(self, channel: str):
        if not channel:
            print("[Twitch Bot] No TWITCH_CLIENT_CHANNEL set. Twitch collector disabled.")
            return

        # Sanitize channel name (remove quotes/whitespace if present in .env)
        channel = channel.strip("'\" ")
        if not channel:
            print("[Twitch Bot] Channel name empty after sanitization.")
            return

        self.thread = threading.Thread(target=self._run_bot, args=(channel,), daemon=True)
        self.thread.start()

    def _run_bot(self, channel: str):
        try:
            # Anonymous Twitch IRC connection
            self.chat = twitch.Chat(
                channel=channel,
                nickname="justinfan12345",
                oauth="SCHMOOPIIE",
            )
        except Exception as e:
            print(f"[Twitch Bot Error] {e}")
            return

        def on_message(message):
            text = message.text if hasattr(message, "text") else str(message)
            self.collector.add_event(f"[Chat] {message.sender}: {text}")

        self.chat.subscribe(on_message)


twitch_collector = TwitchEventCollector()
twitch_bot_manager = TwitchBotManager(twitch_collector)
