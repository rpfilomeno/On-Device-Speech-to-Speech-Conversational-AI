"""Self-check for the /idle and /twitch prompt-source toggles.

The idle countdown runs while either toggle is on; it only stops when both
are off. Plain asserts, no framework. Run with:  python tests/test_twitch_toggle.py
(Import is slow: speech_to_speech.py pulls in transformers + textual.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import speech_to_speech as s


def reset(idle=True, twitch=True):
    import time

    s.idle_enabled = idle
    s.twitch_enabled = twitch
    s.idle_mode = False
    s.now_event.clear()
    s.twitch_collector.clear_all()
    # Expire the countdown so every case tests an armed trigger.
    s.pipeline_last_activity = time.time() - 9999.0


def trigger_fires():
    """Mirror of the pipeline gate: (idle_enabled or twitch_enabled) and (...)"""
    return (s.idle_enabled or s.twitch_enabled) and (
        s.idle_mode or s._idle_elapsed() >= 9999.0 or s.now_event.is_set()
    )


def main():
    # Defaults: both on, countdown armed
    reset()
    assert s.idle_enabled is True
    assert s.twitch_enabled is True

    # /idle off alone does NOT stop the countdown — twitch still triggers it
    reset()
    s.idle_enabled = False
    assert trigger_fires(), "countdown must keep running while twitch prompts are on"

    # /twitch off alone does NOT stop the countdown — idle still triggers it
    reset()
    s.twitch_enabled = False
    assert trigger_fires(), "countdown must keep running while idle prompts are on"

    # Both off: the only state where the countdown stops
    reset()
    s.idle_enabled = False
    s.twitch_enabled = False
    assert not trigger_fires(), "both toggles off must stop the countdown"

    # Prompt selection priority: twitch chats first, then idle prompts,
    # then silent skip
    reset()
    s.twitch_collector.add_event("[Chat] bob: hello")
    events = s.twitch_collector.get_recent_events(max_size=50, max_age=180)
    assert events == ["[Chat] bob: hello"], events

    reset(idle=False)
    s.twitch_collector.add_event("[Chat] bob: hi")
    assert s.twitch_collector.has_recent_events(50, 180)
    # twitch-only with empty queue -> skip path (no idle prompt fallback)
    s.twitch_collector.clear_all()
    assert not s.twitch_collector.has_recent_events(50, 180)

    # /twitch off drains the buffer so stale chats never resurface
    reset()
    s.twitch_collector.add_event("[Chat] bob: stale")
    cleared = s.twitch_collector.clear_all()
    assert cleared == 1
    assert len(s.twitch_collector.snapshot()) == 0

    print("test_twitch_toggle: OK")


if __name__ == "__main__":
    main()
