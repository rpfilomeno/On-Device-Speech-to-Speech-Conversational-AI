"""Self-check for the idle-mode / typing-pause countdown logic.

Plain asserts, no framework. Run with:  python tests/test_idle.py
(Import is slow: speech_to_speech.py pulls in transformers + textual.)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import speech_to_speech as s


def set_state(last_activity, typing_start, last_typing):
    s.pipeline_last_activity = last_activity
    s.typing_pause_start = typing_start
    s.last_typing_activity = last_typing


def main():
    now = time.time()
    T = s._TYPING_PAUSE_SECONDS

    # No typing ever: plain elapsed clock
    set_state(now - 30, None, 0.0)
    assert abs(s._idle_elapsed() - 30) < 0.01, s._idle_elapsed()

    # Typing active: countdown frozen at its pause-start value (28s had elapsed)
    set_state(now - 30, now - 2, now - 1)
    assert abs(s._idle_elapsed() - 28) < 0.01, s._idle_elapsed()

    # Window closed but pipeline hasn't shifted yet: raw clock (transient, fixed next loop)
    set_state(now - 30, now - 7, now - 6)
    assert abs(s._idle_elapsed() - 30) < 0.01, s._idle_elapsed()

    # Pipeline shift is what the loop does when the window closes: 30 - 7 = 23s
    set_state(now - 30, now - 7, now - 6)
    assert s.typing_pause_start is not None
    paused = s.typing_pause_start
    s.pipeline_last_activity += now - paused
    s.typing_pause_start = None
    assert abs(s._idle_elapsed() - 23) < 0.01, s._idle_elapsed()

    # Adaptive playback: speed stays in [0.92, 1.0] and only reacts to real gaps
    assert s._adaptive_speed(0.0) == 1.0
    assert s._adaptive_speed(0.3) == 1.0
    mid = s._adaptive_speed(1.4)
    assert s._PLAY_SPEED_MIN < mid < 1.0
    assert s._adaptive_speed(2.5) == s._PLAY_SPEED_MIN
    assert s._adaptive_speed(10.0) == s._PLAY_SPEED_MIN

    # Poll delay: minimal once behind, lazy otherwise
    assert s._adaptive_poll_delay(0.0) == s._POLL_LAZY_DELAY
    assert s._adaptive_poll_delay(1.0) == s.settings.PLAYBACK_DELAY

    # Recovery easing: follows target while behind, then eases to 1.0 over 5 rounds
    s._smoothed_speed = 1.0
    s._ease_rounds_left = 0
    assert s._recovery_speed(0.93) == 0.93
    assert s._ease_rounds_left == 0
    eased = [s._recovery_speed(1.0) for _ in range(5)]
    assert eased[0] > 0.93 and eased[0] < 1.0, eased[0]
    assert eased == sorted(eased), "easing must move monotonically toward 1.0"
    assert eased[-1] == 1.0
    assert s._recovery_speed(1.0) == 1.0

    print("test_idle: OK")


if __name__ == "__main__":
    main()
