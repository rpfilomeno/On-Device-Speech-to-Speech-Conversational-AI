"""Minimal runnable check for barge-in debounce + noise-floor logic.

Run: uv run python tests/test_barge_in.py
"""
import sys
from pathlib import Path
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.utils.speech as speech
from src.utils.speech import TurnAudioPlayer

speech.settings.SPEECH_CHECK_TIMEOUT = 0.1


def make_player():
    p = object.__new__(TurnAudioPlayer)
    p._playing = True
    p._roll = []
    p._capture = None
    p._interrupt = threading.Event()
    p._last_hot = None
    p._noise_floor = 0.0
    p._last_speech_time = 0.0
    p._stop_events = []
    return p


def block(level):
    return np.full((1024, 1), level, dtype=np.float32)


def feed(p, levels, step=0.02):
    clock = {"t": 0.0}

    def _now():
        return clock["t"]

    speech.time.perf_counter = _now
    for lvl in levels:
        p._input_callback(block(lvl), 1024, None, None)
        clock["t"] += step
    speech.time.perf_counter = time.perf_counter


quiet = 0.001  # below fixed threshold 0.005
loud = 0.05    # well above any trigger derived from a quiet floor


def main():
    # Transient: one loud block then silence must never barge.
    p = make_player()
    feed(p, [quiet] * 10 + [loud, quiet])
    assert not p.is_interrupted(), "transient spike barged in"

    # Sustained: loud blocks persisting >= SPEECH_CHECK_TIMEOUT must barge.
    p = make_player()
    feed(p, [quiet] * 10 + [loud] * 6)  # 6 * 0.02s = 0.12s > default 0.1
    assert p.is_interrupted(), "sustained speech did not barge"

    # Quiet-only: must never barge even with hot first block ambiguity.
    p = make_player()
    feed(p, [quiet] * 20)
    assert not p.is_interrupted(), "quiet noise barged in"

    print("barge-in debounce/noise-floor: OK")


if __name__ == "__main__":
    main()
