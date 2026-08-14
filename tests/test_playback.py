"""Minimal unit tests for the adaptive TTS chunking logic.

stdlib unittest, no framework. Run with:  python tests/test_playback.py
(Import is slow: speech_to_speech.py pulls in transformers + textual.)
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import speech_to_speech as s


class TestAdaptivePollDelay(unittest.TestCase):
    def test_poll_delay_fast_when_behind(self):
        self.assertEqual(s._adaptive_poll_delay(0.0), s._POLL_LAZY_DELAY)
        self.assertEqual(s._adaptive_poll_delay(1.0), s.settings.PLAYBACK_DELAY)


class TestTargetSize(unittest.TestCase):
    def setUp(self):
        s._behind_ema = 0.0
        s._smoothed_target = None
        s._ease_rounds_left = 0

    def test_target_for_stays_in_bounds(self):
        full = float(s.settings.TARGET_SIZE)
        self.assertEqual(s._target_for(0.0), full)
        self.assertEqual(s._target_for(0.3), full)
        self.assertEqual(s._target_for(2.5), float(s._TARGET_MIN))
        self.assertLess(s._target_for(1.4), full)
        self.assertGreaterEqual(s._target_for(1.4), float(s._TARGET_MIN))

    def test_shrinks_when_behind_and_eases_back_over_five_rounds(self):
        # Fresh: full size
        self.assertEqual(s._adaptive_target_words(), s.settings.TARGET_SIZE)
        # Fall behind: chunk shrinks (eased, not snapped to the minimum)
        s._behind_ema = 3.0
        low = s._adaptive_target_words()
        self.assertLess(low, s.settings.TARGET_SIZE)
        self.assertGreater(low, s._TARGET_MIN)
        # Caught up: eases back to full size monotonically
        s._behind_ema = 0.0
        eased = [s._adaptive_target_words() for _ in range(6)]
        self.assertEqual(eased, sorted(eased))
        self.assertEqual(eased[-1], s.settings.TARGET_SIZE)

    def test_descent_is_eased(self):
        s._behind_ema = 3.0
        first = s._adaptive_target_words()
        s._behind_ema = 3.0
        second = s._adaptive_target_words()
        self.assertLess(first, s.settings.TARGET_SIZE)
        self.assertLessEqual(second, first)


if __name__ == "__main__":
    unittest.main()
