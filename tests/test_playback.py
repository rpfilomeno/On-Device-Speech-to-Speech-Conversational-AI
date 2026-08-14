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


class TestJitterLearner(unittest.TestCase):
    def setUp(self):
        s._jitter_samples = {}

    def test_picks_lowest_mean_jitter_bucket(self):
        self.assertIsNone(s._record_jitter(10, 100.0))
        self.assertIsNone(s._record_jitter(10, 120.0))
        self.assertEqual(s._record_jitter(10, 110.0), 10)
        self.assertEqual(s._record_jitter(6, 40.0), 10)
        self.assertEqual(s._record_jitter(6, 60.0), 10)
        self.assertEqual(s._record_jitter(6, 50.0), 6)
        self.assertEqual(s._best_target_size(), 6)

    def test_needs_min_samples_per_bucket(self):
        s._record_jitter(6, 10.0)
        s._record_jitter(10, 200.0)
        self.assertIsNone(s._best_target_size())
        s._record_jitter(10, 210.0)
        s._record_jitter(10, 220.0)
        self.assertEqual(s._best_target_size(), 10)


class TestDelayLearner(unittest.TestCase):
    def setUp(self):
        s._delay_samples = {}
        s._delay_explore_index = 0
        s._delay_settled = False
        self._orig_delay = s.settings.PLAYBACK_DELAY

    def tearDown(self):
        s.settings.PLAYBACK_DELAY = self._orig_delay

    def test_explores_then_settles_on_lowest_mean_jitter(self):
        n = len(s._DELAY_CANDIDATES)
        # two full cycles: every candidate has 2 samples, still exploring
        for _ in range(2 * n):
            self.assertIsNone(s._record_delay_jitter(100.0))
        self.assertFalse(s._delay_settled)
        # third cycle: candidates 0..2 stay at 100, candidate 3 drops to 10
        for _ in range(3):
            self.assertIsNone(s._record_delay_jitter(100.0))
        self.assertEqual(s._record_delay_jitter(10.0), s._DELAY_CANDIDATES[3])
        self.assertTrue(s._delay_settled)

    def test_settled_returns_none_and_stops_advancing(self):
        self.assertEqual(s._delay_settled, False)
        s._delay_settled = True
        delay = s.settings.PLAYBACK_DELAY
        self.assertIsNone(s._record_delay_jitter(50.0))
        self.assertEqual(s.settings.PLAYBACK_DELAY, delay)


if __name__ == "__main__":
    unittest.main()
