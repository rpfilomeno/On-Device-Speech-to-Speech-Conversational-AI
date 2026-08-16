"""Runnable check for the TTS target-size learner's balance rule: prefer the
largest TARGET_SIZE whose jitter stays within JITTER_MARGIN_MS of the best.

Run: uv run python tests/test_pacing_choice.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.text_to_speech import pacing
from src.text_to_speech.state import settings

settings.JITTER_MARGIN_MS = 100.0


def buckets(data):
    pacing._jitter_samples = {size: [m] * pacing._MIN_JITTER_SAMPLES for size, m in data.items()}


# No bucket with enough samples -> None
pacing._jitter_samples = {6: [100.0]}
assert pacing.best_target_size() is None, "not enough samples -> None"

# Larger size within margin of the best jitter -> preferred (keeps TARGET_SIZE high)
buckets({6: 100.0, 10: 160.0})   # best 100, ceiling 200
assert pacing.best_target_size() == 10, "within-margin larger size wins"

# Size with meaningfully worse jitter -> rejected (latency protected)
buckets({6: 100.0, 10: 350.0})   # 10 above ceiling 200
assert pacing.best_target_size() == 6, "worse-jitter size rejected"

# Largest within margin of best, not the best itself
buckets({6: 120.0, 10: 100.0, 15: 190.0})   # best 100, ceiling 200
assert pacing.best_target_size() == 15, "largest acceptable size wins"

# Ties across the whole range -> largest wins
buckets({6: 150.0, 12: 150.0, 18: 150.0})
assert pacing.best_target_size() == 18, "equal jitter -> largest"

print("target-size balance rule: OK")
