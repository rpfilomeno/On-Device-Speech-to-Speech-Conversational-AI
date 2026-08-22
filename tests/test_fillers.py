"""Self-check for src/utils/fillers.py (plain asserts, no pytest)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.utils.fillers import pick_filler, _wavs


def main():
    # all three wav sets loaded from recordings/
    for sentiment in ("positive", "negative", "neutral"):
        wavs = _wavs(sentiment)
        assert wavs, f"no wavs found in recordings/{sentiment}"
        assert all(isinstance(w, np.ndarray) and len(w) > 0 for w in wavs)

    # sentiment routing
    assert pick_filler("I love this, it's wonderful!") is not None
    assert pick_filler("I hate this, it's awful.") is not None
    assert pick_filler("The table has four legs.") is not None
    assert pick_filler("") is None
    assert pick_filler("   ") is None

    print("test_fillers: all assertions passed")


if __name__ == "__main__":
    main()
