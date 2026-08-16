"""Runnable check for the greedy TextChunker: cut at the furthest natural break
(punctuation or semantic connector) at or before the target size, so chunks fill
up to TARGET_SIZE words.

Plain asserts, no framework. Run:  uv run python tests/test_text_chunker.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.text_chunker import TextChunker


def words(n):
    return [f"w{i}" for i in range(n)]


c = TextChunker()

# Shorter than target -> whole text
assert c.find_break_point(["a", "b"], 20) == 2

# No natural break in the window -> hard cut at exactly target_size
assert c.find_break_point(words(25), 20) == 20

# Greedy: the furthest natural break wins, not an earlier higher-priority one
# (w3 has a period — priority 5 — but w18 has a comma, so 19 words get cut).
t = words(25)
t[3] = "stop."
t[18] = "pause,"
assert c.find_break_point(t, 20) == 19, c.find_break_point(t, 20)

# Semantic connectors are natural breaks too
t = words(25)
t[15] = "and"
assert c.find_break_point(t, 20) == 16

# A break just past the target does not overshoot; chunk stays at target_size
t = words(24)
t[21] = "done."
assert c.find_break_point(t, 20) == 20

print("test_text_chunker: OK")
