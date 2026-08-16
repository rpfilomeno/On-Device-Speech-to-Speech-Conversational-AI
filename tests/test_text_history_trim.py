"""Smoke check that the split src.text_to_speech package preserved the
context-budget trimming behavior.

Run: uv run python tests/test_text_history_trim.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.text_to_speech import history
from src.text_to_speech.state import settings

settings.CONTEXT_WINDOW = 2048
settings.CONTEXT_TRIM_RATIO = 0.8
budget = int(2048 * 0.8)

assert history.estimate_tokens("x" * 400) == 100


def msgs(pairs, extra=""):
    m = [{"role": "system", "content": "sys"}]
    for i in range(pairs):
        m.append({"role": "user", "content": f"user{i} " + "a" * 500})
        m.append({"role": "assistant", "content": f"asst{i} " + "b" * 500})
    m.append({"role": "user", "content": extra or "current question"})
    return m


def tok(m):
    return sum(history.estimate_tokens(x["content"]) for x in m)


m = msgs(2)
assert history.trim_history_to_budget(m) == 0 and len(m) == 6

m = msgs(30)
dropped = history.trim_history_to_budget(m)
assert dropped > 0
assert m[0]["role"] == "system"
assert m[1]["content"].startswith("user0")
assert "current" in m[-1]["content"]
assert tok(m) <= budget

m = msgs(30)
assert history.trim_history(m), "retry-path fallback drops something"
assert m[1]["content"].startswith("user0"), "anchor kept"
assert m[-1]["role"] == "user"

print("text package history trim: OK")
