"""Runnable check for the context-budget history trimming.

Run: uv run python tests/test_history_trim.py
(Import is slow: speech_to_speech.py pulls in transformers + textual.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import speech_to_speech as s

# _estimate_tokens: ~4 chars/token, never below 1
assert s._estimate_tokens("") == 1, "empty text -> 1 token"
assert s._estimate_tokens("abcd") == 1, "4 chars -> 1 token"
assert s._estimate_tokens("x" * 400) == 100, "400 chars -> ~100 tokens"


def msgs(pairs, extra=""):
    m = [{"role": "system", "content": "sys"}]
    for i in range(pairs):
        m.append({"role": "user", "content": f"user{i} " + "a" * 500})
        m.append({"role": "assistant", "content": f"asst{i} " + "b" * 500})
    m.append({"role": "user", "content": extra or "current question"})
    return m


def tok(m):
    return sum(s._estimate_tokens(x["content"]) for x in m)


def roles(m):
    return [x["role"] for x in m]


s.settings.CONTEXT_WINDOW = 2048
s.settings.CONTEXT_TRIM_RATIO = 0.8
budget = int(2048 * 0.8)

# Small history, well under budget -> no-op
m = msgs(2)
assert s._trim_history_to_budget(m) == 0, "under budget -> nothing trimmed"
assert len(m) == 6, "under budget -> messages untouched"

# Huge history -> drops oldest MIDDLE pairs, keeps anchor + recent + current
m = msgs(30)
assert len(m) == 62
dropped = s._trim_history_to_budget(m)
assert dropped > 0, "over budget -> must trim"
assert m[0]["role"] == "system", "system prompt kept"
assert m[1]["content"].startswith("user0"), "first turn kept as anchor"
assert m[2]["content"].startswith("asst0"), "anchor assistant kept"
assert m[-1]["role"] == "user" and "current" in m[-1]["content"], "current user message kept"
assert tok(m) <= budget, f"prompt {tok(m)} over budget after trim"
assert len(m) == 2 + 2 * (30 - dropped), "exactly the dropped pairs were removed"
# after the anchor, remaining middle messages are the most recent turns (no gaps)
kept_users = [x["content"].split()[0] for x in m if x["role"] == "user"]
assert kept_users[0] == "user0" and kept_users[-1] == "current", "anchor + current at ends"
middle = kept_users[1:-1]
assert middle == [f"user{i}" for i in range(1 + dropped, 30)], \
    f"recent turns preserved contiguously: {middle}"

# Reactive _trim_history: deterministic half-drop of the middle, anchor intact
m = msgs(30)
assert s._trim_history(m), "retry-path trim should drop something"
assert m[0]["role"] == "system", "system prompt kept after retry trim"
assert m[1]["content"].startswith("user0"), "anchor kept after retry trim"
assert m[-1]["role"] == "user", "current user message kept after retry trim"
assert roles(m) == roles(m)[:1] + ["user", "assistant"] * ((len(m) - 2) // 2) + ["user"], \
    "roles strictly alternate, no orphaning"

# Never drops below system + anchor + current even when hopelessly over budget
m = msgs(0, extra="z" * 5000)
assert s._trim_history_to_budget(m) == 0, "single turn cannot be trimmed"
assert len(m) == 2
m = msgs(1, extra="z" * 5000)
assert s._trim_history_to_budget(m) == 0, "anchor + current only -> nothing droppable"
assert len(m) == 4

print("history trim to budget: OK")
