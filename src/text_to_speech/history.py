"""Chat-history management: token budgeting, proactive trimming, and the
deterministic retry-path fallback."""

from src.utils.config import settings

from . import state

_TRIM_GRACE_RETRIES = 2  # transient failures don't shrink context; trim only after this many consecutive retries
_MAX_LLM_RETRIES = 5  # after this many blank responses, start a new session
_ANCHOR_TURNS = 1  # keep the first completed turn as a topic anchor when trimming


def trim_history(messages: list) -> bool:
    """Deterministic fallback for the retry path: drop half the droppable middle
    turns (after the anchor), keeping the system prompt, the anchor turn, the
    recent turns, and the current user message. Returns True if anything was
    dropped."""
    droppable = (len(messages) - 2) // 2 - _ANCHOR_TURNS
    if droppable < 1:
        return False
    n_drop = max(1, droppable // 2)
    first = 1 + 2 * _ANCHOR_TURNS
    del messages[first:first + n_drop * 2]
    return True


def retry_llm(messages: list, retry: int, reason: str) -> bool:
    """Retry a failed LLM call. The first few failures (server busy, model
    loading) retry as-is and let the history grow; only later failures start
    trimming older turns so the context can actually shrink. Returns True if a
    retry is worth another attempt."""
    if retry >= _TRIM_GRACE_RETRIES:
        if not trim_history(messages):
            return False
        state.emit("log", f"{reason} - trimmed history and retrying.")
    else:
        state.emit("log", f"{reason} - retrying without trimming history.")
    return True


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token); no tokenizer is available offline."""
    return max(1, len(text) // 4)


def trim_history_to_budget(messages: list) -> int:
    """Keep the system prompt, the first (anchor) turn, and the most recent turns,
    dropping the oldest middle turns until the prompt fits within
    CONTEXT_WINDOW * CONTEXT_TRIM_RATIO tokens. The current user message
    (messages[-1]) is always kept. Returns turns dropped (0 = nothing trimmed)."""
    budget = int(settings.CONTEXT_WINDOW * settings.CONTEXT_TRIM_RATIO)
    tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
    dropped = 0
    first_droppable = 1 + 2 * _ANCHOR_TURNS
    max_droppable_pairs = (len(messages) - 2) // 2 - _ANCHOR_TURNS
    while tokens > budget and dropped < max_droppable_pairs:
        i = first_droppable + dropped * 2
        tokens -= estimate_tokens(messages[i].get("content", ""))
        tokens -= estimate_tokens(messages[i + 1].get("content", ""))
        dropped += 1
    if dropped:
        del messages[first_droppable:first_droppable + dropped * 2]
    return dropped
