"""Adaptive TTS playback pacing: words-per-chunk target, poll delay, and the
runtime jitter learners that persist tuned values to settings.json.

All mutable state lives in this module; callers that must write to it (e.g.
audio_playback_worker's _behind_ema) should reference it as `pacing._behind_ema`
so they mutate this module's globals rather than a local copy.
"""

from src.utils.config import settings

# Adaptive chunking: when TTS synthesis falls behind the player, shrink the
# words-per-chunk target so sentences synthesize faster (fewer words = shorter
# audio = less time behind), and poll fast so the next chunk is grabbed the
# instant it's ready. The chunk size eases the same way the old playback-speed
# recovery did: smooth descent, cubic ease-out back to full size.
_TARGET_MIN = 6
_TARGET_RAMP_START = 0.3
_TARGET_RAMP_END = 2.5
_BEHIND_EMA_ALPHA = 0.3
_BEHIND_DECAY = 0.8
_POLL_LAZY_DELAY = 0.05
_POLL_BEHIND_THRESHOLD = 0.5

# Persistent across turns (until app exit): a turn that starts slow already
# carries the previous adaptation instead of re-learning the first gap.
_behind_ema = 0.0
_EASE_ROUNDS = 5
_smoothed_target: float | None = None
_ease_rounds_left = 0
_ease_start_value = 0.0


def target_for(behind_ema: float) -> float:
    """Raw words-per-chunk target for a given behind-average."""
    full = float(settings.TARGET_SIZE)
    if behind_ema <= _TARGET_RAMP_START:
        return full
    if behind_ema >= _TARGET_RAMP_END:
        return float(_TARGET_MIN)
    t = (behind_ema - _TARGET_RAMP_START) / (_TARGET_RAMP_END - _TARGET_RAMP_START)
    return full - t * (full - _TARGET_MIN)


def _ease_to(current: float, target: float) -> float:
    """Ease `current` toward `target`: smooth descent, 5-round cubic ease-out
    on recovery (the scheme the playback-speed recovery used)."""
    global _ease_rounds_left, _ease_start_value
    if abs(target - current) < 0.05:
        _ease_rounds_left = 0
        return target
    if target < current:
        _ease_rounds_left = 0
        return current + 0.5 * (target - current)
    if _ease_rounds_left == 0:
        _ease_start_value = current
        _ease_rounds_left = _EASE_ROUNDS
    _ease_rounds_left = max(0, _ease_rounds_left - 1)
    progress = 1.0 - _ease_rounds_left / _EASE_ROUNDS
    eased = 1.0 - (1.0 - progress) ** 3
    return _ease_start_value + (target - _ease_start_value) * eased


def adaptive_target_words() -> int:
    """Current eased words-per-chunk target, driven by the behind-average."""
    global _smoothed_target
    target = target_for(_behind_ema)
    if _smoothed_target is None:
        _smoothed_target = target
    else:
        _smoothed_target = _ease_to(_smoothed_target, target)
    return int(round(_smoothed_target))


def adaptive_poll_delay(behind_ema: float) -> float:
    """Poll interval: minimal when behind (grab audio ASAP), lazy when ahead."""
    if behind_ema > _POLL_BEHIND_THRESHOLD:
        return settings.PLAYBACK_DELAY
    return _POLL_LAZY_DELAY


# Runtime jitter learner: bucket per-turn jitter (inter-chunk gap stddev) by the
# words-per-chunk target in effect, then persist the best-scoring target size to
# settings.json so future sessions start already-tuned.
_jitter_samples: dict[int, list[float]] = {}
_MIN_JITTER_SAMPLES = 3


def best_target_size() -> int | None:
    """Largest TARGET_SIZE whose mean jitter stays within JITTER_MARGIN_MS of the
    best (lowest) jitter. Keeps chunk size high while latency is acceptable;
    only a size with meaningfully worse jitter is rejected. Falls back to the
    lowest-jitter bucket when only it qualifies."""
    eligible = [
        (size, sum(samples) / len(samples))
        for size, samples in _jitter_samples.items()
        if len(samples) >= _MIN_JITTER_SAMPLES
    ]
    if not eligible:
        return None
    best_jitter = min(m for _, m in eligible)
    ceiling = best_jitter + settings.JITTER_MARGIN_MS
    acceptable = [size for size, m in eligible if m <= ceiling]
    return max(acceptable) if acceptable else min(eligible)[0]


def record_jitter(target_size: int, jitter_ms: float) -> int | None:
    """Feed one turn's jitter into the learner; returns the current best target size."""
    samples = _jitter_samples.setdefault(target_size, [])
    samples.append(jitter_ms)
    del samples[:-30]
    return best_target_size()


# PLAYBACK_DELAY tuning: round-robin one candidate per turn until every candidate
# has enough jitter samples, then settle on the lowest-mean one and persist it.
_DELAY_CANDIDATES = [0.005, 0.01, 0.02, 0.05]
_delay_samples: dict[float, list[float]] = {}
_delay_explore_index = 0
_delay_settled = False


def record_delay_jitter(jitter_ms: float) -> float | None:
    """Record jitter under the current PLAYBACK_DELAY and advance to the next
    candidate. Once every candidate has enough samples, returns the best
    (lowest mean jitter) delay to adopt, else None."""
    global _delay_explore_index, _delay_settled
    if _delay_settled:
        return None
    if settings.PLAYBACK_DELAY not in _DELAY_CANDIDATES:
        settings.PLAYBACK_DELAY = _DELAY_CANDIDATES[0]
    samples = _delay_samples.setdefault(settings.PLAYBACK_DELAY, [])
    samples.append(jitter_ms)
    del samples[:-30]
    if len(_delay_samples) == len(_DELAY_CANDIDATES) and all(
        len(v) >= _MIN_JITTER_SAMPLES for v in _delay_samples.values()
    ):
        _delay_settled = True
        return min(
            _delay_samples, key=lambda k: sum(_delay_samples[k]) / len(_delay_samples[k])
        )
    _delay_explore_index += 1
    settings.PLAYBACK_DELAY = _DELAY_CANDIDATES[
        _delay_explore_index % len(_DELAY_CANDIDATES)
    ]
    return None
