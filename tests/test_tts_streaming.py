# Self-check: streamed TTS segments must be enqueued one-by-one (not buffered
# into a single blob), so playback can start before synthesis finishes.
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.utils.audio_queue import AudioGenerationQueue


class FakeStreamingGenerator:
    streaming = True

    def generate(self, text, speed=1.0, stream=True, **kwargs):
        for i in range(3):
            time.sleep(0.01)
            yield np.ones(16, dtype=np.float32) * i


def main():
    q = AudioGenerationQueue(FakeStreamingGenerator())
    q.start()
    q.add_sentences(["hello there"])
    deadline = time.time() + 2
    items = []
    while len(items) < 3 and time.time() < deadline:
        audio, sentence, is_first = q.get_next_audio()
        if audio is not None:
            items.append((audio, sentence, is_first))
        else:
            time.sleep(0.005)

    q.stop()

    assert len(items) == 3, f"expected 3 segments, got {len(items)}"
    assert items[0][2] is True, "first segment must be marked is_first"
    assert all(not it[2] for it in items[1:]), "later segments must not be is_first"
    assert [float(it[0][0]) for it in items] == [0.0, 1.0, 2.0], (
        "segments must arrive in stream order"
    )
    print("OK: streamed segments enqueue incrementally in order")


if __name__ == "__main__":
    main()
