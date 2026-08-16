from .config import settings

GRACE_WORDS = 2


class TextChunker:
    """A class to handle intelligent text chunking for voice generation."""

    def __init__(self):
        """Initialize the TextChunker with break points and priorities."""
        self.current_text = []
        self.found_first_sentence = False
        self.semantic_breaks = {
            "however": 4,
            "therefore": 4,
            "furthermore": 4,
            "moreover": 4,
            "nevertheless": 4,
            "while": 3,
            "although": 3,
            "unless": 3,
            "since": 3,
            "and": 2,
            "but": 2,
            "because": 2,
            "then": 2,
        }
        self.punctuation_priorities = {
            ".": 5,
            "!": 5,
            "?": 5,
            ";": 4,
            ":": 4,
            ",": 3,
            "-": 2,
        }

    def _natural_break_priority(self, word: str) -> int:
        """Priority of the natural break at `word` (0 = none)."""
        priority = self.semantic_breaks.get(word.lower(), 0)
        for punct, punct_priority in self.punctuation_priorities.items():
            if word.endswith(punct):
                priority = max(priority, punct_priority)
        return priority

    def should_process(self, text: str) -> bool:
        """Determines if text should be processed based on length or punctuation.

        Fires when a completed sentence (trailing punctuation) lands, or once the
        stream has MORE than the target word count. A natural break available in
        the grace window starts TTS at a real break; past the grace window it
        hard-cuts so long breakless text still gets spoken.

        Args:
            text (str): The text to check.

        Returns:
            bool: True if the text should be processed, False otherwise.
        """
        words = text.split()
        if not words:
            return False

        if any(text.endswith(p) for p in self.punctuation_priorities):
            return True

        target = (
            settings.FIRST_SENTENCE_SIZE
            if not self.found_first_sentence
            else settings.TARGET_SIZE
        )
        if len(words) <= target:
            return False

        if len(words) > target + GRACE_WORDS:
            return True

        return any(
            self._natural_break_priority(w) > 0
            for w in words[: target + GRACE_WORDS]
        )

    def find_break_point(self, words: list, target_size: int) -> int:
        """Finds the greedy break point in text: the furthest natural break
        (punctuation or semantic connector) at or before target_size plus the
        grace window, so chunks fill up to TARGET_SIZE words. Falls back to a
        hard cut at target_size.

        Args:
            words (list): The list of words to find a break point in.
            target_size (int): The target size of the chunk.

        Returns:
            int: The index of the break point.
        """
        if len(words) <= target_size:
            return len(words)

        for i in range(min(len(words), target_size + GRACE_WORDS) - 1, -1, -1):
            if self._natural_break_priority(words[i]) > 0:
                return i + 1

        return target_size

    def process(self, text: str, audio_queue) -> str:
        """Process text chunk and return remaining text.

        Args:
            text (str): The text to process.
            audio_queue: The audio queue to add sentences to.

        Returns:
            str: The remaining text after processing.
        """
        if not text:
            return ""

        words = text.split()
        if not words:
            return ""

        target_size = (
            settings.FIRST_SENTENCE_SIZE
            if not self.found_first_sentence
            else settings.TARGET_SIZE
        )
        split_point = self.find_break_point(words, target_size)

        if split_point:
            chunk = " ".join(words[:split_point]).strip()
            remaining = " ".join(words[split_point:]) if split_point < len(words) else ""
            if chunk:
                chunk = chunk.rstrip(",")
                audio_queue.add_sentences([chunk])
                self.found_first_sentence = True
                return remaining
            else:
                return remaining

        return ""

    def flush(self, audio_queue) -> str:
        """Flushes all remaining text in current_text to the audio queue, ensuring no text is lost.

        Args:
            audio_queue: The audio queue to add sentences to.

        Returns:
            str: The flushed text content.
        """
        full_text = "".join(self.current_text).strip()
        self.current_text = []
        if not full_text:
            return ""

        remaining = full_text
        flushed_chunks = []

        while remaining.strip():
            words = remaining.strip().split()
            if not words:
                break

            target_size = (
                settings.FIRST_SENTENCE_SIZE
                if not self.found_first_sentence
                else settings.TARGET_SIZE
            )

            if len(words) <= target_size:
                chunk = " ".join(words).strip()
                if chunk:
                    chunk = chunk.rstrip(",")
                    audio_queue.add_sentences([chunk])
                    self.found_first_sentence = True
                    flushed_chunks.append(chunk)
                break

            split_point = self.find_break_point(words, target_size)
            if split_point > 0:
                chunk = " ".join(words[:split_point]).strip()
                if chunk:
                    chunk = chunk.rstrip(",")
                    audio_queue.add_sentences([chunk])
                    self.found_first_sentence = True
                    flushed_chunks.append(chunk)

                remaining = " ".join(words[split_point:]).strip() if split_point < len(words) else ""
            else:
                chunk = " ".join(words).strip()
                if chunk:
                    chunk = chunk.rstrip(",")
                    audio_queue.add_sentences([chunk])
                    self.found_first_sentence = True
                    flushed_chunks.append(chunk)
                break

        return " ".join(flushed_chunks)


