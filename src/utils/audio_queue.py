from queue import Queue
import threading
import time
from typing import Optional, Tuple, List
import numpy as np
import logging

from .config import settings

logging.getLogger("phonemizer").setLevel(logging.ERROR)
logging.getLogger("speechbrain.utils.quirks").setLevel(logging.ERROR)
logging.basicConfig(format="%(message)s", level=logging.INFO)


class AudioGenerationQueue:
    """
    A queue system for managing asynchronous in-memory audio generation from text input.
    """

    def __init__(
        self, generator, speed: float = 1.0
    ):
        """
        Initialize the audio generation queue system.

        Args:
            generator: Audio generator instance for text-to-speech conversion
            speed: Speed multiplier for audio generation (default: 1.0)
        """
        self.generator = generator
        self.speed = speed
        self.lock = threading.Lock()
        self.sentence_queue = Queue()
        self.audio_queue = Queue()
        self.is_running = False
        self.is_generating = False
        self.generation_thread = None
        self.sentences_processed = 0
        self.audio_generated = 0
        self.failed_sentences = []

    def start(self):
        """
        Start the audio generation thread if not already running.
        The thread will process sentences from the queue until stopped.
        """
        if not self.is_running:
            self.is_running = True
            self.generation_thread = threading.Thread(target=self._generation_worker)
            self.generation_thread.daemon = True
            self.generation_thread.start()

    def stop(self):
        """
        Stop the audio generation thread gracefully.
        Waits for the current queue and active synthesis to complete before stopping.
        Outputs final processing statistics.
        """
        if self.generation_thread:
            while not self.sentence_queue.empty() or self.is_generating:
                time.sleep(0.05)

            self.is_running = False
            self.generation_thread.join()
            self.generation_thread = None

            logging.info(
                f"[TTS] Audio Generation Complete - Processed: {self.sentences_processed}, Generated: {self.audio_generated}, Failed: {len(self.failed_sentences)}"
            )

    def add_sentences(self, sentences: List[str]):
        """
        Add a list of sentences to the generation queue.

        Args:
            sentences: List of text strings to be converted to audio
        """
        added_count = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                if settings.LOG_TTS_CHUNKS:
                    print(f"\n[TTS Queued] {sentence!r}")
                self.sentence_queue.put(sentence)
                added_count += 1

        if not self.is_running:
            self.start()

    def get_next_audio(self) -> Tuple[Optional[np.ndarray], Optional[str], bool]:
        """
        Retrieve the next generated audio segment from the queue.

        Returns:
            Tuple containing:
                - numpy array of audio data (or None if queue is empty)
                - original sentence text (or None)
                - whether this segment is the first for its sentence
        """
        try:
            audio_data, sentence, is_first = self.audio_queue.get_nowait()
            return audio_data, sentence, is_first
        except:
            return None, None, True

    def clear_queues(self):
        """
        Clear both sentence and audio queues, removing all pending items.
        Returns immediately without waiting for queue processing.
        """
        while not self.sentence_queue.empty():
            try:
                self.sentence_queue.get_nowait()
            except:
                pass

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                pass

    def _generation_worker(self):
        """
        Internal worker method that runs in a separate thread.
        Continuously processes sentences from the queue, generating in-memory audio buffers.
        """
        while self.is_running or not self.sentence_queue.empty():
            try:
                try:
                    sentence = self.sentence_queue.get_nowait()
                    self.sentences_processed += 1
                except:
                    if not self.is_running and self.sentence_queue.empty():
                        break
                    time.sleep(0.01)
                    continue

                try:
                    self.is_generating = True
                    if settings.LOG_TTS_CHUNKS:
                        print(f"[TTS Synthesizing] Chunk #{self.sentences_processed}: {sentence!r}")

                    if getattr(self.generator, "streaming", False):
                        produced = False
                        buffer = []
                        for seg in self.generator.generate(
                            sentence, speed=self.speed, stream=True
                        ):
                            if seg is None or len(seg) == 0:
                                continue
                            produced = True
                            buffer.append(seg)
                        if not produced:
                            raise ValueError("Generated audio data is empty")
                        # One segment per sentence: splitting it into smaller pieces
                        # would make playback reopen the audio device per piece,
                        # which sounds choppy. The server buffers the WAV anyway,
                        # so buffering here costs no latency.
                        self.audio_queue.put(
                            (np.concatenate(buffer), sentence, True)
                        )
                    else:
                        audio_data, _ = self.generator.generate(
                            sentence, speed=self.speed
                        )
                        if audio_data is None or len(audio_data) == 0:
                            raise ValueError("Generated audio data is empty")
                        self.audio_queue.put((audio_data, sentence, True))

                    self.audio_generated += 1
                    if settings.LOG_TTS_CHUNKS:
                        print(f"[TTS Synthesized] Chunk #{self.audio_generated} successfully: {sentence!r}")
                except Exception as e:
                    error_msg = str(e)
                    if settings.LOG_TTS_CHUNKS:
                        print(f"[TTS Error] Failed chunk #{self.sentences_processed} {sentence!r}: {error_msg}")
                    self.failed_sentences.append((sentence, error_msg))
                    continue
                finally:
                    self.is_generating = False

            except Exception as e:
                if not self.is_running and self.sentence_queue.empty():
                    break
                time.sleep(0.1)
