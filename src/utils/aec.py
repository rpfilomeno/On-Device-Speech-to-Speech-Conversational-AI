"""AEC (Acoustic Echo Cancellation) module.

Provides a normalized adaptive filter (NLMS) to suppress speaker playback echo
from microphone input signal before Voice Activity Detection (VAD).
"""

import numpy as np


class EchoCanceller:
    """Normalized Least Mean Squares (NLMS) Acoustic Echo Canceller."""

    def __init__(self, filter_length: int = 512, mu: float = 0.1, eps: float = 1e-6):
        """
        Args:
            filter_length: Number of filter taps.
            mu: Step-size (learning rate) for NLMS filter update.
            eps: Regularization factor to prevent division by zero.
        """
        self.filter_length = filter_length
        self.mu = mu
        self.eps = eps
        self.weights = np.zeros(filter_length, dtype=np.float32)
        self.ref_buffer = np.zeros(filter_length, dtype=np.float32)

    def reset(self):
        """Resets filter weights and reference audio buffer."""
        self.weights.fill(0.0)
        self.ref_buffer.fill(0.0)

    def process_sample(self, mic_sample: float, ref_sample: float) -> float:
        """Processes a single sample through NLMS adaptive filter.

        Args:
            mic_sample: Microphone input sample.
            ref_sample: Reference speaker playback sample.

        Returns:
            float: Error signal (cleaned microphone audio after echo subtraction).
        """
        # Shift reference buffer left and insert new reference sample at the end
        self.ref_buffer[1:] = self.ref_buffer[:-1]
        self.ref_buffer[0] = ref_sample

        # Predict echo: y = w^T * x
        echo_est = np.dot(self.weights, self.ref_buffer)

        # Subtract estimated echo from microphone signal: e = d - y
        error = mic_sample - echo_est

        # NLMS Weight update: w(n+1) = w(n) + (mu / (x^T * x + eps)) * e(n) * x(n)
        norm = np.dot(self.ref_buffer, self.ref_buffer) + self.eps
        self.weights += (self.mu / norm) * error * self.ref_buffer

        return float(error)

    def process_block(self, mic_block: np.ndarray, ref_block: np.ndarray) -> np.ndarray:
        """Processes a 1D block of microphone samples against speaker reference samples.

        Args:
            mic_block: 1D numpy array of mic samples (float32).
            ref_block: 1D numpy array of reference playback samples (float32).

        Returns:
            np.ndarray: Echo-cancelled microphone audio block.
        """
        mic_block = np.asarray(mic_block, dtype=np.float32)
        ref_block = np.asarray(ref_block, dtype=np.float32)

        min_len = min(len(mic_block), len(ref_block))
        if min_len == 0:
            return mic_block

        cleaned = np.zeros(min_len, dtype=np.float32)
        for i in range(min_len):
            cleaned[i] = self.process_sample(mic_block[i], ref_block[i])

        if len(mic_block) > min_len:
            # If mic block longer than ref block, append remaining mic samples as-is
            cleaned = np.concatenate([cleaned, mic_block[min_len:]])

        return cleaned
