"""Audio processing pipeline: Spectral Subtraction Noise Suppression, Transient Suppression & AGC.

- Noise Suppressor: Spectral subtraction tracking noise profile for fan/AC noise.
- Transient Suppressor: Median filter / threshold limiter for keyboard clicks.
- Automatic Gain Control (AGC): Normalizes target speech volume.
"""

import numpy as np


class NoiseSuppressor:
    """Spectral subtraction noise suppressor for stationary noise (fan/AC)."""

    def __init__(self, alpha: float = 2.0, beta: float = 0.02):
        """
        Args:
            alpha: Over-subtraction factor.
            beta: Spectral floor factor.
        """
        self.alpha = alpha
        self.beta = beta
        self.noise_psd = None

    def process_block(self, audio_block: np.ndarray) -> np.ndarray:
        """Applies spectral subtraction to 1D float32 audio block."""
        if len(audio_block) == 0:
            return audio_block

        # STFT via FFT
        fft_spec = np.fft.rfft(audio_block)
        magnitude = np.abs(fft_spec)
        phase = np.angle(fft_spec)

        # Estimate noise PSD on silent/low-energy initial frames
        if self.noise_psd is None:
            self.noise_psd = magnitude
        else:
            # Smooth noise estimate when frame energy is low
            if np.mean(magnitude) < np.mean(self.noise_psd) * 1.5:
                self.noise_psd = 0.95 * self.noise_psd + 0.05 * magnitude

        # Spectral subtraction: |S|^2 = |Y|^2 - alpha * |N|^2
        subtracted = magnitude**2 - self.alpha * (self.noise_psd**2)
        # Apply spectral floor
        floor = self.beta * (self.noise_psd**2)
        subtracted = np.maximum(subtracted, floor)

        clean_mag = np.sqrt(subtracted)
        clean_spec = clean_mag * np.exp(1j * phase)

        # Inverse FFT
        cleaned_block = np.fft.irfft(clean_spec, n=len(audio_block))
        return cleaned_block.astype(np.float32)


class TransientSuppressor:
    """Impulse/transient noise suppressor for sudden clicks and keyboard taps."""

    def __init__(self, threshold_factor: float = 3.5):
        self.threshold_factor = threshold_factor

    def process_block(self, audio_block: np.ndarray) -> np.ndarray:
        """Suppresses high-amplitude short-duration spikes."""
        if len(audio_block) < 3:
            return audio_block

        out = audio_block.copy()
        mean_amp = np.mean(np.abs(out))
        std_amp = np.std(np.abs(out))
        limit = mean_amp + self.threshold_factor * std_amp

        # Identify spikes exceeding dynamic limit
        spikes = np.abs(out) > limit
        if np.any(spikes):
            # Soft clip/median replacement for transient spikes
            window_size = 5
            for idx in np.where(spikes)[0]:
                start = max(0, idx - window_size // 2)
                end = min(len(out), idx + window_size // 2 + 1)
                out[idx] = np.median(out[start:end])

        return out


class AutomaticGainControl:
    """Automatic Gain Control (AGC) for normalizing audio volume."""

    def __init__(self, target_db: float = -16.0, max_gain_db: float = 30.0):
        self.target_amplitude = 10 ** (target_db / 20.0)
        self.max_gain = 10 ** (max_gain_db / 20.0)

    def process_block(self, audio_block: np.ndarray) -> np.ndarray:
        """Scales audio block to target RMS amplitude."""
        if len(audio_block) == 0:
            return audio_block

        rms = np.sqrt(np.mean(audio_block**2))
        if rms < 1e-5:
            return audio_block

        gain = self.target_amplitude / rms
        gain = min(gain, self.max_gain)
        return (audio_block * gain).astype(np.float32)
