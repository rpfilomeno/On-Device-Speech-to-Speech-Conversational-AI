import numpy as np
from src.utils.audio_processor import (
    NoiseSuppressor,
    TransientSuppressor,
    AutomaticGainControl,
)


def test_audio_processors():
    # 1. Noise Suppressor Test (Fan background noise)
    ns = NoiseSuppressor()
    t = np.linspace(0, 1, 1024, dtype=np.float32)
    steady_noise = np.random.normal(0, 0.05, 1024).astype(np.float32)
    cleaned_noise = ns.process_block(steady_noise)
    assert np.mean(np.abs(cleaned_noise)) < np.mean(
        np.abs(steady_noise)
    ), "Noise suppression failed"

    # 2. Transient Suppressor Test (Click/Tap spike)
    ts = TransientSuppressor()
    speech_with_click = np.random.normal(0, 0.01, 1024).astype(np.float32)
    speech_with_click[500] = 0.9  # sudden click spike
    cleaned_click = ts.process_block(speech_with_click)
    assert cleaned_click[500] < 0.5, "Transient click suppression failed"

    # 3. AGC Test (Normalization)
    agc = AutomaticGainControl(target_db=-16.0)
    low_speech = np.random.normal(0, 0.005, 1024).astype(np.float32)
    boosted_speech = agc.process_block(low_speech)
    assert np.mean(np.abs(boosted_speech)) > np.mean(
        np.abs(low_speech)
    ), "AGC amplification failed"

    print("All audio processor tests passed!")


if __name__ == "__main__":
    test_audio_processors()
