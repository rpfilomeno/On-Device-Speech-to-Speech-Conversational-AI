import numpy as np
from src.utils.aec import EchoCanceller


def test_aec_cancellation():
    aec = EchoCanceller(filter_length=64, mu=0.2)

    # Reference speaker output (sine wave)
    t = np.linspace(0, 1, 16000, dtype=np.float32)
    ref = np.sin(2 * np.pi * 440 * t)

    # Simulated echo (speaker sound picked up by mic with gain 0.5)
    echo = 0.5 * ref
    mic = echo.copy()

    # Run AEC block by block
    block_size = 512
    cleaned_blocks = []

    for i in range(0, len(mic), block_size):
        m_block = mic[i : i + block_size]
        r_block = ref[i : i + block_size]
        c_block = aec.process_block(m_block, r_block)
        cleaned_blocks.append(c_block)

    cleaned = np.concatenate(cleaned_blocks)

    # Initial vs final energy check (after convergence)
    initial_energy = np.mean(np.abs(cleaned[:1024]))
    converged_energy = np.mean(np.abs(cleaned[-2048:]))

    print(f"Initial energy: {initial_energy:.4f}")
    print(f"Converged energy: {converged_energy:.4f}")

    assert converged_energy < initial_energy * 0.2, "AEC failed to attenuate echo"
    print("AEC test passed!")


if __name__ == "__main__":
    test_aec_cancellation()
