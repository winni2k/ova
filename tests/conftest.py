import pytest
import struct
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    """Fixture that provides a test client for the FastAPI app."""
    from ova.api import app

    return TestClient(app)


@pytest.fixture
def wav_blob():
    """
    Creates a minimal valid WAV file blob for testing.

    Format: PCM16, mono, 16000 Hz, 0.1 seconds of silence.
    """
    sample_rate = 16000
    num_channels = 1
    bits_per_sample = 16
    duration_seconds = 0.1

    # Generate silence samples
    num_samples = int(sample_rate * duration_seconds)
    samples = [0] * num_samples  # silence

    # Calculate sizes
    block_align = (num_channels * bits_per_sample) // 8
    byte_rate = sample_rate * block_align
    data_size = num_samples * 2  # 2 bytes per sample (16-bit)

    # Build WAV header
    wav_data = bytearray()

    # RIFF header
    wav_data.extend(b"RIFF")
    wav_data.extend(struct.pack("<I", 36 + data_size))  # file size - 8
    wav_data.extend(b"WAVE")

    # fmt chunk
    wav_data.extend(b"fmt ")
    wav_data.extend(struct.pack("<I", 16))  # fmt chunk size
    wav_data.extend(struct.pack("<H", 1))  # audio format (PCM)
    wav_data.extend(struct.pack("<H", num_channels))
    wav_data.extend(struct.pack("<I", sample_rate))
    wav_data.extend(struct.pack("<I", byte_rate))
    wav_data.extend(struct.pack("<H", block_align))
    wav_data.extend(struct.pack("<H", bits_per_sample))

    # data chunk
    wav_data.extend(b"data")
    wav_data.extend(struct.pack("<I", data_size))

    # Write samples as 16-bit signed integers
    for sample in samples:
        wav_data.extend(struct.pack("<h", sample))

    return bytes(wav_data)
