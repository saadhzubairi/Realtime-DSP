"""Configuration constants and settings."""

from dataclasses import dataclass
from typing import Optional
import os

SAMPLE_RATE_LOW = 16000
DEFAULT_SAMPLE_RATE = SAMPLE_RATE_LOW

FRAME_SIZE_MS = 20
HOP_SIZE_MS = 10

def samples_from_ms(ms: int, sample_rate: int = DEFAULT_SAMPLE_RATE) -> int:
    return int(ms * sample_rate / 1000)

DEFAULT_BUFFER_SIZE = 256
LPC_ORDER = 16
FFT_SIZE = 512


@dataclass
class AudioConfig:
    """Runtime audio configuration."""
    sample_rate: int = DEFAULT_SAMPLE_RATE
    buffer_size: int = DEFAULT_BUFFER_SIZE
    channels: int = 1
    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None
    
    @property
    def frame_size(self) -> int:
        return samples_from_ms(FRAME_SIZE_MS, self.sample_rate)
    
    @property
    def hop_size(self) -> int:
        return samples_from_ms(HOP_SIZE_MS, self.sample_rate)
    
    @property
    def fft_size(self) -> int:
        return FFT_SIZE
    
    @property
    def lpc_order(self) -> int:
        return LPC_ORDER


def get_profiles_directory() -> str:
    """Get the directory for storing voice profiles."""
    profiles_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'profiles')
    os.makedirs(profiles_dir, exist_ok=True)
    return profiles_dir
