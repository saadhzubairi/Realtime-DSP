"""
Configuration constants and settings for the voice transformation system.
Target constraints and default parameters.
"""

from dataclasses import dataclass, field
from typing import Optional
import json
import os

# =============================================================================
# AUDIO CONFIGURATION
# =============================================================================

# Sample rates (Hz)
SAMPLE_RATE_LOW = 16000      # CPU-light prototyping
SAMPLE_RATE_HIGH = 48000     # High fidelity

# Default sample rate
DEFAULT_SAMPLE_RATE = SAMPLE_RATE_LOW

# Frame and hop sizes (in samples at 16 kHz)
# 20-30ms frames, 10ms hop
FRAME_SIZE_MS = 20           # milliseconds
HOP_SIZE_MS = 10             # milliseconds

def samples_from_ms(ms: int, sample_rate: int = DEFAULT_SAMPLE_RATE) -> int:
    """Convert milliseconds to samples."""
    return int(ms * sample_rate / 1000)

# Buffer sizes for PyAudio callback (samples)
BUFFER_SIZES = [128, 256, 512, 1024]
DEFAULT_BUFFER_SIZE = 256

# Ring buffer sizes (in frames, not samples)
INPUT_RING_BUFFER_FRAMES = 32
OUTPUT_RING_BUFFER_FRAMES = 32

# =============================================================================
# DSP PARAMETERS
# =============================================================================

# LPC analysis
LPC_ORDER = 16               # Order for 16 kHz (rule of thumb: fs/1000 + 4)
LPC_ORDER_HIGH = 24          # Order for 48 kHz
PRE_EMPHASIS_COEFF = 0.97

# Pitch detection (YIN algorithm)
YIN_THRESHOLD = 0.15         # Harmonicity threshold for voiced/unvoiced
F0_MIN = 50                  # Hz - minimum fundamental frequency
F0_MAX = 500                 # Hz - maximum fundamental frequency
PITCH_MEDIAN_FILTER_SIZE = 5 # Frames for smoothing

# Formant estimation
NUM_FORMANTS = 3             # F1, F2, F3
FORMANT_BANDWIDTH_THRESHOLD = 400  # Hz - reject formants with bandwidth > this

# STFT parameters
FFT_SIZE = 512               # For 16 kHz
FFT_SIZE_HIGH = 2048         # For 48 kHz
WINDOW_TYPE = 'hann'

# =============================================================================
# CALIBRATION PARAMETERS
# =============================================================================

CALIBRATION_DURATION_S = 5.0  # seconds of recording for calibration
MIN_VOICED_FRAMES_RATIO = 0.3 # Minimum ratio of voiced frames for valid profile

# =============================================================================
# TRANSFORM PARAMETERS
# =============================================================================

# Smoothing for real-time controls (exponential moving average)
EMA_ALPHA_FAST = 0.3         # Fast response
EMA_ALPHA_SLOW = 0.1         # Smooth response
EMA_ALPHA_ENVELOPE = 0.05    # Very smooth for envelope ratio

# Pitch mapping clamp range (semitones from median)
PITCH_SHIFT_MAX_SEMITONES = 12

# Formant warp range
FORMANT_WARP_MIN = 0.8
FORMANT_WARP_MAX = 1.25

# =============================================================================
# UI PARAMETERS
# =============================================================================

UI_UPDATE_INTERVAL_MS = 50   # ~20 fps for meters/plots (reduced from 33 for performance)
METER_DECAY_RATE = 0.85      # Level meter decay per update

# Spectrogram display
SPECTROGRAM_HISTORY_FRAMES = 100
SPECTROGRAM_FREQ_BINS = 256

# =============================================================================
# THREADING PARAMETERS
# =============================================================================

QUEUE_MAX_SIZE = 64          # Maximum frames in inter-thread queues
DSP_WORKER_TIMEOUT_S = 0.01  # Timeout for queue operations


@dataclass
class AudioConfig:
    """Runtime audio configuration."""
    sample_rate: int = DEFAULT_SAMPLE_RATE
    buffer_size: int = DEFAULT_BUFFER_SIZE
    channels: int = 1  # Mono only
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
        return FFT_SIZE if self.sample_rate <= SAMPLE_RATE_LOW else FFT_SIZE_HIGH
    
    @property
    def lpc_order(self) -> int:
        return LPC_ORDER if self.sample_rate <= SAMPLE_RATE_LOW else LPC_ORDER_HIGH


@dataclass
class TransformConfig:
    """Live transform parameters (controlled by UI sliders)."""
    wet_dry: float = 1.0              # 0.0 = dry, 1.0 = wet
    pitch_strength: float = 0.0       # 0.0 = no pitch mapping
    formant_strength: float = 0.0     # 0.0 = no formant mapping
    envelope_strength: float = 0.0    # 0.0 = no envelope matching
    unvoiced_mode: str = 'bypass'     # 'bypass' or 'noise_shaped'
    
    # Computed warp factor from calibration
    formant_warp_factor: float = 1.0


@dataclass 
class AppState:
    """Shared application state."""
    audio_config: AudioConfig = field(default_factory=AudioConfig)
    transform_config: TransformConfig = field(default_factory=TransformConfig)
    
    # Runtime state
    is_streaming: bool = False
    is_calibrating: bool = False
    
    # Metrics (updated by DSP worker)
    input_level_db: float = -60.0
    output_level_db: float = -60.0
    current_f0: float = 0.0
    underrun_count: int = 0
    overrun_count: int = 0
    dsp_time_avg_ms: float = 0.0
    dsp_time_max_ms: float = 0.0
    queue_depth: int = 0
    
    # Profile paths
    source_profile_path: Optional[str] = None
    target_profile_path: Optional[str] = None


def get_profiles_directory() -> str:
    """Get the directory for storing voice profiles."""
    profiles_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'profiles')
    os.makedirs(profiles_dir, exist_ok=True)
    return profiles_dir


def save_config(config: AppState, path: str):
    """Save configuration to JSON file."""
    data = {
        'audio': {
            'sample_rate': config.audio_config.sample_rate,
            'buffer_size': config.audio_config.buffer_size,
        },
        'transform': {
            'wet_dry': config.transform_config.wet_dry,
            'pitch_strength': config.transform_config.pitch_strength,
            'formant_strength': config.transform_config.formant_strength,
            'envelope_strength': config.transform_config.envelope_strength,
            'unvoiced_mode': config.transform_config.unvoiced_mode,
        }
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_config(path: str) -> dict:
    """Load configuration from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)
