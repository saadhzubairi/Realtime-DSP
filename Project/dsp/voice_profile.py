"""
Voice profile module.
Handles profile extraction, serialization, and loading.
"""

import numpy as np
import json
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from datetime import datetime

from .pitch import PitchTracker, hz_to_log_f0, log_f0_to_hz
from .lpc import compute_lpc, lpc_to_spectrum
from .formant import FormantTracker, Formant, FormantEstimate
from utils.config import (
    DEFAULT_SAMPLE_RATE, 
    FRAME_SIZE_MS, 
    HOP_SIZE_MS,
    LPC_ORDER,
    samples_from_ms,
    get_profiles_directory,
)


@dataclass
class VoiceProfile:
    """
    Voice profile containing extracted features for voice transformation.
    """
    # Metadata
    name: str = ""
    created_at: str = ""
    sample_rate: int = DEFAULT_SAMPLE_RATE
    frame_size: int = 0
    hop_size: int = 0
    duration_s: float = 0.0
    
    # Pitch statistics (log-F0 domain)
    f0_median_hz: float = 0.0
    f0_p05_hz: float = 0.0
    f0_p95_hz: float = 0.0
    f0_std_log: float = 0.0
    voiced_ratio: float = 0.0
    
    # Spectral envelope (average LPC-based envelope in log magnitude)
    envelope_log_mag: np.ndarray = field(default_factory=lambda: np.array([]))
    envelope_fft_size: int = 512
    
    # LPC coefficients summary (average)
    lpc_coefficients: np.ndarray = field(default_factory=lambda: np.array([]))
    lpc_order: int = LPC_ORDER
    lpc_gain: float = 0.0
    
    # Formant medians
    formant_f1_median: float = 0.0
    formant_f2_median: float = 0.0
    formant_f3_median: float = 0.0
    
    def __post_init__(self):
        """Ensure numpy arrays are proper type."""
        if not isinstance(self.envelope_log_mag, np.ndarray):
            self.envelope_log_mag = np.array(self.envelope_log_mag, dtype=np.float32)
        if not isinstance(self.lpc_coefficients, np.ndarray):
            self.lpc_coefficients = np.array(self.lpc_coefficients, dtype=np.float32)
    
    @property
    def is_valid(self) -> bool:
        """Check if profile has valid data."""
        return (
            self.f0_median_hz > 0 and
            self.voiced_ratio > 0.1 and
            len(self.envelope_log_mag) > 0
        )
    
    def get_envelope_magnitude(self) -> np.ndarray:
        """Get envelope as linear magnitude."""
        return np.exp(self.envelope_log_mag)
    
    def get_summary(self) -> str:
        """Get human-readable summary of the profile."""
        return f"""Voice Profile: {self.name}
Created: {self.created_at}
Duration: {self.duration_s:.1f}s

Pitch (F0):
  Median: {self.f0_median_hz:.1f} Hz
  Range (5%-95%): {self.f0_p05_hz:.1f} - {self.f0_p95_hz:.1f} Hz
  Voiced ratio: {self.voiced_ratio*100:.1f}%

Formants:
  F1: {self.formant_f1_median:.0f} Hz
  F2: {self.formant_f2_median:.0f} Hz
  F3: {self.formant_f3_median:.0f} Hz
"""


def extract_profile(
    audio: np.ndarray,
    sample_rate: int,
    name: str = "",
    frame_size_ms: int = FRAME_SIZE_MS,
    hop_size_ms: int = HOP_SIZE_MS,
    lpc_order: int = LPC_ORDER,
    fft_size: int = 512
) -> VoiceProfile:
    """
    Extract a voice profile from audio data.
    
    Args:
        audio: Audio samples (mono, float32)
        sample_rate: Sample rate in Hz
        name: Profile name
        frame_size_ms: Frame size in milliseconds
        hop_size_ms: Hop size in milliseconds
        lpc_order: LPC order for envelope estimation
        fft_size: FFT size for envelope
        
    Returns:
        Extracted VoiceProfile
    """
    frame_size = samples_from_ms(frame_size_ms, sample_rate)
    hop_size = samples_from_ms(hop_size_ms, sample_rate)
    
    # Initialize trackers
    pitch_tracker = PitchTracker(
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size
    )
    
    formant_tracker = FormantTracker(
        sample_rate=sample_rate,
        lpc_order=lpc_order
    )
    
    # Storage for frame-level features
    envelopes = []
    lpc_coeffs_list = []
    lpc_gains = []
    
    # Process frame by frame (skip some frames for speed)
    num_frames = (len(audio) - frame_size) // hop_size + 1
    
    # Skip every N frames to speed up extraction (still get enough samples)
    # For 5 seconds at 16kHz with 10ms hop = 500 frames
    # Processing every 4th frame = 125 frames, still plenty for statistics
    skip_factor = max(1, num_frames // 150)  # Target ~150 frames max
    
    for i in range(0, num_frames, skip_factor):
        start = i * hop_size
        end = start + frame_size
        
        if end > len(audio):
            break
        
        frame = audio[start:end].astype(np.float32)
        
        # Apply window
        windowed = frame * np.hanning(frame_size)
        
        # Pitch tracking
        pitch_estimate = pitch_tracker.process_frame(frame)
        
        # Formant tracking
        formant_estimate = formant_tracker.process_frame(frame)
        
        # LPC analysis (only for voiced frames)
        if pitch_estimate.is_voiced:
            lpc_result = compute_lpc(windowed, lpc_order)
            
            if lpc_result.gain > 0:
                envelope = lpc_to_spectrum(
                    lpc_result.coefficients,
                    fft_size,
                    lpc_result.gain
                )
                # Store log magnitude envelope
                envelopes.append(np.log(np.maximum(envelope, 1e-10)))
                lpc_coeffs_list.append(lpc_result.coefficients)
                lpc_gains.append(lpc_result.gain)
    
    # Aggregate statistics
    pitch_stats = pitch_tracker.get_statistics()
    formant_stats = formant_tracker.get_statistics()
    
    # Average envelope (median in log domain for robustness)
    if envelopes:
        envelope_array = np.array(envelopes)
        avg_envelope_log = np.median(envelope_array, axis=0)
    else:
        avg_envelope_log = np.zeros(fft_size // 2 + 1, dtype=np.float32)
    
    # Average LPC coefficients
    if lpc_coeffs_list:
        avg_lpc = np.mean(lpc_coeffs_list, axis=0)
        avg_gain = np.mean(lpc_gains)
    else:
        avg_lpc = np.zeros(lpc_order + 1, dtype=np.float32)
        avg_gain = 0.0
    
    # Compute F0 std in log domain
    if pitch_stats['f0_median'] > 0 and pitch_stats['f0_p05'] > 0:
        f0_values_log = [hz_to_log_f0(pitch_stats['f0_p05']),
                        hz_to_log_f0(pitch_stats['f0_median']),
                        hz_to_log_f0(pitch_stats['f0_p95'])]
        f0_std_log = np.std(f0_values_log)
    else:
        f0_std_log = 0.0
    
    # Create profile
    profile = VoiceProfile(
        name=name,
        created_at=datetime.now().isoformat(),
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size,
        duration_s=len(audio) / sample_rate,
        
        f0_median_hz=pitch_stats['f0_median'],
        f0_p05_hz=pitch_stats['f0_p05'],
        f0_p95_hz=pitch_stats['f0_p95'],
        f0_std_log=f0_std_log,
        voiced_ratio=pitch_stats['voiced_ratio'],
        
        envelope_log_mag=avg_envelope_log.astype(np.float32),
        envelope_fft_size=fft_size,
        
        lpc_coefficients=avg_lpc.astype(np.float32),
        lpc_order=lpc_order,
        lpc_gain=avg_gain,
        
        formant_f1_median=formant_stats.get('F1_median', 0.0),
        formant_f2_median=formant_stats.get('F2_median', 0.0),
        formant_f3_median=formant_stats.get('F3_median', 0.0),
    )
    
    return profile


def save_profile(profile: VoiceProfile, filepath: str):
    """
    Save a voice profile to disk.
    
    Saves as .npz for numpy arrays and .json for metadata.
    
    Args:
        profile: Profile to save
        filepath: Base path (without extension)
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # Save numpy arrays
    npz_path = filepath + '.npz'
    np.savez(
        npz_path,
        envelope_log_mag=profile.envelope_log_mag,
        lpc_coefficients=profile.lpc_coefficients
    )
    
    # Save metadata as JSON
    json_path = filepath + '.json'
    metadata = {
        'name': profile.name,
        'created_at': profile.created_at,
        'sample_rate': profile.sample_rate,
        'frame_size': profile.frame_size,
        'hop_size': profile.hop_size,
        'duration_s': profile.duration_s,
        'f0_median_hz': profile.f0_median_hz,
        'f0_p05_hz': profile.f0_p05_hz,
        'f0_p95_hz': profile.f0_p95_hz,
        'f0_std_log': profile.f0_std_log,
        'voiced_ratio': profile.voiced_ratio,
        'envelope_fft_size': profile.envelope_fft_size,
        'lpc_order': profile.lpc_order,
        'lpc_gain': profile.lpc_gain,
        'formant_f1_median': profile.formant_f1_median,
        'formant_f2_median': profile.formant_f2_median,
        'formant_f3_median': profile.formant_f3_median,
    }
    
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_profile(filepath: str) -> VoiceProfile:
    """
    Load a voice profile from disk.
    
    Args:
        filepath: Base path (without extension)
        
    Returns:
        Loaded VoiceProfile
    """
    # Load numpy arrays
    npz_path = filepath + '.npz'
    npz_data = np.load(npz_path)
    
    # Load metadata
    json_path = filepath + '.json'
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    # Create profile
    profile = VoiceProfile(
        name=metadata['name'],
        created_at=metadata['created_at'],
        sample_rate=metadata['sample_rate'],
        frame_size=metadata['frame_size'],
        hop_size=metadata['hop_size'],
        duration_s=metadata['duration_s'],
        f0_median_hz=metadata['f0_median_hz'],
        f0_p05_hz=metadata['f0_p05_hz'],
        f0_p95_hz=metadata['f0_p95_hz'],
        f0_std_log=metadata['f0_std_log'],
        voiced_ratio=metadata['voiced_ratio'],
        envelope_log_mag=npz_data['envelope_log_mag'].astype(np.float32),
        envelope_fft_size=metadata['envelope_fft_size'],
        lpc_coefficients=npz_data['lpc_coefficients'].astype(np.float32),
        lpc_order=metadata['lpc_order'],
        lpc_gain=metadata['lpc_gain'],
        formant_f1_median=metadata['formant_f1_median'],
        formant_f2_median=metadata['formant_f2_median'],
        formant_f3_median=metadata['formant_f3_median'],
    )
    
    return profile


def list_profiles(directory: Optional[str] = None) -> List[str]:
    """
    List available voice profiles in a directory.
    
    Args:
        directory: Directory to search (default: profiles directory)
        
    Returns:
        List of profile base names (without extension)
    """
    if directory is None:
        directory = get_profiles_directory()
    
    profiles = []
    
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                base_name = filename[:-5]  # Remove .json
                npz_path = os.path.join(directory, base_name + '.npz')
                if os.path.exists(npz_path):
                    profiles.append(base_name)
    
    return sorted(profiles)
