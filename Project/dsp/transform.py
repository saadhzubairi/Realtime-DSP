"""
Simple transform pipeline module.
Implements real-time frequency scaling via FFT bin shifting.
Based on the TakeHomeMid Q1 approach.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import threading

from .voice_profile import VoiceProfile
from utils.config import TransformConfig, AudioConfig


@dataclass  
class TransformState:
    """State for the transform pipeline."""
    # Previous overlap tail for OLA
    prev_tail: Optional[np.ndarray] = None
    
    # Smoothed pitch ratio
    smoothed_pitch_ratio: float = 1.0


class VoiceTransformPipeline:
    """
    Simple real-time voice transformation pipeline.
    
    Uses FFT bin interpolation to shift frequencies.
    Much lighter than full LPC/formant processing.
    """
    
    def __init__(
        self,
        audio_config: AudioConfig,
        source_profile: Optional[VoiceProfile] = None,
        target_profile: Optional[VoiceProfile] = None
    ):
        """
        Initialize the transform pipeline.
        
        Args:
            audio_config: Audio configuration
            source_profile: Source voice profile (A)
            target_profile: Target voice profile (B)
        """
        self.audio_config = audio_config
        self.source_profile = source_profile
        self.target_profile = target_profile
        
        # Configuration (can be updated at runtime)
        self.config = TransformConfig()
        self._config_lock = threading.Lock()
        
        # State
        self.state = TransformState()
        
        # Frame size and hop
        self.frame_size = audio_config.frame_size
        self.hop_size = audio_config.hop_size
        
        # Initialize overlap tail
        self.state.prev_tail = np.zeros(self.frame_size - self.hop_size, dtype=np.float32)
    
    def set_profiles(self, source: VoiceProfile, target: VoiceProfile):
        """Set source and target voice profiles."""
        self.source_profile = source
        self.target_profile = target
    
    def update_config(self, config: TransformConfig):
        """Update transform configuration (thread-safe)."""
        with self._config_lock:
            self.config = config
    
    def reset(self):
        """Reset pipeline state."""
        self.state = TransformState()
        self.state.prev_tail = np.zeros(self.frame_size - self.hop_size, dtype=np.float32)
    
    def _compute_pitch_ratio(self) -> float:
        """
        Compute pitch ratio from source to target profiles.
        
        Returns:
            Ratio to multiply source F0 by to get target F0
        """
        if self.source_profile is None or self.target_profile is None:
            return 1.0
        
        src_f0 = self.source_profile.f0_median_hz
        tgt_f0 = self.target_profile.f0_median_hz
        
        if src_f0 > 0 and tgt_f0 > 0:
            return tgt_f0 / src_f0
        
        return 1.0
    
    def _process_block_fft_scaling(self, input_block: np.ndarray, alpha: float) -> np.ndarray:
        """
        Process a block using FFT frequency scaling with interpolation.
        
        This shifts frequencies by interpolating FFT bins.
        
        Args:
            input_block: Input audio samples
            alpha: Scaling factor (>1 = shift up, <1 = shift down)
            
        Returns:
            Frequency-scaled output block
        """
        # Apply window
        window = np.hanning(len(input_block))
        windowed = input_block * window
        
        # FFT
        X = np.fft.rfft(windowed)
        Y = np.zeros_like(X)
        
        # Interpolate bins to shift frequencies
        for src_ind in range(X.size):
            dst_ind = src_ind / alpha
            if dst_ind < X.size - 1:
                i0 = int(np.floor(dst_ind))
                i1 = i0 + 1
                t = dst_ind - i0
                Y[src_ind] = (1 - t) * X[i0] + t * X[i1]
        
        # IFFT
        y = np.fft.irfft(Y, n=len(input_block))
        
        return y.astype(np.float32)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Process a single frame through the transform pipeline.
        
        Args:
            frame: Input audio frame (frame_size samples)
            
        Returns:
            Tuple of (output frame, metrics dict)
        """
        # Get current config (thread-safe)
        with self._config_lock:
            wet_dry = self.config.wet_dry
            pitch_strength = self.config.pitch_strength
        
        metrics = {}
        
        # Compute pitch ratio from profiles
        base_pitch_ratio = self._compute_pitch_ratio()
        
        # Apply strength
        # At strength=0, ratio=1.0 (no change)
        # At strength=1, ratio=full profile ratio
        alpha = 1.0 + (base_pitch_ratio - 1.0) * pitch_strength
        
        # Smooth the ratio
        smooth_factor = 0.1
        self.state.smoothed_pitch_ratio = (
            smooth_factor * alpha + 
            (1 - smooth_factor) * self.state.smoothed_pitch_ratio
        )
        
        metrics['pitch_ratio'] = self.state.smoothed_pitch_ratio
        metrics['f0'] = 0.0  # We're not doing pitch detection in simple mode
        metrics['is_voiced'] = True
        
        # Process with FFT scaling
        if abs(self.state.smoothed_pitch_ratio - 1.0) > 0.01:
            output_block = self._process_block_fft_scaling(frame, self.state.smoothed_pitch_ratio)
        else:
            # No shift needed, just pass through with window
            window = np.hanning(len(frame))
            output_block = (frame * window).astype(np.float32)
        
        # Apply wet/dry mix
        if wet_dry < 1.0:
            dry_block = frame.copy()
            output_block = wet_dry * output_block + (1.0 - wet_dry) * dry_block
        
        # The caller (DSPWorker) handles overlap-add, so just return the windowed frame
        return output_block, metrics
    
    def process_with_overlap_add(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Process a frame with built-in overlap-add.
        
        This method handles OLA internally if needed.
        
        Args:
            frame: Input audio frame
            
        Returns:
            Tuple of (hop_size output samples, metrics)
        """
        output_block, metrics = self.process_frame(frame)
        
        # Overlap-add with previous tail
        if self.state.prev_tail is not None and len(self.state.prev_tail) > 0:
            overlap_len = len(self.state.prev_tail)
            if overlap_len <= len(output_block):
                output_block[:overlap_len] = 0.5 * (
                    output_block[:overlap_len] + self.state.prev_tail
                )
        
        # Extract output hop and store tail
        hop_output = output_block[:self.hop_size].copy()
        self.state.prev_tail = output_block[self.hop_size:].copy()
        
        return hop_output, metrics
