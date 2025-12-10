"""
Transform pipeline - simple direct processing.
No buffering, no accumulation - processes samples immediately.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import threading

from .voice_profile import VoiceProfile
from .pitch_shift import SmoothPitchShifter
from .formant import FormantWarpProcessor
from utils.config import TransformConfig, AudioConfig


@dataclass  
class TransformState:
    """State for the transform pipeline."""
    smoothed_pitch_ratio: float = 1.0


class VoiceTransformPipeline:
    """
    Simple voice transformation - processes samples immediately.
    No buffering, no delay.
    """
    
    def __init__(
        self,
        audio_config: AudioConfig,
        source_profile: Optional[VoiceProfile] = None,
        target_profile: Optional[VoiceProfile] = None
    ):
        self.audio_config = audio_config
        self.source_profile = source_profile
        self.target_profile = target_profile
        
        # Configuration
        self.config = TransformConfig()
        self._config_lock = threading.Lock()
        
        # State
        self.state = TransformState()
        
        # Simple pitch shifter
        self.pitch_shifter = SmoothPitchShifter(
            sample_rate=audio_config.sample_rate
        )
        
        # Formant processor
        self.formant_processor = FormantWarpProcessor(
            sample_rate=audio_config.sample_rate,
            lpc_order=audio_config.lpc_order,
            fft_size=audio_config.fft_size
        )
        self._profile_formant_warp = 1.0
    
    def set_profiles(self, source: VoiceProfile, target: VoiceProfile):
        """Set source and target voice profiles."""
        self.source_profile = source
        self.target_profile = target
        self._update_formant_warp_from_profiles()
    
    def update_config(self, config: TransformConfig):
        """Update transform configuration."""
        with self._config_lock:
            self.config = config
    
    def reset(self):
        """Reset pipeline state."""
        self.state = TransformState()
        self.pitch_shifter.reset()
        if self.formant_processor:
            self.formant_processor.reset()
    
    def _compute_pitch_ratio(self) -> float:
        """Compute pitch ratio from profiles."""
        if self.source_profile is None or self.target_profile is None:
            return 1.0
        
        src_f0 = self.source_profile.f0_median_hz
        tgt_f0 = self.target_profile.f0_median_hz
        
        if src_f0 > 0 and tgt_f0 > 0:
            return tgt_f0 / src_f0
        
        return 1.0
    
    def process_direct(self, samples: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Process samples directly - no buffering, immediate output.
        
        Args:
            samples: Input audio samples
            
        Returns:
            Tuple of (output samples, metrics dict)
        """
        # Get current config
        with self._config_lock:
            wet_dry = self.config.wet_dry
            pitch_strength = self.config.pitch_strength
            formant_strength = self.config.formant_strength
        
        metrics = {'f0': 0.0, 'is_voiced': True}
        
        # Compute target pitch ratio
        base_ratio = self._compute_pitch_ratio()
        target_ratio = 1.0 + (base_ratio - 1.0) * pitch_strength
        target_ratio = max(0.5, min(2.0, target_ratio))
        
        self.state.smoothed_pitch_ratio = target_ratio
        metrics['pitch_ratio'] = target_ratio
        
        # Set pitch ratio and process
        self.pitch_shifter.set_pitch_ratio(target_ratio)
        shifted = self.pitch_shifter.process(samples)
        
        # Optional formant warping
        if (
            formant_strength > 0.0
            and self.formant_processor is not None
        ):
            warp_factor = 1.0 + (self._profile_formant_warp - 1.0) * formant_strength
            shifted = self.formant_processor.process(
                shifted,
                warp_factor=warp_factor,
                mix=formant_strength
            )
        
        # Apply wet/dry mix
        if wet_dry < 1.0:
            output = wet_dry * shifted + (1 - wet_dry) * samples
        else:
            output = shifted
        
        return output.astype(np.float32), metrics
    
    # Aliases for compatibility
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        return self.process_direct(frame)
    
    def process_with_overlap_add(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        return self.process_direct(frame)

    def _update_formant_warp_from_profiles(self):
        """Estimate warp factor from calibrated profile statistics."""
        ratios = []
        if self.source_profile and self.target_profile:
            pairs = [
                (self.source_profile.formant_f1_median, self.target_profile.formant_f1_median),
                (self.source_profile.formant_f2_median, self.target_profile.formant_f2_median)
            ]
            for src_val, tgt_val in pairs:
                if src_val > 0 and tgt_val > 0:
                    ratios.append(tgt_val / src_val)
        
        if ratios:
            warp = float(np.clip(np.median(ratios), 0.8, 1.25))
        else:
            warp = 1.0
        
        self._profile_formant_warp = warp
        with self._config_lock:
            self.config.formant_warp_factor = warp
