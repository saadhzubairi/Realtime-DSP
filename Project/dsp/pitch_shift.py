"""
Ultra-simple real-time pitch shifting.

No complex buffering - just direct resampling of each incoming block.
Trades some quality for ZERO latency.
"""

import numpy as np


class SmoothPitchShifter:
    """
    Dead-simple pitch shifter - just resample each block directly.
    No overlap-add, no accumulation, no latency.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        block_size: int = 512,  # ignored
        hop_ratio: float = 0.5,  # ignored
    ):
        self.sample_rate = sample_rate
        self.pitch_ratio = 1.0
        self._target_ratio = 1.0
        
        # Tiny overlap buffer for smoothing block boundaries
        self.overlap_size = 32
        self.prev_tail = np.zeros(self.overlap_size, dtype=np.float32)
    
    def set_pitch_ratio(self, ratio: float):
        """Set target pitch ratio."""
        self._target_ratio = np.clip(ratio, 0.5, 2.0)
    
    def process(self, input_samples: np.ndarray) -> np.ndarray:
        """
        Process samples. Returns EXACTLY len(input_samples) output.
        Zero latency - processes each block immediately.
        """
        x = input_samples.astype(np.float32)
        n = len(x)
        
        # Smooth pitch ratio
        self.pitch_ratio = 0.3 * self._target_ratio + 0.7 * self.pitch_ratio
        ratio = self.pitch_ratio
        
        # Bypass if near unity
        if abs(ratio - 1.0) < 0.01:
            return x.copy()
        
        # Simple resampling: read input at variable rate
        # ratio > 1 = higher pitch = read faster = fewer unique input samples used
        # ratio < 1 = lower pitch = read slower = input samples get stretched
        
        # Calculate output indices mapped to input
        out_indices = np.arange(n) * ratio
        
        # Clamp to valid input range
        out_indices = np.clip(out_indices, 0, n - 1.001)
        
        # Linear interpolation
        idx_floor = out_indices.astype(np.int32)
        idx_ceil = np.minimum(idx_floor + 1, n - 1)
        frac = (out_indices - idx_floor).astype(np.float32)
        
        output = (1 - frac) * x[idx_floor] + frac * x[idx_ceil]
        
        # Simple crossfade with previous block tail to reduce clicks
        if self.overlap_size > 0 and len(self.prev_tail) == self.overlap_size:
            fade_len = min(self.overlap_size, n)
            fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
            fade_out = 1 - fade_in
            output[:fade_len] = fade_out * self.prev_tail[:fade_len] + fade_in * output[:fade_len]
        
        # Save tail for next block
        if n >= self.overlap_size:
            self.prev_tail = output[-self.overlap_size:].copy()
        
        return output.astype(np.float32)
    
    def reset(self):
        """Reset state."""
        self.pitch_ratio = 1.0
        self._target_ratio = 1.0
        self.prev_tail = np.zeros(self.overlap_size, dtype=np.float32)


# Aliases for compatibility
VariableDelayPitchShifter = SmoothPitchShifter
SimplePitchShifter = SmoothPitchShifter
