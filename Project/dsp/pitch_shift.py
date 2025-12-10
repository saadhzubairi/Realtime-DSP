"""
Pitch shifting with lightweight processing to minimize latency
while reducing aliasing and block artifacts.
"""

import numpy as np
from scipy.signal import firwin, lfilter


class SmoothPitchShifter:
    """
    Pitch shifter that performs block-wise resampling with:
    - Higher-order interpolation
    - Optional anti-aliasing filtering
    - Raised-cosine crossfades between blocks
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
        self.ratio_smoothing = 0.6  # more aggressive smoothing
        
        # Larger overlap window (60% of 160-sample hop by default)
        self.overlap_size = 96
        self.prev_tail = np.zeros(self.overlap_size, dtype=np.float32)
        self._crossfade_window = self._create_crossfade(self.overlap_size)
        
        # Window cache for block windowing
        self._window_cache = {}
        
        # Anti-aliasing filter (designed on demand)
        self._filter_taps = 65
        self._lowpass_taps = None
        self._lowpass_state = np.zeros(self._filter_taps - 1, dtype=np.float32)
        self._last_filter_ratio = None
    
    def set_pitch_ratio(self, ratio: float):
        """Set target pitch ratio."""
        self._target_ratio = np.clip(ratio, 0.5, 2.0)
    
    def process(self, input_samples: np.ndarray) -> np.ndarray:
        """
        Process samples. Returns EXACTLY len(input_samples) output.
        """
        x = input_samples.astype(np.float32)
        n = len(x)
        
        if n == 0:
            return x
        
        # Smooth pitch ratio to limit jumps
        alpha = self.ratio_smoothing
        self.pitch_ratio = alpha * self._target_ratio + (1 - alpha) * self.pitch_ratio
        ratio = self.pitch_ratio
        
        if abs(ratio - 1.0) < 0.01:
            output = x.copy()
        else:
            # Window input to reduce boundary clicks
            window = self._get_block_window(n)
            x_windowed = x * window
            
            # Anti-aliasing filter for upward shifts
            if ratio > 1.02:
                taps = self._get_lowpass_taps(ratio)
                x_windowed, self._lowpass_state = lfilter(
                    taps,
                    [1.0],
                    x_windowed,
                    zi=self._lowpass_state
                )
            
            # Resample via cubic interpolation
            output = self._resample_block(x_windowed, ratio, n)
        
        # Raised cosine crossfade with previous block tail
        if self.overlap_size > 0 and len(self.prev_tail) == self.overlap_size:
            fade_len = min(self.overlap_size, n)
            fade = self._crossfade_window[:fade_len]
            prev = self.prev_tail[:fade_len]
            output[:fade_len] = (1 - fade) * prev + fade * output[:fade_len]
        
        # Save tail for next block
        if n >= self.overlap_size:
            self.prev_tail = output[-self.overlap_size:].copy()
        
        return output.astype(np.float32)
    
    def reset(self):
        """Reset state."""
        self.pitch_ratio = 1.0
        self._target_ratio = 1.0
        self.prev_tail = np.zeros(self.overlap_size, dtype=np.float32)
        self._lowpass_state = np.zeros(self._filter_taps - 1, dtype=np.float32)
    
    def _resample_block(self, x: np.ndarray, ratio: float, length: int) -> np.ndarray:
        """Resample with Catmull-Rom cubic interpolation."""
        n = len(x)
        if abs(ratio - 1.0) < 0.01:
            return x.copy()
        
        indices = np.arange(length) * ratio
        np.clip(indices, 0, n - 1.001, out=indices)
        
        i0 = np.floor(indices).astype(np.int32)
        frac = (indices - i0).astype(np.float32)
        
        im1 = np.clip(i0 - 1, 0, n - 1)
        i1 = i0
        i2 = np.clip(i0 + 1, 0, n - 1)
        ip2 = np.clip(i0 + 2, 0, n - 1)
        
        x0 = x[im1]
        x1 = x[i1]
        x2 = x[i2]
        x3 = x[ip2]
        
        # Catmull-Rom spline interpolation
        frac2 = frac * frac
        frac3 = frac2 * frac
        
        a = -0.5 * x0 + 1.5 * x1 - 1.5 * x2 + 0.5 * x3
        b = x0 - 2.5 * x1 + 2 * x2 - 0.5 * x3
        c = -0.5 * x0 + 0.5 * x2
        d = x1
        
        return (a * frac3 + b * frac2 + c * frac + d).astype(np.float32)
    
    def _create_crossfade(self, size: int) -> np.ndarray:
        """Raised-cosine (Hann) crossfade window."""
        if size <= 1:
            return np.ones(1, dtype=np.float32)
        fade = np.linspace(0, np.pi, size, dtype=np.float32)
        return 0.5 * (1 - np.cos(fade))
    
    def _get_block_window(self, length: int) -> np.ndarray:
        """Cache Hann windows per block length."""
        if length not in self._window_cache:
            window = np.hanning(length).astype(np.float32)
            # Avoid full attenuation at edges to maintain gain
            window = 0.85 + 0.15 * window
            self._window_cache[length] = window
        return self._window_cache[length]
    
    def _get_lowpass_taps(self, ratio: float) -> np.ndarray:
        """Design/update lowpass taps when pitch shift ratio increases."""
        if (
            self._lowpass_taps is None or
            self._last_filter_ratio is None or
            abs(ratio - self._last_filter_ratio) > 0.05
        ):
            cutoff_hz = 0.5 * self.sample_rate / max(ratio, 1.0)
            cutoff = min(cutoff_hz, 0.45 * self.sample_rate)
            self._lowpass_taps = firwin(
                self._filter_taps,
                cutoff=cutoff,
                window='hann',
                fs=self.sample_rate
            ).astype(np.float32)
            self._lowpass_state = np.zeros(self._filter_taps - 1, dtype=np.float32)
            self._last_filter_ratio = ratio
        return self._lowpass_taps


# Aliases for compatibility
VariableDelayPitchShifter = SmoothPitchShifter
SimplePitchShifter = SmoothPitchShifter
