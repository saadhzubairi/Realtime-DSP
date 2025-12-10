"""
Pitch detection and tracking module.
Implements YIN algorithm and related utilities for real-time F0 estimation.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from scipy import signal


@dataclass
class PitchEstimate:
    """Result of pitch estimation."""
    f0: float              # Fundamental frequency in Hz (0 if unvoiced)
    confidence: float      # Confidence/harmonicity (0-1)
    is_voiced: bool        # Whether the frame is voiced
    period_samples: float  # Period in samples


class YINPitchDetector:
    """
    YIN algorithm for fundamental frequency estimation.
    
    Based on: "YIN, a fundamental frequency estimator for speech and music"
    by A. de Cheveigné and H. Kawahara (2002)
    """
    
    def __init__(
        self,
        sample_rate: int,
        frame_size: int,
        f0_min: float = 50.0,
        f0_max: float = 500.0,
        threshold: float = 0.15
    ):
        """
        Initialize YIN pitch detector.
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_size: Size of analysis frame in samples
            f0_min: Minimum F0 to detect (Hz)
            f0_max: Maximum F0 to detect (Hz)
            threshold: Harmonicity threshold for voiced decision
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.threshold = threshold
        
        # Compute lag range from F0 limits
        self.tau_min = int(sample_rate / f0_max)
        self.tau_max = min(int(sample_rate / f0_min), frame_size // 2)
        
        # Pre-allocate buffers
        self._diff = np.zeros(self.tau_max + 1, dtype=np.float32)
        self._cmndf = np.zeros(self.tau_max + 1, dtype=np.float32)
    
    def _difference_function(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the difference function d_t(tau) using FFT-based autocorrelation.
        
        d_t(tau) = sum_{j=0}^{W-1} (x_j - x_{j+tau})^2
        
        Uses: d(tau) = r(0) + r_shifted(0) - 2*r(tau)
        where r is autocorrelation computed via FFT.
        """
        n = len(x)
        
        # FFT-based autocorrelation (much faster than O(n^2) loops)
        # Pad to next power of 2 for efficiency
        fft_size = 1
        while fft_size < 2 * n:
            fft_size *= 2
        
        # Compute autocorrelation via FFT
        x_padded = np.zeros(fft_size)
        x_padded[:n] = x
        X = np.fft.rfft(x_padded)
        r_full = np.fft.irfft(X * np.conj(X))
        r = r_full[:self.tau_max + 1]
        
        # Compute cumulative energy for shifted windows
        x_sq = x ** 2
        cum_sum = np.cumsum(x_sq)
        
        # d(tau) = r(0) + r_shifted(0) - 2*r(tau)
        # r(0) = sum(x^2) for first window
        # r_shifted(0) = sum(x[tau:]^2) for shifted window
        self._diff[0] = 0
        for tau in range(1, min(self.tau_max + 1, n)):
            # Energy of x[0:n-tau] and x[tau:n]
            e1 = cum_sum[n - tau - 1] if n - tau > 0 else 0
            e2 = cum_sum[n - 1] - cum_sum[tau - 1] if tau > 0 else cum_sum[n - 1]
            self._diff[tau] = e1 + e2 - 2 * r[tau]
        
        return self._diff
    
    def _cumulative_mean_normalized_difference(self, d: np.ndarray) -> np.ndarray:
        """
        Compute cumulative mean normalized difference function.
        
        d'(tau) = d(tau) / ((1/tau) * sum_{j=1}^{tau} d(j))
        """
        self._cmndf[0] = 1.0
        
        running_sum = 0.0
        for tau in range(1, self.tau_max + 1):
            running_sum += d[tau]
            if running_sum == 0:
                self._cmndf[tau] = 1.0
            else:
                self._cmndf[tau] = d[tau] / (running_sum / tau)
        
        return self._cmndf
    
    def _absolute_threshold(self, cmndf: np.ndarray) -> int:
        """
        Find the first minimum below threshold (absolute threshold step).
        
        Returns:
            Estimated period in samples, or 0 if not found
        """
        tau = self.tau_min
        
        while tau < self.tau_max:
            if cmndf[tau] < self.threshold:
                # Find local minimum
                while tau + 1 < self.tau_max and cmndf[tau + 1] < cmndf[tau]:
                    tau += 1
                return tau
            tau += 1
        
        # No pitch found - return the global minimum
        return int(np.argmin(cmndf[self.tau_min:self.tau_max + 1])) + self.tau_min
    
    def _parabolic_interpolation(self, cmndf: np.ndarray, tau: int) -> float:
        """
        Refine period estimate using parabolic interpolation.
        
        Args:
            cmndf: Cumulative mean normalized difference function
            tau: Initial period estimate
            
        Returns:
            Refined period estimate
        """
        if tau <= self.tau_min or tau >= self.tau_max:
            return float(tau)
        
        # Parabolic interpolation using three points
        alpha = cmndf[tau - 1]
        beta = cmndf[tau]
        gamma = cmndf[tau + 1]
        
        denominator = 2 * (alpha - 2 * beta + gamma)
        if abs(denominator) < 1e-10:
            return float(tau)
        
        peak = (alpha - gamma) / denominator
        return tau + peak
    
    def estimate(self, frame: np.ndarray) -> PitchEstimate:
        """
        Estimate pitch from a single frame.
        
        Args:
            frame: Audio frame (frame_size samples)
            
        Returns:
            PitchEstimate with F0, confidence, and voiced decision
        """
        # Step 1: Difference function
        d = self._difference_function(frame)
        
        # Step 2: Cumulative mean normalized difference
        cmndf = self._cumulative_mean_normalized_difference(d)
        
        # Step 3: Absolute threshold
        tau = self._absolute_threshold(cmndf)
        
        # Step 4: Parabolic interpolation
        tau_refined = self._parabolic_interpolation(cmndf, tau)
        
        # Get confidence (inverse of CMNDF value at the detected period)
        confidence = 1.0 - min(cmndf[tau], 1.0)
        
        # Voiced decision
        is_voiced = cmndf[tau] < self.threshold
        
        # Calculate F0
        if is_voiced and tau_refined > 0:
            f0 = self.sample_rate / tau_refined
        else:
            f0 = 0.0
        
        return PitchEstimate(
            f0=f0,
            confidence=confidence,
            is_voiced=is_voiced,
            period_samples=tau_refined
        )


class PitchTracker:
    """
    Tracks pitch over time with smoothing and hysteresis.
    """
    
    def __init__(
        self,
        sample_rate: int,
        frame_size: int,
        hop_size: int,
        f0_min: float = 50.0,
        f0_max: float = 500.0,
        threshold: float = 0.15,
        median_filter_size: int = 5
    ):
        """
        Initialize pitch tracker.
        
        Args:
            sample_rate: Audio sample rate
            frame_size: Analysis frame size
            hop_size: Hop size between frames
            f0_min: Minimum F0
            f0_max: Maximum F0
            threshold: Voiced threshold
            median_filter_size: Size of median filter for smoothing
        """
        self._detector = YINPitchDetector(
            sample_rate=sample_rate,
            frame_size=frame_size,
            f0_min=f0_min,
            f0_max=f0_max,
            threshold=threshold
        )
        
        self._median_size = median_filter_size
        self._pitch_history = []
        
        # Statistics
        self._f0_values = []
        self._voiced_count = 0
        self._total_count = 0
    
    def process_frame(self, frame: np.ndarray) -> PitchEstimate:
        """
        Process a frame and return smoothed pitch estimate.
        
        Args:
            frame: Audio frame
            
        Returns:
            Smoothed pitch estimate
        """
        # Get raw estimate
        estimate = self._detector.estimate(frame)
        
        # Update statistics
        self._total_count += 1
        if estimate.is_voiced:
            self._voiced_count += 1
            self._f0_values.append(estimate.f0)
        
        # Add to history for median filtering
        self._pitch_history.append(estimate.f0)
        if len(self._pitch_history) > self._median_size:
            self._pitch_history.pop(0)
        
        # Apply median filter (only over voiced frames)
        voiced_history = [f for f in self._pitch_history if f > 0]
        if len(voiced_history) >= 3:
            smoothed_f0 = np.median(voiced_history)
        else:
            smoothed_f0 = estimate.f0
        
        return PitchEstimate(
            f0=smoothed_f0 if estimate.is_voiced else 0.0,
            confidence=estimate.confidence,
            is_voiced=estimate.is_voiced,
            period_samples=estimate.period_samples
        )
    
    def get_statistics(self) -> dict:
        """Get accumulated pitch statistics."""
        if not self._f0_values:
            return {
                'f0_median': 0.0,
                'f0_p05': 0.0,
                'f0_p95': 0.0,
                'voiced_ratio': 0.0
            }
        
        f0_array = np.array(self._f0_values)
        
        return {
            'f0_median': float(np.median(f0_array)),
            'f0_p05': float(np.percentile(f0_array, 5)),
            'f0_p95': float(np.percentile(f0_array, 95)),
            'voiced_ratio': self._voiced_count / max(self._total_count, 1)
        }
    
    def reset(self):
        """Reset tracker state."""
        self._pitch_history = []
        self._f0_values = []
        self._voiced_count = 0
        self._total_count = 0


def hz_to_log_f0(f0_hz: float, eps: float = 1e-10) -> float:
    """Convert F0 in Hz to log scale."""
    return np.log(max(f0_hz, eps))


def log_f0_to_hz(log_f0: float) -> float:
    """Convert log F0 to Hz."""
    return np.exp(log_f0)


def hz_to_semitones(f0_hz: float, reference_hz: float = 440.0) -> float:
    """Convert F0 in Hz to semitones relative to reference."""
    if f0_hz <= 0:
        return 0.0
    return 12 * np.log2(f0_hz / reference_hz)


def semitones_to_hz(semitones: float, reference_hz: float = 440.0) -> float:
    """Convert semitones to Hz."""
    return reference_hz * np.power(2, semitones / 12)


def map_pitch_linear(
    f0_source: float,
    source_median: float,
    source_std: float,
    target_median: float,
    target_std: float,
    strength: float = 1.0
) -> float:
    """
    Map pitch from source to target distribution.
    
    Works in log-F0 space for natural scaling.
    
    Args:
        f0_source: Source F0 in Hz
        source_median: Source F0 median (Hz)
        source_std: Source F0 std in log domain
        target_median: Target F0 median (Hz)
        target_std: Target F0 std in log domain
        strength: Mapping strength (0-1)
        
    Returns:
        Mapped F0 in Hz
    """
    if f0_source <= 0:
        return 0.0
    
    # Convert to log domain
    log_f0 = np.log(f0_source)
    log_source_median = np.log(source_median)
    log_target_median = np.log(target_median)
    
    # Linear mapping in log domain
    # p' = μB + (σB/σA) * (p - μA)
    if source_std > 0:
        log_f0_mapped = log_target_median + (target_std / source_std) * (log_f0 - log_source_median)
    else:
        log_f0_mapped = log_target_median + (log_f0 - log_source_median)
    
    # Blend with original based on strength
    log_f0_out = (1 - strength) * log_f0 + strength * log_f0_mapped
    
    return np.exp(log_f0_out)
