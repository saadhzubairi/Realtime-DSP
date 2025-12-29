"""Pitch detection and tracking using YIN algorithm."""

import numpy as np
from dataclasses import dataclass


@dataclass
class PitchEstimate:
    """Result of pitch estimation."""
    f0: float
    confidence: float
    is_voiced: bool
    period_samples: float


class YINPitchDetector:
    """YIN algorithm for fundamental frequency estimation."""
    
    def __init__(
        self,
        sample_rate: int,
        frame_size: int,
        f0_min: float = 50.0,
        f0_max: float = 500.0,
        threshold: float = 0.15
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.threshold = threshold
        
        self.tau_min = int(sample_rate / f0_max)
        self.tau_max = min(int(sample_rate / f0_min), frame_size // 2)
        
        self._diff = np.zeros(self.tau_max + 1, dtype=np.float32)
        self._cmndf = np.zeros(self.tau_max + 1, dtype=np.float32)
    
    def _difference_function(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        
        fft_size = 1
        while fft_size < 2 * n:
            fft_size *= 2
        
        x_padded = np.zeros(fft_size)
        x_padded[:n] = x
        X = np.fft.rfft(x_padded)
        r_full = np.fft.irfft(X * np.conj(X))
        r = r_full[:self.tau_max + 1]
        
        x_sq = x ** 2
        cum_sum = np.cumsum(x_sq)
        
        self._diff[0] = 0
        for tau in range(1, min(self.tau_max + 1, n)):
            e1 = cum_sum[n - tau - 1] if n - tau > 0 else 0
            e2 = cum_sum[n - 1] - cum_sum[tau - 1] if tau > 0 else cum_sum[n - 1]
            self._diff[tau] = e1 + e2 - 2 * r[tau]
        
        return self._diff
    
    def _cumulative_mean_normalized_difference(self, d: np.ndarray) -> np.ndarray:
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
        tau = self.tau_min
        
        while tau < self.tau_max:
            if cmndf[tau] < self.threshold:
                while tau + 1 < self.tau_max and cmndf[tau + 1] < cmndf[tau]:
                    tau += 1
                return tau
            tau += 1
        
        return int(np.argmin(cmndf[self.tau_min:self.tau_max + 1])) + self.tau_min
    
    def _parabolic_interpolation(self, cmndf: np.ndarray, tau: int) -> float:
        if tau <= self.tau_min or tau >= self.tau_max:
            return float(tau)
        
        alpha = cmndf[tau - 1]
        beta = cmndf[tau]
        gamma = cmndf[tau + 1]
        
        denominator = 2 * (alpha - 2 * beta + gamma)
        if abs(denominator) < 1e-10:
            return float(tau)
        
        peak = (alpha - gamma) / denominator
        return tau + peak
    
    def estimate(self, frame: np.ndarray) -> PitchEstimate:
        d = self._difference_function(frame)
        cmndf = self._cumulative_mean_normalized_difference(d)
        tau = self._absolute_threshold(cmndf)
        tau_refined = self._parabolic_interpolation(cmndf, tau)
        
        confidence = 1.0 - min(cmndf[tau], 1.0)
        is_voiced = cmndf[tau] < self.threshold
        
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
    """Tracks pitch over time with smoothing."""
    
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
        self._detector = YINPitchDetector(
            sample_rate=sample_rate,
            frame_size=frame_size,
            f0_min=f0_min,
            f0_max=f0_max,
            threshold=threshold
        )
        
        self._median_size = median_filter_size
        self._pitch_history = []
        
        self._f0_values = []
        self._voiced_count = 0
        self._total_count = 0
    
    def process_frame(self, frame: np.ndarray) -> PitchEstimate:
        estimate = self._detector.estimate(frame)
        
        self._total_count += 1
        if estimate.is_voiced:
            self._voiced_count += 1
            self._f0_values.append(estimate.f0)
        
        self._pitch_history.append(estimate.f0)
        if len(self._pitch_history) > self._median_size:
            self._pitch_history.pop(0)
        
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
        self._pitch_history = []
        self._f0_values = []
        self._voiced_count = 0
        self._total_count = 0


def hz_to_log_f0(f0_hz: float, eps: float = 1e-10) -> float:
    return np.log(max(f0_hz, eps))


def log_f0_to_hz(log_f0: float) -> float:
    return np.exp(log_f0)
