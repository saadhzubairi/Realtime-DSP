"""
PSOLA (Pitch-Synchronous Overlap-Add) pitch shifter.

Leverages YIN pitch detection to extract pitch periods and reposition
waveform segments to match a target pitch ratio.
"""

import numpy as np
from typing import Optional

from .pitch import YINPitchDetector
from utils.config import F0_MIN, F0_MAX


class PSOLAPitchShifter:
    """Streaming PSOLA pitch shifter."""
    
    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        frame_size: int,
        f0_min: float = F0_MIN,
        f0_max: float = F0_MAX
    ):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.analysis_size = max(frame_size * 2, int(sample_rate * 0.04))
        self.detector = YINPitchDetector(
            sample_rate=sample_rate,
            frame_size=self.analysis_size,
            f0_min=f0_min,
            f0_max=f0_max
        )
        self.analysis_frame = np.zeros(self.analysis_size, dtype=np.float32)
        
        self.min_period = int(sample_rate / f0_max)
        self.max_period = int(sample_rate / max(f0_min, 1.0))
        self.prev_period = max(self.min_period, int(sample_rate / 150))
        
        self.pitch_ratio = 1.0
        self._target_ratio = 1.0
        self._last_ratio = 1.0
        self._fast_alpha = 0.7
        self._slow_alpha = 0.4
        self._dead_zone = 0.02
        self._fast_threshold = 0.08
        
        self.prev_tail = np.zeros(self.hop_size, dtype=np.float32)
    
    def set_pitch_ratio(self, ratio: float):
        self._target_ratio = float(np.clip(ratio, 0.5, 2.0))
    
    def process(self, samples: np.ndarray) -> np.ndarray:
        x = samples.astype(np.float32)
        if len(x) == 0:
            return x
        
        self._update_analysis_frame(x)
        estimate = self.detector.estimate(self.analysis_frame * np.hanning(self.analysis_size))
        
        if not estimate.is_voiced or estimate.f0 <= 0:
            output = x.copy()
        else:
            period = self._f0_to_period(estimate.f0)
            ratio = self._smooth_ratio()
            if abs(ratio - 1.0) < self._dead_zone:
                output = x.copy()
            else:
                output = self._psola_synthesize(x, period, ratio)
                self.prev_period = period
        
        output = self._blend_with_tail(output)
        self.prev_tail = output[-self.hop_size:].copy()
        return output.astype(np.float32)
    
    def reset(self):
        self.analysis_frame.fill(0)
        self.prev_period = max(self.min_period, int(self.sample_rate / 150))
        self.pitch_ratio = 1.0
        self._target_ratio = 1.0
        self._last_ratio = 1.0
        self.prev_tail = np.zeros(self.hop_size, dtype=np.float32)
    
    def _update_analysis_frame(self, block: np.ndarray):
        shift = len(block)
        if shift >= self.analysis_size:
            self.analysis_frame[:] = block[-self.analysis_size:]
        else:
            self.analysis_frame[:-shift] = self.analysis_frame[shift:]
            self.analysis_frame[-shift:] = block
    
    def _f0_to_period(self, f0: float) -> int:
        period = int(round(self.sample_rate / max(f0, 1e-3)))
        return int(np.clip(period, self.min_period, self.max_period))
    
    def _smooth_ratio(self) -> float:
        target = max(self._target_ratio, 1e-6)
        log_target = np.log(target)
        log_current = np.log(max(self._last_ratio, 1e-6))
        delta = abs(log_target - log_current)
        alpha = self._fast_alpha if delta > self._fast_threshold else self._slow_alpha
        log_ratio = alpha * log_target + (1 - alpha) * log_current
        self._last_ratio = float(np.exp(log_ratio))
        return self._last_ratio
    
    def _psola_synthesize(self, block: np.ndarray, period: int, ratio: float) -> np.ndarray:
        window_len = max(period * 2, 4)
        half = window_len // 2
        window = np.hanning(window_len).astype(np.float32)
        padded = np.pad(block, (half, half), mode='reflect')
        
        n = len(block)
        analysis_centers = np.arange(half, half + n, period)
        if len(analysis_centers) == 0:
            analysis_centers = np.array([half + n // 2])
        
        target_period = max(int(round(period / ratio)), 1)
        synth_centers = []
        pos = analysis_centers[0]
        while pos < half + n:
            synth_centers.append(int(pos))
            pos += target_period
        if not synth_centers:
            synth_centers = [analysis_centers[0]]
        
        # Map synthesis centers to nearest analysis centers
        analysis_positions = np.linspace(
            0,
            len(analysis_centers) - 1,
            num=len(synth_centers)
        )
        analysis_indices = np.clip(
            np.round(analysis_positions).astype(int),
            0,
            len(analysis_centers) - 1
        )
        
        output = np.zeros(len(padded), dtype=np.float32)
        for synth, idx in zip(synth_centers, analysis_indices):
            center = analysis_centers[idx]
            start = center - half
            segment = padded[start:start + window_len].copy()
            seg_len = len(segment)
            w = window[:seg_len]
            dest_start = synth - half
            if dest_start < 0:
                segment = segment[-dest_start:]
                w = w[-dest_start:]
                dest_start = 0
            dest_end = dest_start + len(segment)
            if dest_end > len(output):
                clip = dest_end - len(output)
                if clip >= len(segment):
                    continue
                segment = segment[:-clip]
                w = w[:-clip]
                dest_end = len(output)
            output[dest_start:dest_start + len(w)] += segment * w
        
        return output[half:half + n]
    
    def _blend_with_tail(self, block: np.ndarray) -> np.ndarray:
        if len(self.prev_tail) == 0:
            return block
        fade_len = min(len(self.prev_tail), len(block))
        if fade_len == 0:
            return block
        fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
        fade_out = 1 - fade_in
        block[:fade_len] = (
            fade_out * self.prev_tail[:fade_len] +
            fade_in * block[:fade_len]
        )
        return block


# Alias for compatibility
PSOLA_PitchShifter = PSOLAPitchShifter
