"""
Librosa-powered phase vocoder pitch shifter.

Uses librosa.effects.pitch_shift with rolling context to provide higher-quality
pitch conversion than simple resampling while remaining relatively low-latency.
"""

import numpy as np
import librosa

try:
    import resampy  # noqa: F401  (ensure dependency is available)
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "PhaseVocoderPitchShifter requires the 'resampy' package. "
        "Install it via 'pip install resampy'."
    ) from exc

from utils.logging_utils import dsp_logger


class PhaseVocoderPitchShifter:
    """Streaming wrapper around librosa's phase-vocoder pitch shifting."""
    
    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        frame_size: int,
    ):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.context_size = max(frame_size * 2, hop_size * 8)
        self._context = np.zeros(self.context_size, dtype=np.float32)
        
        self.pitch_ratio = 1.0
        self._target_ratio = 1.0
        self._last_ratio = 1.0
        self._fast_alpha = 0.7
        self._slow_alpha = 0.45
        self._fast_threshold = 0.07  # log-domain delta for fast smoothing
        self._dead_zone = 0.01
        self._librosa_failed = False
    
    def set_pitch_ratio(self, ratio: float):
        self._target_ratio = float(np.clip(ratio, 0.5, 2.0))
    
    def process(self, samples: np.ndarray) -> np.ndarray:
        x = samples.astype(np.float32)
        if len(x) == 0:
            return x
        
        ratio = self._smooth_ratio()
        
        if self._librosa_failed:
            return x.copy()
        
        if abs(ratio - 1.0) < self._dead_zone:
            self._update_context(x)
            return x.copy()
        
        n_steps = self._ratio_to_semitones(ratio)
        outputs = []
        idx = 0
        
        while idx < len(x):
            block = x[idx:idx + self.hop_size]
            block_len = len(block)
            if block_len < self.hop_size:
                block = np.pad(block, (0, self.hop_size - block_len))
            
            window = np.concatenate([self._context, block])
            try:
                shifted = librosa.effects.pitch_shift(
                    window,
                    sr=self.sample_rate,
                    n_steps=n_steps,
                    res_type="kaiser_best"
                ).astype(np.float32)
            except Exception as exc:  # pylint: disable=broad-except
                dsp_logger.error(f"Librosa pitch_shift failed: {exc}")
                self._librosa_failed = True
                return x.copy()
            
            out_block = shifted[-self.hop_size:]
            outputs.append(out_block[:block_len])
            
            self._update_context(block)
            idx += block_len
        
        return np.concatenate(outputs) if outputs else x
    
    def reset(self):
        self._context = np.zeros(self.context_size, dtype=np.float32)
        self.pitch_ratio = 1.0
        self._target_ratio = 1.0
        self._last_ratio = 1.0
        self._librosa_failed = False
    
    def _update_context(self, block: np.ndarray):
        """Maintain rolling context for future blocks."""
        self._context = np.concatenate([self._context, block])[-self.context_size:]
    
    def _smooth_ratio(self) -> float:
        target = max(self._target_ratio, 1e-6)
        log_target = np.log(target)
        log_current = np.log(max(self._last_ratio, 1e-6))
        delta = abs(log_target - log_current)
        alpha = self._fast_alpha if delta > self._fast_threshold else self._slow_alpha
        log_ratio = alpha * log_target + (1 - alpha) * log_current
        self._last_ratio = float(np.exp(log_ratio))
        return self._last_ratio
    
    @staticmethod
    def _ratio_to_semitones(ratio: float) -> float:
        return 12.0 * np.log2(max(ratio, 1e-6))


# Backwards-compatible alias
PhaseVocoderShifter = PhaseVocoderPitchShifter
