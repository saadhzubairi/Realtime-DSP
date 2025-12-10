"""
Phase vocoder based pitch shifter for real-time operation.

Combines STFT overlap-add with phase propagation and lightweight
resampling to deliver smoother pitch shifts than direct resampling.
"""

import numpy as np
from typing import Optional

from .stft import STFTParams, STFT, OverlapAddProcessor


class PhaseVocoderPitchShifter:
    """
    Streaming phase vocoder pitch shifter.
    
    Workflow:
        - Accumulate input samples into overlapping frames
        - Perform STFT and compute instantaneous frequency per bin
        - Scale phase evolution by time-stretch factor (1 / pitch_ratio)
        - Reconstruct time-stretched audio via overlap-add
        - Resample stretched output back to the original duration
    """
    
    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        frame_size: Optional[int] = None,
        fft_size: Optional[int] = None
    ):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.frame_size = frame_size or hop_size * 4
        self.fft_size = fft_size or 1 << (self.frame_size - 1).bit_length()
        
        self.params = STFTParams(
            frame_size=self.frame_size,
            hop_size=self.hop_size,
            fft_size=self.fft_size,
            window_type='hann'
        )
        self.stft = STFT(self.params)
        self._analysis_buffer = OverlapAddProcessor(self.frame_size, self.hop_size)
        self._ola_output = OverlapAddProcessor(self.frame_size, self.hop_size)
        
        self.pitch_ratio = 1.0
        self._target_ratio = 1.0
        
        self._prev_phase = np.zeros(self.stft.n_bins, dtype=np.float32)
        self._phase_acc = np.zeros(self.stft.n_bins, dtype=np.float32)
        self._phase_advance = 2 * np.pi * np.arange(self.stft.n_bins) * self.hop_size / self.fft_size
        self._initialized = False
        
        self._stretched_buffer = np.zeros(0, dtype=np.float32)
        self._resample_pos = 0.0
    
    def set_pitch_ratio(self, ratio: float):
        """Set desired pitch ratio (0.5x - 2x)."""
        self._target_ratio = np.clip(ratio, 0.5, 2.0)
    
    def process(self, samples: np.ndarray) -> np.ndarray:
        """Process a hop of samples and return a hop-sized block."""
        x = samples.astype(np.float32)
        if len(x) == 0:
            return x
        
        ratio = self._target_ratio
        self.pitch_ratio = ratio
        
        # If near unity, bypass for lower latency
        if abs(ratio - 1.0) < 0.02:
            return x.copy()
        
        stretch = 1.0 / ratio
        
        frames_ready = self._analysis_buffer.push_samples(x)
        for _ in range(frames_ready):
            frame = self._analysis_buffer.get_frame()
            self._process_frame(frame, stretch)
        
        output = self._render_output(len(x), ratio)
        if np.allclose(output, 0.0):
            # Fallback to dry if PV still warming up
            return x.copy()
        return output.astype(np.float32)
    
    def reset(self):
        """Reset all buffers and state."""
        self._analysis_buffer.reset()
        self._ola_output.reset()
        self._prev_phase.fill(0)
        self._phase_acc.fill(0)
        self._initialized = False
        self._stretched_buffer = np.zeros(0, dtype=np.float32)
        self._resample_pos = 0.0
    
    def _process_frame(self, frame: np.ndarray, stretch: float):
        """Phase vocoder analysis/synthesis for one frame."""
        spectrum = self.stft.analyze_complex(frame)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        
        if not self._initialized:
            self._prev_phase = phase
            self._phase_acc = phase.copy()
            self._initialized = True
        
        delta = phase - self._prev_phase
        self._prev_phase = phase
        
        # Map phase difference to principal value
        delta -= self._phase_advance
        delta = delta - 2 * np.pi * np.round(delta / (2 * np.pi))
        
        # True frequency
        inst_phase = self._phase_advance + delta
        self._phase_acc += inst_phase * stretch
        
        pv_spectrum = magnitude * np.exp(1j * self._phase_acc)
        frame_resynth = self.stft.synthesize_complex(pv_spectrum)
        
        self._ola_output.add_output_frame(frame_resynth)
        hop_out = max(1, int(round(self.hop_size * stretch)))
        chunk = self._ola_output.get_output_samples(hop_out)
        if chunk is not None and len(chunk) > 0:
            self._stretched_buffer = np.concatenate([self._stretched_buffer, chunk])
    
    def _render_output(self, num_samples: int, ratio: float) -> np.ndarray:
        """Resample stretched buffer back to desired duration."""
        if len(self._stretched_buffer) < 2:
            return np.zeros(num_samples, dtype=np.float32)
        
        output = np.zeros(num_samples, dtype=np.float32)
        for i in range(num_samples):
            pos = self._resample_pos
            idx = int(pos)
            if idx + 1 >= len(self._stretched_buffer):
                break
            frac = pos - idx
            a = self._stretched_buffer[idx]
            b = self._stretched_buffer[idx + 1]
            output[i] = (1 - frac) * a + frac * b
            self._resample_pos += ratio
        
        consumed = int(self._resample_pos)
        if consumed > 0:
            if consumed >= len(self._stretched_buffer):
                self._stretched_buffer = np.zeros(0, dtype=np.float32)
                self._resample_pos = 0.0
            else:
                self._stretched_buffer = self._stretched_buffer[consumed:]
                self._resample_pos -= consumed
        
        return output


# Backwards-compatible alias
PhaseVocoderShifter = PhaseVocoderPitchShifter
