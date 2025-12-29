"""WORLD Vocoder for high-quality voice transformation."""

import numpy as np
import pyworld as pw
from dataclasses import dataclass


@dataclass
class WorldParams:
    """Parameters extracted by WORLD analysis."""
    f0: np.ndarray
    sp: np.ndarray
    ap: np.ndarray
    sample_rate: int
    frame_period: float


class WorldVocoder:
    """WORLD vocoder for high-quality voice transformation."""
    
    def __init__(self, sample_rate: int = 16000, frame_period: float = 5.0):
        self.sample_rate = sample_rate
        self.frame_period = frame_period
        self.fft_size = pw.get_cheaptrick_fft_size(sample_rate)
    
    def analyze(self, audio: np.ndarray) -> WorldParams:
        """Analyze audio to extract WORLD parameters."""
        audio = audio.astype(np.float64)
        
        f0, t = pw.dio(audio, self.sample_rate, frame_period=self.frame_period)
        f0 = pw.stonemask(audio, f0, t, self.sample_rate)
        
        sp = pw.cheaptrick(audio, f0, t, self.sample_rate)
        ap = pw.d4c(audio, f0, t, self.sample_rate)
        
        return WorldParams(
            f0=f0,
            sp=sp,
            ap=ap,
            sample_rate=self.sample_rate,
            frame_period=self.frame_period
        )
    
    def synthesize(self, params: WorldParams) -> np.ndarray:
        """Synthesize audio from WORLD parameters."""
        audio = pw.synthesize(
            params.f0,
            params.sp,
            params.ap,
            params.sample_rate,
            params.frame_period
        )
        return audio.astype(np.float32)
    
    def shift_pitch(self, params: WorldParams, semitones: float) -> WorldParams:
        """Shift pitch by semitones."""
        ratio = 2.0 ** (semitones / 12.0)
        
        new_f0 = params.f0.copy()
        voiced = new_f0 > 0
        new_f0[voiced] = new_f0[voiced] * ratio
        
        return WorldParams(
            f0=new_f0,
            sp=params.sp,
            ap=params.ap,
            sample_rate=params.sample_rate,
            frame_period=params.frame_period
        )
    
    def shift_formants(self, params: WorldParams, shift_ratio: float) -> WorldParams:
        """Shift formants (spectral envelope)."""
        sp = params.sp
        new_sp = np.zeros_like(sp)
        
        freq_bins = sp.shape[1]
        
        for i in range(sp.shape[0]):
            for j in range(freq_bins):
                src_idx = int(j / shift_ratio)
                if 0 <= src_idx < freq_bins:
                    new_sp[i, j] = sp[i, src_idx]
                else:
                    new_sp[i, j] = sp[i, -1]
        
        return WorldParams(
            f0=params.f0,
            sp=new_sp,
            ap=params.ap,
            sample_rate=params.sample_rate,
            frame_period=params.frame_period
        )
    
    def transform_voice(
        self,
        audio: np.ndarray,
        pitch_shift: float = 0.0,
        formant_shift: float = 1.0
    ) -> np.ndarray:
        """Full voice transformation pipeline."""
        params = self.analyze(audio)
        
        if abs(pitch_shift) > 0.01:
            params = self.shift_pitch(params, pitch_shift)
        
        if abs(formant_shift - 1.0) > 0.01:
            params = self.shift_formants(params, formant_shift)
        
        output = self.synthesize(params)
        
        if len(output) > len(audio):
            output = output[:len(audio)]
        elif len(output) < len(audio):
            output = np.pad(output, (0, len(audio) - len(output)))
        
        return output


class RealtimeWorldVocoder:
    """Real-time wrapper for WORLD vocoder with overlap-add."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        block_size: int = 1600,
        overlap: float = 0.5
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.overlap = overlap
        self.hop_size = int(block_size * (1 - overlap))
        
        self.vocoder = WorldVocoder(sample_rate, frame_period=5.0)
        
        self.input_buffer = np.zeros(block_size, dtype=np.float32)
        self.output_buffer = np.zeros(block_size, dtype=np.float32)
        self.prev_output = np.zeros(block_size, dtype=np.float32)
        
        self.fade_in = np.linspace(0, 1, int(block_size * overlap), dtype=np.float32)
        self.fade_out = np.linspace(1, 0, int(block_size * overlap), dtype=np.float32)
        
        self.pitch_shift = 0.0
        self.formant_shift = 1.0
        self.enabled = True
        
        self._process_time_ms = 0.0
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process audio block with voice transformation."""
        import time
        start = time.perf_counter()
        
        if not self.enabled:
            self._process_time_ms = 0.0
            return audio.copy()
        
        audio = audio.astype(np.float32)
        
        output = self.vocoder.transform_voice(
            audio,
            pitch_shift=self.pitch_shift,
            formant_shift=self.formant_shift
        )
        
        overlap_samples = len(self.fade_in)
        if overlap_samples > 0 and len(self.prev_output) >= overlap_samples:
            output[:overlap_samples] = (
                output[:overlap_samples] * self.fade_in +
                self.prev_output[-overlap_samples:] * self.fade_out
            )
        
        self.prev_output = output.copy()
        
        self._process_time_ms = (time.perf_counter() - start) * 1000
        return output
    
    def set_pitch_shift(self, semitones: float):
        self.pitch_shift = semitones
    
    def set_formant_shift(self, ratio: float):
        self.formant_shift = ratio
    
    def set_enabled(self, enabled: bool):
        self.enabled = enabled
    
    def reset(self):
        self.prev_output = np.zeros(self.block_size, dtype=np.float32)
    
    @property
    def process_time_ms(self) -> float:
        return self._process_time_ms
