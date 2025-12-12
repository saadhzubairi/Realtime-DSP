"""
WORLD Vocoder for high-quality voice transformation.

WORLD is a high-quality vocoder that can:
1. Extract pitch (F0), spectral envelope, and aperiodicity
2. Modify pitch and formants independently  
3. Resynthesize with excellent quality

Much better quality than Griffin-Lim, works well for real-time.
"""

import numpy as np
import pyworld as pw
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class WorldParams:
    """Parameters extracted by WORLD analysis."""
    f0: np.ndarray          # Fundamental frequency contour
    sp: np.ndarray          # Spectral envelope
    ap: np.ndarray          # Aperiodicity
    sample_rate: int        # Sample rate used
    frame_period: float     # Frame period in ms


class WorldVocoder:
    """
    WORLD vocoder for high-quality voice transformation.
    
    WORLD provides excellent quality for pitch shifting and
    formant modification while maintaining natural sound.
    """
    
    def __init__(self, sample_rate: int = 16000, frame_period: float = 5.0):
        """
        Initialize WORLD vocoder.
        
        Args:
            sample_rate: Audio sample rate
            frame_period: Frame period in milliseconds (5.0 = 200 fps)
        """
        self.sample_rate = sample_rate
        self.frame_period = frame_period
        
        # FFT size based on sample rate
        self.fft_size = pw.get_cheaptrick_fft_size(sample_rate)
    
    def analyze(self, audio: np.ndarray) -> WorldParams:
        """
        Analyze audio to extract WORLD parameters.
        
        Args:
            audio: Audio samples (float64, mono)
            
        Returns:
            WorldParams with F0, spectral envelope, and aperiodicity
        """
        # Ensure float64 for WORLD
        audio = audio.astype(np.float64)
        
        # Extract F0 using DIO (fast) + StoneMask (refinement)
        f0, t = pw.dio(audio, self.sample_rate, frame_period=self.frame_period)
        f0 = pw.stonemask(audio, f0, t, self.sample_rate)
        
        # Extract spectral envelope
        sp = pw.cheaptrick(audio, f0, t, self.sample_rate)
        
        # Extract aperiodicity
        ap = pw.d4c(audio, f0, t, self.sample_rate)
        
        return WorldParams(
            f0=f0,
            sp=sp,
            ap=ap,
            sample_rate=self.sample_rate,
            frame_period=self.frame_period
        )
    
    def synthesize(self, params: WorldParams) -> np.ndarray:
        """
        Synthesize audio from WORLD parameters.
        
        Args:
            params: WORLD parameters from analyze()
            
        Returns:
            Synthesized audio as float32
        """
        audio = pw.synthesize(
            params.f0,
            params.sp,
            params.ap,
            params.sample_rate,
            params.frame_period
        )
        return audio.astype(np.float32)
    
    def shift_pitch(
        self,
        params: WorldParams,
        semitones: float
    ) -> WorldParams:
        """
        Shift pitch by semitones.
        
        Args:
            params: WORLD parameters
            semitones: Pitch shift in semitones (+12 = octave up)
            
        Returns:
            Modified WorldParams
        """
        ratio = 2.0 ** (semitones / 12.0)
        
        # Shift F0
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
    
    def shift_formants(
        self,
        params: WorldParams,
        shift_ratio: float
    ) -> WorldParams:
        """
        Shift formants (spectral envelope).
        
        Args:
            params: WORLD parameters
            shift_ratio: Formant shift ratio (>1 = higher, <1 = lower)
            
        Returns:
            Modified WorldParams
        """
        sp = params.sp
        new_sp = np.zeros_like(sp)
        
        freq_bins = sp.shape[1]
        
        for i in range(sp.shape[0]):
            # Shift spectral envelope
            for j in range(freq_bins):
                src_idx = int(j / shift_ratio)
                if 0 <= src_idx < freq_bins:
                    new_sp[i, j] = sp[i, src_idx]
                else:
                    new_sp[i, j] = sp[i, -1]  # Use last value
        
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
        """
        Full voice transformation pipeline.
        
        Args:
            audio: Input audio (mono, float)
            pitch_shift: Pitch shift in semitones
            formant_shift: Formant shift ratio
            
        Returns:
            Transformed audio
        """
        # Analyze
        params = self.analyze(audio)
        
        # Apply pitch shift
        if abs(pitch_shift) > 0.01:
            params = self.shift_pitch(params, pitch_shift)
        
        # Apply formant shift
        if abs(formant_shift - 1.0) > 0.01:
            params = self.shift_formants(params, formant_shift)
        
        # Synthesize
        output = self.synthesize(params)
        
        # Match length
        if len(output) > len(audio):
            output = output[:len(audio)]
        elif len(output) < len(audio):
            output = np.pad(output, (0, len(audio) - len(output)))
        
        return output


class RealtimeWorldVocoder:
    """
    Real-time wrapper for WORLD vocoder with overlap-add.
    
    Processes audio in blocks with crossfading for smooth output.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        block_size: int = 1600,  # 100ms at 16kHz
        overlap: float = 0.5
    ):
        """
        Initialize real-time WORLD vocoder.
        
        Args:
            sample_rate: Audio sample rate
            block_size: Processing block size in samples
            overlap: Overlap ratio for crossfading (0.0-0.75)
        """
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.overlap = overlap
        self.hop_size = int(block_size * (1 - overlap))
        
        # WORLD vocoder
        self.vocoder = WorldVocoder(sample_rate, frame_period=5.0)
        
        # Buffers
        self.input_buffer = np.zeros(block_size, dtype=np.float32)
        self.output_buffer = np.zeros(block_size, dtype=np.float32)
        self.prev_output = np.zeros(block_size, dtype=np.float32)
        
        # Crossfade window
        self.fade_in = np.linspace(0, 1, int(block_size * overlap), dtype=np.float32)
        self.fade_out = np.linspace(1, 0, int(block_size * overlap), dtype=np.float32)
        
        # Transform settings
        self.pitch_shift = 0.0
        self.formant_shift = 1.0
        self.enabled = True
        
        # Stats
        self._process_time_ms = 0.0
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio block with voice transformation.
        
        Args:
            audio: Input audio block
            
        Returns:
            Transformed audio block
        """
        import time
        start = time.perf_counter()
        
        if not self.enabled:
            self._process_time_ms = 0.0
            return audio.copy()
        
        # Ensure float32
        audio = audio.astype(np.float32)
        
        # Transform
        output = self.vocoder.transform_voice(
            audio,
            pitch_shift=self.pitch_shift,
            formant_shift=self.formant_shift
        )
        
        # Apply crossfade with previous output
        overlap_samples = len(self.fade_in)
        if overlap_samples > 0 and len(self.prev_output) >= overlap_samples:
            output[:overlap_samples] = (
                output[:overlap_samples] * self.fade_in +
                self.prev_output[-overlap_samples:] * self.fade_out
            )
        
        # Store for next crossfade
        self.prev_output = output.copy()
        
        self._process_time_ms = (time.perf_counter() - start) * 1000
        return output
    
    def set_pitch_shift(self, semitones: float):
        """Set pitch shift in semitones."""
        self.pitch_shift = semitones
    
    def set_formant_shift(self, ratio: float):
        """Set formant shift ratio."""
        self.formant_shift = ratio
    
    def set_enabled(self, enabled: bool):
        """Enable/disable transformation."""
        self.enabled = enabled
    
    def reset(self):
        """Reset buffers."""
        self.prev_output = np.zeros(self.block_size, dtype=np.float32)
    
    @property
    def process_time_ms(self) -> float:
        """Get last processing time in milliseconds."""
        return self._process_time_ms
