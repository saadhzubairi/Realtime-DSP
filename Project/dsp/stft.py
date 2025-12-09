"""
STFT (Short-Time Fourier Transform) module for real-time processing.
Provides windowing, FFT, and reconstruction utilities.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class STFTParams:
    """Parameters for STFT processing."""
    frame_size: int
    hop_size: int
    fft_size: int
    window_type: str = 'hann'
    
    def __post_init__(self):
        # Create analysis and synthesis windows
        self.analysis_window = self._create_window()
        self.synthesis_window = self._create_synthesis_window()
    
    def _create_window(self) -> np.ndarray:
        """Create the analysis window."""
        if self.window_type == 'hann':
            return np.hanning(self.frame_size).astype(np.float32)
        elif self.window_type == 'hamming':
            return np.hamming(self.frame_size).astype(np.float32)
        elif self.window_type == 'blackman':
            return np.blackman(self.frame_size).astype(np.float32)
        else:
            return np.hanning(self.frame_size).astype(np.float32)
    
    def _create_synthesis_window(self) -> np.ndarray:
        """
        Create synthesis window for perfect reconstruction.
        Uses the COLA (Constant Overlap-Add) principle.
        """
        # For Hann window with 50% overlap, the synthesis window
        # should also be Hann to achieve unity gain
        win = self._create_window()
        
        # Normalize for overlap-add reconstruction
        # Sum of squared windows at each sample should be constant
        overlap_factor = self.frame_size // self.hop_size
        
        # Simple normalization (works for standard overlaps)
        return win / (overlap_factor / 2)


class STFT:
    """
    Real-time STFT processor.
    
    Handles windowing, FFT analysis, and maintains phase continuity
    for overlap-add reconstruction.
    """
    
    def __init__(self, params: STFTParams):
        """
        Initialize STFT processor.
        
        Args:
            params: STFT parameters
        """
        self.params = params
        
        # Pre-compute frequency bins
        self.freq_bins = np.fft.rfftfreq(params.fft_size, 1.0)
        self.n_bins = len(self.freq_bins)
        
        # Phase accumulator for phase vocoder
        self._last_phase = np.zeros(self.n_bins, dtype=np.float32)
        self._phase_advance = 2 * np.pi * np.arange(self.n_bins) * params.hop_size / params.fft_size
    
    def analyze(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze a frame to get magnitude and phase.
        
        Args:
            frame: Input frame (frame_size samples)
            
        Returns:
            Tuple of (magnitude, phase) arrays
        """
        # Apply analysis window
        windowed = frame * self.params.analysis_window
        
        # Zero-pad if FFT size > frame size
        if self.params.fft_size > self.params.frame_size:
            padded = np.zeros(self.params.fft_size, dtype=np.float32)
            padded[:self.params.frame_size] = windowed
            windowed = padded
        
        # Compute FFT
        spectrum = np.fft.rfft(windowed)
        
        # Get magnitude and phase
        magnitude = np.abs(spectrum).astype(np.float32)
        phase = np.angle(spectrum).astype(np.float32)
        
        return magnitude, phase
    
    def synthesize(self, magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """
        Synthesize a frame from magnitude and phase.
        
        Args:
            magnitude: Magnitude spectrum
            phase: Phase spectrum
            
        Returns:
            Synthesized frame (frame_size samples)
        """
        # Reconstruct complex spectrum
        spectrum = magnitude * np.exp(1j * phase)
        
        # Inverse FFT
        frame = np.fft.irfft(spectrum, n=self.params.fft_size)
        
        # Truncate to frame size and apply synthesis window
        frame = frame[:self.params.frame_size].astype(np.float32)
        frame = frame * self.params.synthesis_window
        
        return frame
    
    def analyze_complex(self, frame: np.ndarray) -> np.ndarray:
        """
        Analyze a frame to get complex spectrum.
        
        Args:
            frame: Input frame
            
        Returns:
            Complex spectrum
        """
        windowed = frame * self.params.analysis_window
        
        if self.params.fft_size > self.params.frame_size:
            padded = np.zeros(self.params.fft_size, dtype=np.float32)
            padded[:self.params.frame_size] = windowed
            windowed = padded
        
        return np.fft.rfft(windowed)
    
    def synthesize_complex(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Synthesize from complex spectrum.
        
        Args:
            spectrum: Complex spectrum
            
        Returns:
            Synthesized frame
        """
        frame = np.fft.irfft(spectrum, n=self.params.fft_size)
        frame = frame[:self.params.frame_size].astype(np.float32)
        return frame * self.params.synthesis_window
    
    def get_frequencies(self, sample_rate: int) -> np.ndarray:
        """Get frequency values for each bin."""
        return self.freq_bins * sample_rate


class OverlapAddProcessor:
    """
    Manages overlap-add reconstruction for frame-by-frame processing.
    """
    
    def __init__(self, frame_size: int, hop_size: int):
        """
        Initialize overlap-add processor.
        
        Args:
            frame_size: Size of processing frames
            hop_size: Hop size between frames
        """
        self.frame_size = frame_size
        self.hop_size = hop_size
        
        # Input buffer for accumulating samples
        self._input_buffer = np.zeros(frame_size, dtype=np.float32)
        self._input_count = 0
        
        # Output buffer for overlap-add
        self._output_buffer = np.zeros(frame_size * 2, dtype=np.float32)
        self._output_read_idx = 0
    
    def push_samples(self, samples: np.ndarray) -> int:
        """
        Push input samples and return number of frames ready.
        
        Args:
            samples: Input samples
            
        Returns:
            Number of complete frames available
        """
        n_samples = len(samples)
        frames_ready = 0
        
        for i in range(n_samples):
            self._input_buffer[self._input_count] = samples[i]
            self._input_count += 1
            
            if self._input_count >= self.frame_size:
                frames_ready += 1
                # Shift buffer by hop_size
                self._input_buffer[:-self.hop_size] = self._input_buffer[self.hop_size:]
                self._input_count = self.frame_size - self.hop_size
        
        return frames_ready
    
    def get_frame(self) -> np.ndarray:
        """Get the current input frame."""
        return self._input_buffer.copy()
    
    def add_output_frame(self, frame: np.ndarray):
        """
        Add a processed frame using overlap-add.
        
        Args:
            frame: Processed frame to add
        """
        # Add to output buffer
        self._output_buffer[:self.frame_size] += frame
    
    def get_output_samples(self, n_samples: int) -> Optional[np.ndarray]:
        """
        Get output samples.
        
        Args:
            n_samples: Number of samples to get
            
        Returns:
            Output samples or None if not enough
        """
        # Shift output buffer and return the hop
        output = self._output_buffer[:n_samples].copy()
        self._output_buffer[:-n_samples] = self._output_buffer[n_samples:]
        self._output_buffer[-n_samples:] = 0
        return output
    
    def reset(self):
        """Reset all buffers."""
        self._input_buffer.fill(0)
        self._input_count = 0
        self._output_buffer.fill(0)
        self._output_read_idx = 0


def compute_power_spectrum(magnitude: np.ndarray) -> np.ndarray:
    """Compute power spectrum from magnitude spectrum."""
    return magnitude ** 2


def compute_log_magnitude(magnitude: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    """Compute log magnitude spectrum."""
    return np.log(np.maximum(magnitude, floor))


def magnitude_to_db(magnitude: np.ndarray, ref: float = 1.0, min_db: float = -80.0) -> np.ndarray:
    """Convert magnitude to dB scale."""
    db = 20 * np.log10(np.maximum(magnitude, 1e-10) / ref)
    return np.maximum(db, min_db)


def db_to_magnitude(db: np.ndarray, ref: float = 1.0) -> np.ndarray:
    """Convert dB to magnitude scale."""
    return ref * np.power(10, db / 20)
