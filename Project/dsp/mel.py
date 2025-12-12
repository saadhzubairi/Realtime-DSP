"""
Mel-spectrogram feature extraction utilities for neural vocoder frontends.
"""

import numpy as np
from scipy.signal import get_window


class MelFeatureExtractor:
    """
    Computes log-mel spectrogram frames from streaming audio.
    
    Designed to feed neural vocoders such as WaveRNN/LPCNet.
    Uses proper overlap-add buffering for streaming operation.
    """
    
    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        fmin: float = 80.0,
        fmax: float = 7600.0
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate / 2
        
        self.window = get_window("hann", n_fft, fftbins=True).astype(np.float32)
        self.mel_filter_bank = self._create_mel_filter()
        
        # Circular buffer for streaming - hold n_fft samples
        self.buffer = np.zeros(n_fft, dtype=np.float32)
        self.buffer_fill = 0  # How many valid samples in buffer
        
        # Previous mel frame for smoothing
        self._prev_mel = None
        self._smoothing_alpha = 0.3
    
    def process_block(self, block: np.ndarray) -> np.ndarray:
        """
        Push a block of audio samples and return most recent mel frame.
        
        Handles variable-sized input blocks properly.
        """
        block = np.asarray(block, dtype=np.float32)
        
        if len(block) == 0:
            # Return previous or zeros
            if self._prev_mel is not None:
                return self._prev_mel
            return np.zeros(self.n_mels, dtype=np.float32)
        
        # Add new samples to buffer
        if len(block) >= self.n_fft:
            # Block is larger than FFT size - just use the last n_fft samples
            self.buffer[:] = block[-self.n_fft:]
            self.buffer_fill = self.n_fft
        else:
            # Shift buffer and add new samples
            shift = len(block)
            self.buffer[:-shift] = self.buffer[shift:]
            self.buffer[-shift:] = block
            self.buffer_fill = min(self.buffer_fill + shift, self.n_fft)
        
        # Compute mel if we have enough samples
        if self.buffer_fill >= self.n_fft // 2:
            mel = self._compute_mel(self.buffer.copy())
            
            # Apply temporal smoothing
            if self._prev_mel is not None:
                mel = (self._smoothing_alpha * mel + 
                       (1 - self._smoothing_alpha) * self._prev_mel)
            
            self._prev_mel = mel.copy()
            return mel
        else:
            # Not enough samples yet
            if self._prev_mel is not None:
                return self._prev_mel
            return np.zeros(self.n_mels, dtype=np.float32)
    
    def _compute_mel(self, frame: np.ndarray) -> np.ndarray:
        frame = frame * self.window
        spectrum = np.fft.rfft(frame)
        power = (np.abs(spectrum) ** 2).astype(np.float32)
        mel = self.mel_filter_bank @ power
        mel = np.maximum(mel, 1e-8)
        log_mel = np.log(mel)
        return log_mel.astype(np.float32)
    
    def _create_mel_filter(self) -> np.ndarray:
        n_fft_bins = self.n_fft // 2 + 1
        mel_min = self._hz_to_mel(self.fmin)
        mel_max = self._hz_to_mel(self.fmax)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin_indices = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        filter_bank = np.zeros((self.n_mels, n_fft_bins), dtype=np.float32)
        for m in range(1, self.n_mels + 1):
            left = bin_indices[m - 1]
            center = bin_indices[m]
            right = bin_indices[m + 1]
            if center == left:
                center += 1
            if right == center:
                right += 1
            for k in range(left, center):
                if 0 <= k < n_fft_bins:
                    filter_bank[m - 1, k] = (k - left) / (center - left)
            for k in range(center, right):
                if 0 <= k < n_fft_bins:
                    filter_bank[m - 1, k] = (right - k) / (right - center)
        return filter_bank
    
    @staticmethod
    def _hz_to_mel(freq: float) -> float:
        return 2595 * np.log10(1 + freq / 700.0)
    
    @staticmethod
    def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700 * (10 ** (mel / 2595.0) - 1)

    def reset(self):
        """Reset internal buffer."""
        self.buffer.fill(0)
        self.buffer_fill = 0
        self._prev_mel = None
