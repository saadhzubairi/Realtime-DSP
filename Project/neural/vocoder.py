"""
Neural vocoder implementations for real-time synthesis.

Provides multiple backends:
1. GriffinLimVocoder - Fast iterative mel-to-waveform (CPU friendly, real-time capable)
2. WaveRNNVocoder - Higher quality but slower (optional, requires torch)
3. HybridVocoder - Combines both with automatic fallback
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
import threading
import queue
from dataclasses import dataclass

# Try to import torch for optional WaveRNN support
try:
    import torch
    import torchaudio
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    torchaudio = None
    _TORCH_AVAILABLE = False


@dataclass
class VocoderConfig:
    """Configuration for vocoder synthesis."""
    sample_rate: int = 16000
    hop_length: int = 160
    n_fft: int = 1024
    n_mels: int = 80
    fmin: float = 80.0
    fmax: float = 7600.0
    griffin_lim_iters: int = 16  # Fast but decent quality


class GriffinLimVocoder:
    """
    Griffin-Lim based vocoder for real-time mel-to-waveform synthesis.
    
    Uses iterative phase estimation to convert mel spectrograms to audio.
    Much faster than autoregressive vocoders and suitable for real-time use.
    """
    
    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        n_fft: int = 1024,
        n_mels: int = 80,
        fmin: float = 80.0,
        fmax: float = 7600.0,
        n_iter: int = 16
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.n_iter = n_iter
        self.fmin = fmin
        self.fmax = fmax or sample_rate / 2
        
        # Pre-compute mel filterbank and its pseudo-inverse
        self.mel_basis = self._create_mel_filterbank()
        self.mel_basis_pinv = np.linalg.pinv(self.mel_basis)
        
        # Synthesis window
        self.window = np.hanning(n_fft).astype(np.float32)
        
        # State for overlap-add synthesis
        self.mel_buffer = []
        self.max_mel_frames = 8  # Keep small buffer for low latency
        self.prev_phase = None
        self.overlap_buffer = np.zeros(n_fft, dtype=np.float32)
        
        # Crossfade for smooth output
        self.fade_len = hop_length // 4
        self.fade_in = np.linspace(0, 1, self.fade_len, dtype=np.float32)
        self.fade_out = 1 - self.fade_in
        self.prev_tail = np.zeros(self.fade_len, dtype=np.float32)
    
    def _create_mel_filterbank(self) -> np.ndarray:
        """Create mel filterbank matrix."""
        n_fft_bins = self.n_fft // 2 + 1
        
        def hz_to_mel(f):
            return 2595 * np.log10(1 + f / 700.0)
        
        def mel_to_hz(m):
            return 700 * (10 ** (m / 2595.0) - 1)
        
        mel_min = hz_to_mel(self.fmin)
        mel_max = hz_to_mel(self.fmax)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_indices = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        filter_bank = np.zeros((self.n_mels, n_fft_bins), dtype=np.float32)
        for m in range(1, self.n_mels + 1):
            left = bin_indices[m - 1]
            center = bin_indices[m]
            right = bin_indices[m + 1]
            
            if center <= left:
                center = left + 1
            if right <= center:
                right = center + 1
            
            for k in range(left, center):
                if 0 <= k < n_fft_bins:
                    filter_bank[m - 1, k] = (k - left) / max(center - left, 1)
            for k in range(center, right):
                if 0 <= k < n_fft_bins:
                    filter_bank[m - 1, k] = (right - k) / max(right - center, 1)
        
        return filter_bank
    
    def _mel_to_linear(self, log_mel: np.ndarray) -> np.ndarray:
        """Convert log-mel spectrogram back to linear magnitude spectrum."""
        # Undo log
        mel_power = np.exp(log_mel)
        
        # Invert mel filterbank (pseudo-inverse)
        linear_power = np.maximum(self.mel_basis_pinv @ mel_power, 0)
        
        # Return magnitude
        return np.sqrt(linear_power)
    
    def _griffin_lim_frame(self, magnitude: np.ndarray) -> np.ndarray:
        """
        Apply Griffin-Lim algorithm for a single frame.
        Uses previous phase as initialization for faster convergence.
        """
        if self.prev_phase is None:
            # Initialize with random phase
            phase = np.exp(2j * np.pi * np.random.random(len(magnitude)))
        else:
            # Use previous phase as starting point
            phase = self.prev_phase
        
        # Iterative refinement
        for _ in range(self.n_iter):
            # Combine magnitude and phase
            stft = magnitude * phase
            
            # Inverse FFT
            signal = np.fft.irfft(stft, n=self.n_fft)
            
            # Apply window
            signal = signal * self.window
            
            # Forward FFT
            stft = np.fft.rfft(signal)
            
            # Extract phase
            phase = np.exp(1j * np.angle(stft))
        
        # Store phase for next frame
        self.prev_phase = phase
        
        # Final synthesis
        stft = magnitude * phase
        signal = np.fft.irfft(stft, n=self.n_fft)
        
        return signal.astype(np.float32)
    
    def synthesize(self, log_mel: np.ndarray) -> np.ndarray:
        """
        Synthesize audio from a log-mel frame.
        
        Args:
            log_mel: Log-mel spectrogram frame (n_mels,)
            
        Returns:
            Audio samples (hop_length,)
        """
        # Ensure correct shape
        if log_mel.ndim > 1:
            log_mel = log_mel.flatten()
        
        if len(log_mel) != self.n_mels:
            # Pad or truncate
            if len(log_mel) < self.n_mels:
                padded = np.zeros(self.n_mels, dtype=np.float32)
                padded[:len(log_mel)] = log_mel
                log_mel = padded
            else:
                log_mel = log_mel[:self.n_mels]
        
        # Convert mel to linear magnitude
        magnitude = self._mel_to_linear(log_mel)
        
        # Apply Griffin-Lim for this frame
        frame = self._griffin_lim_frame(magnitude)
        
        # Overlap-add with previous frame
        output = np.zeros(self.hop_length, dtype=np.float32)
        
        # Add overlapping portion from previous frame
        overlap_samples = self.n_fft - self.hop_length
        if overlap_samples > 0:
            # Blend with overlap buffer
            blend_len = min(self.hop_length, overlap_samples)
            output[:blend_len] = (
                self.overlap_buffer[:blend_len] + 
                frame[:blend_len]
            )
            if self.hop_length > blend_len:
                output[blend_len:] = frame[blend_len:self.hop_length]
        else:
            output = frame[:self.hop_length]
        
        # Update overlap buffer
        self.overlap_buffer = frame[self.hop_length:].copy()
        if len(self.overlap_buffer) < self.n_fft - self.hop_length:
            pad = np.zeros(self.n_fft - self.hop_length - len(self.overlap_buffer))
            self.overlap_buffer = np.concatenate([self.overlap_buffer, pad])
        
        # Apply crossfade with previous output for smoothness
        if self.fade_len > 0 and len(self.prev_tail) == self.fade_len:
            output[:self.fade_len] = (
                self.fade_out * self.prev_tail +
                self.fade_in * output[:self.fade_len]
            )
        
        # Store tail for next crossfade
        self.prev_tail = output[-self.fade_len:].copy()
        
        # Normalize to prevent clipping
        max_val = np.abs(output).max()
        if max_val > 0.95:
            output = output * 0.9 / max_val
        
        return output
    
    def reset(self):
        """Reset vocoder state."""
        self.mel_buffer = []
        self.prev_phase = None
        self.overlap_buffer = np.zeros(self.n_fft, dtype=np.float32)
        self.prev_tail = np.zeros(self.fade_len, dtype=np.float32)


class AsyncWaveRNNVocoder:
    """
    Asynchronous WaveRNN vocoder wrapper.
    
    Runs WaveRNN in a background thread to prevent blocking.
    Falls back to most recent output if synthesis can't keep up.
    """
    
    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        mel_bins: int = 80
    ):
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "AsyncWaveRNNVocoder requires torch and torchaudio. "
                "Please install them via pip."
            )
        
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.mel_bins = mel_bins
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load vocoder model
        bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
        params = getattr(bundle, "_wavernn_params", {})
        self.expected_mel_bins = params.get("n_freq", mel_bins)
        vocoder = bundle.get_vocoder()
        self.vocoder = vocoder.to(self.device).eval()
        
        # Threading
        self._input_queue = queue.Queue(maxsize=4)
        self._output_queue = queue.Queue(maxsize=4)
        self._running = True
        self._thread = threading.Thread(target=self._synthesis_loop, daemon=True)
        self._thread.start()
        
        # Fallback output
        self._last_output = np.zeros(hop_length, dtype=np.float32)
        self._synthesis_count = 0
        self._fallback_count = 0
    
    def _synthesis_loop(self):
        """Background synthesis thread."""
        mel_buffer = []
        max_frames = 16
        
        while self._running:
            try:
                # Get mel frame with timeout
                log_mel = self._input_queue.get(timeout=0.1)
                
                # Pad/truncate mel
                if len(log_mel) != self.expected_mel_bins:
                    if len(log_mel) < self.expected_mel_bins:
                        padded = np.zeros(self.expected_mel_bins, dtype=np.float32)
                        padded[:len(log_mel)] = log_mel
                        log_mel = padded
                    else:
                        log_mel = log_mel[:self.expected_mel_bins]
                
                mel_buffer.append(log_mel)
                if len(mel_buffer) > max_frames:
                    mel_buffer = mel_buffer[-max_frames:]
                
                # Only synthesize every few frames to keep up
                if len(mel_buffer) >= 4:
                    mel_tensor = torch.tensor(
                        np.stack(mel_buffer, axis=1),
                        dtype=torch.float32,
                        device=self.device
                    ).unsqueeze(0)
                    
                    with torch.no_grad():
                        audio_out = self.vocoder(mel_tensor)
                    
                    if isinstance(audio_out, tuple):
                        waveform = audio_out[0]
                    else:
                        waveform = audio_out
                    
                    audio_np = waveform.squeeze(0).cpu().numpy().astype(np.float32)
                    
                    # Extract hop-sized chunks and queue them
                    for i in range(0, len(audio_np), self.hop_length):
                        chunk = audio_np[i:i + self.hop_length]
                        if len(chunk) == self.hop_length:
                            try:
                                self._output_queue.put_nowait(chunk)
                            except queue.Full:
                                pass
                    
                    # Clear buffer but keep last frame for continuity
                    mel_buffer = mel_buffer[-1:]
                    self._synthesis_count += 1
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"WaveRNN synthesis error: {e}")
    
    def synthesize(self, log_mel: np.ndarray) -> np.ndarray:
        """
        Submit mel frame and get audio output.
        Returns cached output if synthesis can't keep up.
        """
        # Submit mel frame (non-blocking)
        try:
            self._input_queue.put_nowait(log_mel)
        except queue.Full:
            pass
        
        # Try to get output (non-blocking)
        try:
            self._last_output = self._output_queue.get_nowait()
        except queue.Empty:
            self._fallback_count += 1
        
        return self._last_output.copy()
    
    def reset(self):
        """Reset state."""
        # Clear queues
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                break
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except queue.Empty:
                break
        self._last_output = np.zeros(self.hop_length, dtype=np.float32)
    
    def stop(self):
        """Stop background thread."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


# Default export - use fast Griffin-Lim vocoder
class WaveRNNVocoder(GriffinLimVocoder):
    """
    Default vocoder implementation using Griffin-Lim for real-time performance.
    
    Named WaveRNNVocoder for API compatibility, but uses Griffin-Lim internally
    as WaveRNN is too slow for real-time synthesis without GPU.
    
    For actual WaveRNN synthesis (higher quality but slower), use AsyncWaveRNNVocoder.
    """
    
    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        mel_bins: int = 80
    ):
        super().__init__(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=1024,
            n_mels=mel_bins,
            n_iter=16  # Balance between quality and speed
        )
