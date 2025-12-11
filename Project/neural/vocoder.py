"""
WaveRNN neural vocoder wrapper for real-time synthesis.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

try:
    import torch
    import torchaudio
except ImportError as exc:
    torch = None
    torchaudio = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class WaveRNNVocoder:
    """
    Lightweight wrapper around torchaudio's WaveRNN vocoder.
    
    Converts log-mel frames into waveform samples. Maintains
    a sliding mel buffer so we can request short, real-time chunks.
    """
    
    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        mel_bins: int
    ):
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "WaveRNNVocoder requires torch and torchaudio. "
                "Please install them via pip."
            ) from _IMPORT_ERROR
        
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.mel_bins = mel_bins
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
        params = getattr(bundle, "_wavernn_params", {})
        self.expected_mel_bins = params.get("n_freq", mel_bins)
        self.vocoder_hop = params.get("hop_length", hop_length)
        vocoder = bundle.get_vocoder()
        self.vocoder = vocoder.to(self.device).eval()
        
        self.mel_buffer = []
        self.max_mel_frames = 32
        self.fade = np.linspace(0, 1, hop_length, dtype=np.float32)
        self.prev_tail = np.zeros(hop_length, dtype=np.float32)
    
    def synthesize(self, log_mel: np.ndarray) -> np.ndarray:
        """
        Append a new log-mel frame and synthesize a hop-sized block.
        """
        if log_mel.shape[0] != self.expected_mel_bins:
            log_mel = self._pad_or_truncate_mel(log_mel)
        
        self.mel_buffer.append(log_mel)
        if len(self.mel_buffer) > self.max_mel_frames:
            self.mel_buffer = self.mel_buffer[-self.max_mel_frames:]
        
        mel_tensor = torch.tensor(
            np.stack(self.mel_buffer, axis=1),
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
        if len(audio_np) < self.hop_length:
            padded = np.zeros(self.hop_length, dtype=np.float32)
            padded[:len(audio_np)] = audio_np
            audio_np = padded
        else:
            audio_np = audio_np[-self.hop_length:]
        
        output = self._crossfade(audio_np)
        self.prev_tail = output.copy()
        return output
    
    def _crossfade(self, block: np.ndarray) -> np.ndarray:
        fade_len = min(len(block), len(self.prev_tail))
        if fade_len == 0:
            return block
        fade_in = self.fade[:fade_len]
        fade_out = 1 - fade_in
        block[:fade_len] = (
            fade_out * self.prev_tail[:fade_len] +
            fade_in * block[:fade_len]
        )
        return block
    
    def _pad_or_truncate_mel(self, mel: np.ndarray) -> np.ndarray:
        if mel.shape[0] == self.expected_mel_bins:
            return mel
        if mel.shape[0] < self.expected_mel_bins:
            padded = np.zeros(self.expected_mel_bins, dtype=np.float32)
            padded[:mel.shape[0]] = mel
            return padded
        return mel[:self.expected_mel_bins]
