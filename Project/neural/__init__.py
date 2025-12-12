"""
Neural synthesis modules (vocoder engines, neural effects).
"""

from .vocoder import WaveRNNVocoder, GriffinLimVocoder, AsyncWaveRNNVocoder, VocoderConfig

__all__ = [
    'WaveRNNVocoder',
    'GriffinLimVocoder',
    'AsyncWaveRNNVocoder',
    'VocoderConfig',
]
