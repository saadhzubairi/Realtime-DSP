"""DSP package for voice transformation."""

from .pitch import YINPitchDetector, PitchTracker, PitchEstimate, hz_to_log_f0, log_f0_to_hz
from .lpc import compute_lpc, lpc_to_spectrum, LPCResult
from .formant import Formant, FormantEstimate, estimate_formants, FormantTracker
from .voice_profile import VoiceProfile, extract_profile, save_profile, load_profile
from .world_vocoder import WorldVocoder, RealtimeWorldVocoder, WorldParams

__all__ = [
    'YINPitchDetector',
    'PitchTracker',
    'PitchEstimate',
    'hz_to_log_f0',
    'log_f0_to_hz',
    'compute_lpc',
    'lpc_to_spectrum',
    'LPCResult',
    'Formant',
    'FormantEstimate',
    'estimate_formants',
    'FormantTracker',
    'VoiceProfile',
    'extract_profile',
    'save_profile',
    'load_profile',
    'WorldVocoder',
    'RealtimeWorldVocoder',
    'WorldParams',
]
