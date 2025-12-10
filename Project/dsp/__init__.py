"""
DSP package initialization.
"""

from .stft import (
    STFT,
    STFTParams,
    OverlapAddProcessor,
    compute_power_spectrum,
    compute_log_magnitude,
    magnitude_to_db,
    db_to_magnitude,
)

from .pitch import (
    YINPitchDetector,
    PitchTracker,
    PitchEstimate,
    hz_to_log_f0,
    log_f0_to_hz,
    hz_to_semitones,
    semitones_to_hz,
    map_pitch_linear,
)

from .lpc import (
    compute_lpc,
    lpc_to_spectrum,
    spectrum_to_lpc,
    lpc_filter,
    lpc_inverse_filter,
    LPCResult,
    smooth_envelope,
    interpolate_envelope,
)

from .formant import (
    Formant,
    FormantEstimate,
    estimate_formants,
    FormantTracker,
    warp_envelope,
    bilinear_warp_envelope,
    compute_formant_warp_factor,
)

from .voice_profile import (
    VoiceProfile,
    extract_profile,
    save_profile,
    load_profile,
    list_profiles,
)

from .transform import (
    VoiceTransformPipeline,
    TransformState,
)

from .pitch_shift import (
    VariableDelayPitchShifter,
    SimplePitchShifter,
)

__all__ = [
    # STFT
    'STFT',
    'STFTParams',
    'OverlapAddProcessor',
    'compute_power_spectrum',
    'compute_log_magnitude',
    'magnitude_to_db',
    'db_to_magnitude',
    
    # Pitch
    'YINPitchDetector',
    'PitchTracker',
    'PitchEstimate',
    'hz_to_log_f0',
    'log_f0_to_hz',
    'hz_to_semitones',
    'semitones_to_hz',
    'map_pitch_linear',
    
    # LPC
    'compute_lpc',
    'lpc_to_spectrum',
    'spectrum_to_lpc',
    'lpc_filter',
    'lpc_inverse_filter',
    'LPCResult',
    'smooth_envelope',
    'interpolate_envelope',
    
    # Formant
    'Formant',
    'FormantEstimate',
    'estimate_formants',
    'FormantTracker',
    'warp_envelope',
    'bilinear_warp_envelope',
    'compute_formant_warp_factor',
    
    # Voice Profile
    'VoiceProfile',
    'extract_profile',
    'save_profile',
    'load_profile',
    'list_profiles',
    
    # Transform
    'VoiceTransformPipeline',
    'TransformState',
    
    # Pitch Shifting
    'VariableDelayPitchShifter',
    'SimplePitchShifter',
]
