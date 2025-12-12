"""
RVC (Retrieval-based Voice Conversion) integration module.

Provides real-time voice conversion using pre-trained RVC models.
Only requires a target voice model (Person B) - no source profile needed.
"""

from .inference import RVCVoiceConverter, RVCConfig
from .realtime import RealtimeRVCPipeline

__all__ = [
    'RVCVoiceConverter',
    'RVCConfig', 
    'RealtimeRVCPipeline',
]
