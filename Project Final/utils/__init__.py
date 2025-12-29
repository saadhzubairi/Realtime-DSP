"""Utils package for configuration and logging."""

from .config import (
    AudioConfig,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_BUFFER_SIZE,
    get_profiles_directory,
)

from .logging_utils import (
    setup_logger,
    ui_log_buffer,
    audio_logger,
    dsp_logger,
)

__all__ = [
    'AudioConfig',
    'DEFAULT_SAMPLE_RATE',
    'DEFAULT_BUFFER_SIZE',
    'get_profiles_directory',
    'setup_logger',
    'ui_log_buffer',
    'audio_logger',
    'dsp_logger',
]
