"""
Logging utilities for the voice transformation system.
Thread-safe logging with performance tracking.
"""

import logging
import time
import threading
from collections import deque
from typing import Optional, Deque
from dataclasses import dataclass, field


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            '[%(asctime)s.%(msecs)03d] %(levelname)s [%(name)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


@dataclass
class LogMessage:
    """A log message with timestamp."""
    timestamp: float
    level: str
    message: str
    source: str


class ThreadSafeLogBuffer:
    """
    Thread-safe circular buffer for log messages.
    Used to pass logs from DSP thread to UI thread.
    """
    
    def __init__(self, max_size: int = 100):
        self._buffer: Deque[LogMessage] = deque(maxlen=max_size)
        self._lock = threading.Lock()
    
    def append(self, level: str, message: str, source: str = ""):
        """Add a log message (thread-safe)."""
        msg = LogMessage(
            timestamp=time.time(),
            level=level,
            message=message,
            source=source
        )
        with self._lock:
            self._buffer.append(msg)
    
    def get_all(self) -> list:
        """Get all messages and clear buffer (thread-safe)."""
        with self._lock:
            messages = list(self._buffer)
            self._buffer.clear()
        return messages
    
    def info(self, message: str, source: str = ""):
        self.append("INFO", message, source)
    
    def warning(self, message: str, source: str = ""):
        self.append("WARNING", message, source)
    
    def error(self, message: str, source: str = ""):
        self.append("ERROR", message, source)
    
    def debug(self, message: str, source: str = ""):
        self.append("DEBUG", message, source)


# Global log buffer for UI display
ui_log_buffer = ThreadSafeLogBuffer()

# Module loggers
audio_logger = setup_logger("audio")
dsp_logger = setup_logger("dsp")
ui_logger = setup_logger("ui")
