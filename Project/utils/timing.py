"""
Timing utilities for performance measurement.
Low-overhead timing for DSP callback profiling.
"""

import time
from collections import deque
from typing import Deque, Optional
from dataclasses import dataclass, field
import threading


class HighResTimer:
    """High-resolution timer using time.perf_counter()."""
    
    def __init__(self):
        self._start: float = 0.0
        self._elapsed: float = 0.0
    
    def start(self):
        """Start the timer."""
        self._start = time.perf_counter()
    
    def stop(self) -> float:
        """Stop and return elapsed time in milliseconds."""
        self._elapsed = (time.perf_counter() - self._start) * 1000
        return self._elapsed
    
    @property
    def elapsed_ms(self) -> float:
        """Get last elapsed time in milliseconds."""
        return self._elapsed


@dataclass
class TimingStats:
    """Running statistics for timing measurements."""
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    min_ms: float = float('inf')
    
    # Exponential moving average
    ema_ms: float = 0.0
    ema_alpha: float = 0.1
    
    def update(self, elapsed_ms: float):
        """Update statistics with a new measurement."""
        self.count += 1
        self.total_ms += elapsed_ms
        self.max_ms = max(self.max_ms, elapsed_ms)
        self.min_ms = min(self.min_ms, elapsed_ms)
        
        # Update EMA
        if self.count == 1:
            self.ema_ms = elapsed_ms
        else:
            self.ema_ms = self.ema_alpha * elapsed_ms + (1 - self.ema_alpha) * self.ema_ms
    
    @property
    def avg_ms(self) -> float:
        """Get average time in milliseconds."""
        return self.total_ms / self.count if self.count > 0 else 0.0
    
    def reset(self):
        """Reset all statistics."""
        self.count = 0
        self.total_ms = 0.0
        self.max_ms = 0.0
        self.min_ms = float('inf')
        self.ema_ms = 0.0


class FrameTimer:
    """
    Timer for measuring DSP frame processing time.
    Thread-safe and designed for real-time use.
    """
    
    def __init__(self, window_size: int = 100, ema_alpha: float = 0.1):
        self._timer = HighResTimer()
        self._stats = TimingStats(ema_alpha=ema_alpha)
        self._history: Deque[float] = deque(maxlen=window_size)
        self._lock = threading.Lock()
    
    def start(self):
        """Start timing a frame."""
        self._timer.start()
    
    def stop(self) -> float:
        """Stop timing and record the measurement."""
        elapsed = self._timer.stop()
        with self._lock:
            self._stats.update(elapsed)
            self._history.append(elapsed)
        return elapsed
    
    @property
    def avg_ms(self) -> float:
        """Get running average in milliseconds."""
        with self._lock:
            return self._stats.ema_ms
    
    @property
    def max_ms(self) -> float:
        """Get maximum time in milliseconds."""
        with self._lock:
            return self._stats.max_ms
    
    def get_history(self) -> list:
        """Get timing history for plotting."""
        with self._lock:
            return list(self._history)
    
    def reset(self):
        """Reset all timing data."""
        with self._lock:
            self._stats.reset()
            self._history.clear()


class LatencyEstimator:
    """
    Estimates end-to-end latency from queue depths and buffer sizes.
    """
    
    def __init__(self, hop_size: int, sample_rate: int):
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self._input_queue_depth = 0
        self._output_queue_depth = 0
        self._buffer_size = 256
        self._lock = threading.Lock()
    
    def update(self, input_depth: int, output_depth: int, buffer_size: int):
        """Update queue depths."""
        with self._lock:
            self._input_queue_depth = input_depth
            self._output_queue_depth = output_depth
            self._buffer_size = buffer_size
    
    @property
    def estimated_latency_ms(self) -> float:
        """
        Estimate total latency in milliseconds.
        Latency = input_buffer + queue_depth * hop + output_buffer
        """
        with self._lock:
            # Input buffer latency
            input_latency = self._buffer_size / self.sample_rate * 1000
            
            # Processing queue latency
            queue_latency = (self._input_queue_depth + self._output_queue_depth) * \
                           self.hop_size / self.sample_rate * 1000
            
            # Output buffer latency
            output_latency = self._buffer_size / self.sample_rate * 1000
            
            return input_latency + queue_latency + output_latency
    
    @property
    def queue_latency_ms(self) -> float:
        """Get just the queue-induced latency."""
        with self._lock:
            return (self._input_queue_depth + self._output_queue_depth) * \
                   self.hop_size / self.sample_rate * 1000


def measure_time(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"{func.__name__}: {elapsed:.2f} ms")
        return result
    return wrapper
