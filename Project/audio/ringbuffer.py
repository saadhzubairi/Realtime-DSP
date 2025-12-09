"""
Lock-free-ish ring buffer primitives for audio processing.
Designed for single-producer, single-consumer scenarios.
"""

import numpy as np
import threading
from typing import Optional, Tuple
from dataclasses import dataclass


class RingBuffer:
    """
    Circular buffer for audio samples.
    
    Designed for single-producer, single-consumer use between
    audio callback thread and DSP worker thread.
    
    Uses numpy arrays for efficient operations.
    """
    
    def __init__(self, capacity: int, dtype=np.float32):
        """
        Initialize ring buffer.
        
        Args:
            capacity: Maximum number of samples to store
            dtype: NumPy data type for samples
        """
        self._buffer = np.zeros(capacity, dtype=dtype)
        self._capacity = capacity
        self._write_idx = 0
        self._read_idx = 0
        self._count = 0
        self._lock = threading.Lock()
        self._dtype = dtype
    
    @property
    def capacity(self) -> int:
        """Get buffer capacity."""
        return self._capacity
    
    @property
    def count(self) -> int:
        """Get number of samples currently in buffer (thread-safe)."""
        with self._lock:
            return self._count
    
    @property
    def available_write(self) -> int:
        """Get number of samples that can be written."""
        with self._lock:
            return self._capacity - self._count
    
    @property
    def available_read(self) -> int:
        """Get number of samples that can be read."""
        return self.count
    
    def push(self, data: np.ndarray) -> int:
        """
        Push samples into the buffer.
        
        Args:
            data: NumPy array of samples to push
            
        Returns:
            Number of samples actually pushed
        """
        n = len(data)
        
        with self._lock:
            available = self._capacity - self._count
            n_to_write = min(n, available)
            
            if n_to_write == 0:
                return 0
            
            # Handle wrap-around
            end_space = self._capacity - self._write_idx
            
            if n_to_write <= end_space:
                # No wrap needed
                self._buffer[self._write_idx:self._write_idx + n_to_write] = data[:n_to_write]
            else:
                # Wrap around
                self._buffer[self._write_idx:] = data[:end_space]
                remaining = n_to_write - end_space
                self._buffer[:remaining] = data[end_space:n_to_write]
            
            self._write_idx = (self._write_idx + n_to_write) % self._capacity
            self._count += n_to_write
        
        return n_to_write
    
    def pop(self, n: int) -> Optional[np.ndarray]:
        """
        Pop samples from the buffer.
        
        Args:
            n: Number of samples to pop
            
        Returns:
            NumPy array of samples, or None if not enough samples
        """
        with self._lock:
            if self._count < n:
                return None
            
            result = np.zeros(n, dtype=self._dtype)
            
            # Handle wrap-around
            end_space = self._capacity - self._read_idx
            
            if n <= end_space:
                # No wrap needed
                result[:] = self._buffer[self._read_idx:self._read_idx + n]
            else:
                # Wrap around
                result[:end_space] = self._buffer[self._read_idx:]
                remaining = n - end_space
                result[end_space:] = self._buffer[:remaining]
            
            self._read_idx = (self._read_idx + n) % self._capacity
            self._count -= n
        
        return result
    
    def peek(self, n: int) -> Optional[np.ndarray]:
        """
        Peek at samples without removing them.
        
        Args:
            n: Number of samples to peek
            
        Returns:
            NumPy array of samples, or None if not enough samples
        """
        with self._lock:
            if self._count < n:
                return None
            
            result = np.zeros(n, dtype=self._dtype)
            
            # Handle wrap-around
            end_space = self._capacity - self._read_idx
            
            if n <= end_space:
                result[:] = self._buffer[self._read_idx:self._read_idx + n]
            else:
                result[:end_space] = self._buffer[self._read_idx:]
                remaining = n - end_space
                result[end_space:] = self._buffer[:remaining]
        
        return result
    
    def skip(self, n: int) -> int:
        """
        Skip samples without reading them.
        
        Args:
            n: Number of samples to skip
            
        Returns:
            Number of samples actually skipped
        """
        with self._lock:
            n_to_skip = min(n, self._count)
            self._read_idx = (self._read_idx + n_to_skip) % self._capacity
            self._count -= n_to_skip
        return n_to_skip
    
    def clear(self):
        """Clear all samples from the buffer."""
        with self._lock:
            self._write_idx = 0
            self._read_idx = 0
            self._count = 0
            self._buffer.fill(0)


class OverlapBuffer:
    """
    Buffer for overlap-add STFT processing.
    
    Maintains frame_size samples of history for overlapping analysis frames.
    Outputs hop_size samples at a time for synthesis.
    """
    
    def __init__(self, frame_size: int, hop_size: int, dtype=np.float32):
        """
        Initialize overlap buffer.
        
        Args:
            frame_size: Size of analysis/synthesis frames
            hop_size: Hop size between frames
            dtype: NumPy data type
        """
        self.frame_size = frame_size
        self.hop_size = hop_size
        self._dtype = dtype
        
        # Analysis buffer (stores incoming samples)
        self._analysis_buffer = np.zeros(frame_size, dtype=dtype)
        self._analysis_count = 0
        
        # Synthesis buffer (for overlap-add output)
        self._synthesis_buffer = np.zeros(frame_size, dtype=dtype)
        
        self._lock = threading.Lock()
    
    def push_samples(self, samples: np.ndarray) -> int:
        """
        Push new samples for analysis.
        
        Args:
            samples: Input samples
            
        Returns:
            Number of complete frames available
        """
        with self._lock:
            n = len(samples)
            
            # Shift buffer left and add new samples at the end
            if self._analysis_count + n >= self.frame_size:
                # Shift by hop_size to make room
                shift = min(n, self.hop_size)
                self._analysis_buffer[:-shift] = self._analysis_buffer[shift:]
                self._analysis_buffer[-shift:] = samples[-shift:]
                self._analysis_count = self.frame_size
            else:
                # Just append
                start = self._analysis_count
                self._analysis_buffer[start:start + n] = samples
                self._analysis_count += n
            
            # Return number of complete frames
            if self._analysis_count >= self.frame_size:
                return 1
            return 0
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the current analysis frame (copy).
        
        Returns:
            Frame of frame_size samples, or None if not ready
        """
        with self._lock:
            if self._analysis_count < self.frame_size:
                return None
            return self._analysis_buffer.copy()
    
    def add_synthesis_frame(self, frame: np.ndarray):
        """
        Add a synthesis frame using overlap-add.
        
        Args:
            frame: Synthesis frame (should be frame_size samples)
        """
        with self._lock:
            # Overlap-add: add the new frame to the synthesis buffer
            self._synthesis_buffer += frame
    
    def pop_output(self) -> np.ndarray:
        """
        Pop hop_size samples of synthesized output.
        
        Returns:
            Array of hop_size samples
        """
        with self._lock:
            # Get the output samples
            output = self._synthesis_buffer[:self.hop_size].copy()
            
            # Shift the synthesis buffer
            self._synthesis_buffer[:-self.hop_size] = self._synthesis_buffer[self.hop_size:]
            self._synthesis_buffer[-self.hop_size:] = 0
            
            return output
    
    def clear(self):
        """Clear all buffers."""
        with self._lock:
            self._analysis_buffer.fill(0)
            self._synthesis_buffer.fill(0)
            self._analysis_count = 0


class FrameQueue:
    """
    Thread-safe queue for passing frames between threads.
    Designed for single-producer, single-consumer pattern.
    """
    
    def __init__(self, max_frames: int, frame_size: int, dtype=np.float32):
        """
        Initialize frame queue.
        
        Args:
            max_frames: Maximum number of frames to store
            frame_size: Size of each frame
            dtype: NumPy data type
        """
        self._buffer = np.zeros((max_frames, frame_size), dtype=dtype)
        self._max_frames = max_frames
        self._frame_size = frame_size
        self._write_idx = 0
        self._read_idx = 0
        self._count = 0
        self._lock = threading.Lock()
        self._dtype = dtype
    
    @property
    def count(self) -> int:
        """Get number of frames in queue."""
        with self._lock:
            return self._count
    
    @property
    def is_full(self) -> bool:
        """Check if queue is full."""
        with self._lock:
            return self._count >= self._max_frames
    
    @property
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return self._count == 0
    
    def push(self, frame: np.ndarray) -> bool:
        """
        Push a frame onto the queue.
        
        Args:
            frame: Frame to push (must be frame_size samples)
            
        Returns:
            True if successful, False if queue is full
        """
        with self._lock:
            if self._count >= self._max_frames:
                return False
            
            self._buffer[self._write_idx] = frame[:self._frame_size]
            self._write_idx = (self._write_idx + 1) % self._max_frames
            self._count += 1
        
        return True
    
    def pop(self) -> Optional[np.ndarray]:
        """
        Pop a frame from the queue.
        
        Returns:
            Frame array, or None if queue is empty
        """
        with self._lock:
            if self._count == 0:
                return None
            
            frame = self._buffer[self._read_idx].copy()
            self._read_idx = (self._read_idx + 1) % self._max_frames
            self._count -= 1
        
        return frame
    
    def clear(self):
        """Clear the queue."""
        with self._lock:
            self._write_idx = 0
            self._read_idx = 0
            self._count = 0
            self._buffer.fill(0)
