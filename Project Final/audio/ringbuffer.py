"""Circular buffer primitives for audio processing."""

import numpy as np
import threading
from typing import Optional


class RingBuffer:
    """Circular buffer for audio samples."""
    
    def __init__(self, capacity: int, dtype=np.float32):
        self._buffer = np.zeros(capacity, dtype=dtype)
        self._capacity = capacity
        self._write_idx = 0
        self._read_idx = 0
        self._count = 0
        self._lock = threading.Lock()
        self._dtype = dtype
    
    @property
    def capacity(self) -> int:
        return self._capacity
    
    @property
    def count(self) -> int:
        with self._lock:
            return self._count
    
    @property
    def available_write(self) -> int:
        with self._lock:
            return self._capacity - self._count
    
    @property
    def available_read(self) -> int:
        return self.count
    
    def push(self, data: np.ndarray) -> int:
        """Push samples into the buffer."""
        n = len(data)
        
        with self._lock:
            available = self._capacity - self._count
            n_to_write = min(n, available)
            
            if n_to_write == 0:
                return 0
            
            end_space = self._capacity - self._write_idx
            
            if n_to_write <= end_space:
                self._buffer[self._write_idx:self._write_idx + n_to_write] = data[:n_to_write]
            else:
                self._buffer[self._write_idx:] = data[:end_space]
                remaining = n_to_write - end_space
                self._buffer[:remaining] = data[end_space:n_to_write]
            
            self._write_idx = (self._write_idx + n_to_write) % self._capacity
            self._count += n_to_write
        
        return n_to_write
    
    def pop(self, n: int) -> Optional[np.ndarray]:
        """Pop samples from the buffer."""
        with self._lock:
            if self._count < n:
                return None
            
            result = np.zeros(n, dtype=self._dtype)
            end_space = self._capacity - self._read_idx
            
            if n <= end_space:
                result[:] = self._buffer[self._read_idx:self._read_idx + n]
            else:
                result[:end_space] = self._buffer[self._read_idx:]
                remaining = n - end_space
                result[end_space:] = self._buffer[:remaining]
            
            self._read_idx = (self._read_idx + n) % self._capacity
            self._count -= n
        
        return result
    
    def peek(self, n: int) -> Optional[np.ndarray]:
        """Peek at samples without removing them."""
        with self._lock:
            if self._count < n:
                return None
            
            result = np.zeros(n, dtype=self._dtype)
            end_space = self._capacity - self._read_idx
            
            if n <= end_space:
                result[:] = self._buffer[self._read_idx:self._read_idx + n]
            else:
                result[:end_space] = self._buffer[self._read_idx:]
                remaining = n - end_space
                result[end_space:] = self._buffer[:remaining]
        
        return result
    
    def skip(self, n: int) -> int:
        """Skip samples without reading them."""
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
