"""
Real-time RVC pipeline for streaming voice conversion.

Handles buffering, resampling, and crossfading for smooth real-time conversion.
"""

from __future__ import annotations

import numpy as np
import threading
import queue
import time
from typing import Optional, Tuple, Callable
from dataclasses import dataclass

try:
    import resampy
    _RESAMPY_AVAILABLE = True
except ImportError:
    _RESAMPY_AVAILABLE = False
    
try:
    from scipy import signal
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

from .inference import RVCVoiceConverter, RVCConfig


@dataclass
class RealtimeConfig:
    """Configuration for real-time RVC processing."""
    
    # Audio settings
    input_sample_rate: int = 16000   # Microphone sample rate
    output_sample_rate: int = 16000  # Speaker sample rate
    
    # Buffer settings (in seconds)
    block_time: float = 0.1         # Processing block size (100ms default)
    crossfade_time: float = 0.02    # Crossfade duration (20ms)
    extra_context: float = 0.5      # Extra audio for pitch detection context
    
    # Threading
    use_async: bool = True          # Process in background thread
    max_queue_size: int = 4         # Max pending blocks


class AudioResampler:
    """Simple audio resampler using scipy or resampy."""
    
    def __init__(self, input_sr: int, output_sr: int):
        self.input_sr = input_sr
        self.output_sr = output_sr
    
    def resample(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio to target sample rate."""
        if self.input_sr == self.output_sr:
            return audio
        
        if _RESAMPY_AVAILABLE:
            return resampy.resample(audio, self.input_sr, self.output_sr)
        elif _SCIPY_AVAILABLE:
            num_samples = int(len(audio) * self.output_sr / self.input_sr)
            return signal.resample(audio, num_samples).astype(np.float32)
        else:
            # Fallback: simple linear interpolation
            ratio = self.output_sr / self.input_sr
            new_len = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_len)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


class RealtimeRVCPipeline:
    """
    Real-time RVC voice conversion pipeline.
    
    Handles:
    - Audio buffering and block accumulation
    - Resampling between different sample rates
    - Crossfading for smooth output
    - Optional async processing
    """
    
    def __init__(
        self,
        converter: RVCVoiceConverter,
        config: Optional[RealtimeConfig] = None
    ):
        """
        Initialize the real-time pipeline.
        
        Args:
            converter: RVCVoiceConverter instance with loaded model
            config: Real-time configuration
        """
        self.converter = converter
        self.config = config or RealtimeConfig()
        
        # Calculate block sizes in samples
        self.input_block_size = int(self.config.block_time * self.config.input_sample_rate)
        self.crossfade_samples = int(self.config.crossfade_time * self.config.output_sample_rate)
        self.context_samples = int(self.config.extra_context * self.config.input_sample_rate)
        
        # Input buffer for accumulating samples
        self._input_buffer = np.zeros(0, dtype=np.float32)
        
        # Context buffer for pitch extraction
        self._context_buffer = np.zeros(self.context_samples, dtype=np.float32)
        
        # Output crossfade buffer
        self._crossfade_buffer = np.zeros(self.crossfade_samples, dtype=np.float32)
        self._has_prev_output = False
        
        # Resamplers
        self._upsampler = AudioResampler(
            self.config.input_sample_rate,
            40000  # RVC typically expects 40kHz
        )
        self._downsampler = AudioResampler(
            40000,  # RVC outputs at 40kHz
            self.config.output_sample_rate
        )
        
        # Threading for async processing
        self._lock = threading.Lock()
        self._processing = False
        
        if self.config.use_async:
            self._input_queue = queue.Queue(maxsize=self.config.max_queue_size)
            self._output_queue = queue.Queue(maxsize=self.config.max_queue_size)
            self._running = True
            self._thread = threading.Thread(target=self._processing_loop, daemon=True)
            self._thread.start()
        
        # Metrics
        self._last_process_time = 0.0
        self._frames_processed = 0
    
    def push_audio(self, samples: np.ndarray) -> Optional[np.ndarray]:
        """
        Push audio samples and get converted output if ready.
        
        Args:
            samples: Input audio samples (float32)
            
        Returns:
            Converted audio if a full block is ready, None otherwise
        """
        samples = samples.astype(np.float32)
        
        with self._lock:
            # Accumulate input
            self._input_buffer = np.concatenate([self._input_buffer, samples])
        
        # Check if we have enough for a block
        if len(self._input_buffer) >= self.input_block_size:
            if self.config.use_async:
                return self._process_async()
            else:
                return self._process_sync()
        
        return None
    
    def _process_sync(self) -> Optional[np.ndarray]:
        """Process synchronously."""
        with self._lock:
            if len(self._input_buffer) < self.input_block_size:
                return None
            
            # Extract block
            block = self._input_buffer[:self.input_block_size]
            self._input_buffer = self._input_buffer[self.input_block_size:]
        
        return self._convert_block(block)
    
    def _process_async(self) -> Optional[np.ndarray]:
        """Process asynchronously using background thread."""
        with self._lock:
            if len(self._input_buffer) < self.input_block_size:
                pass
            else:
                # Extract block
                block = self._input_buffer[:self.input_block_size]
                self._input_buffer = self._input_buffer[self.input_block_size:]
                
                # Queue for processing
                try:
                    self._input_queue.put_nowait(block)
                except queue.Full:
                    pass  # Skip if queue is full
        
        # Try to get processed output
        try:
            return self._output_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _processing_loop(self):
        """Background processing thread."""
        while self._running:
            try:
                block = self._input_queue.get(timeout=0.1)
                output = self._convert_block(block)
                if output is not None:
                    try:
                        self._output_queue.put_nowait(output)
                    except queue.Full:
                        pass  # Drop if output queue full
            except queue.Empty:
                continue
            except Exception as e:
                print(f"RVC processing error: {e}")
    
    def _convert_block(self, block: np.ndarray) -> Optional[np.ndarray]:
        """Convert a block of audio."""
        if not self.converter.is_ready:
            return block  # Pass through if no model
        
        start_time = time.perf_counter()
        
        try:
            # Add context for better pitch detection
            with_context = np.concatenate([self._context_buffer, block])
            
            # Update context buffer
            if len(block) >= len(self._context_buffer):
                self._context_buffer = block[-len(self._context_buffer):]
            else:
                shift = len(block)
                self._context_buffer[:-shift] = self._context_buffer[shift:]
                self._context_buffer[-shift:] = block
            
            # Upsample to RVC's expected sample rate
            upsampled = self._upsampler.resample(with_context)
            
            # Convert through RVC
            converted, output_sr = self.converter.convert_audio(
                upsampled,
                sample_rate=40000  # Upsampled rate
            )
            
            # Remove the context portion from output
            context_output_samples = int(
                len(self._context_buffer) * output_sr / self.config.input_sample_rate
            )
            if len(converted) > context_output_samples:
                converted = converted[context_output_samples:]
            
            # Downsample to output sample rate
            if output_sr != self.config.output_sample_rate:
                self._downsampler.input_sr = output_sr
                converted = self._downsampler.resample(converted)
            
            # Apply crossfade with previous output
            output = self._apply_crossfade(converted)
            
            self._last_process_time = (time.perf_counter() - start_time) * 1000
            self._frames_processed += 1
            
            return output
            
        except Exception as e:
            print(f"Block conversion error: {e}")
            return block  # Return original on error
    
    def _apply_crossfade(self, block: np.ndarray) -> np.ndarray:
        """Apply crossfade with previous block for smooth transitions."""
        if not self._has_prev_output or self.crossfade_samples == 0:
            self._has_prev_output = True
            if len(block) >= self.crossfade_samples:
                self._crossfade_buffer = block[-self.crossfade_samples:].copy()
            return block
        
        # Crossfade at the start of the block
        fade_len = min(self.crossfade_samples, len(block))
        if fade_len > 0:
            fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
            fade_out = 1 - fade_in
            
            block[:fade_len] = (
                fade_out * self._crossfade_buffer[:fade_len] +
                fade_in * block[:fade_len]
            )
        
        # Store tail for next crossfade
        if len(block) >= self.crossfade_samples:
            self._crossfade_buffer = block[-self.crossfade_samples:].copy()
        
        return block
    
    def get_output_blocking(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        """
        Get output, blocking until available or timeout.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            Converted audio or None if timeout
        """
        if not self.config.use_async:
            return None
        
        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def flush(self) -> Optional[np.ndarray]:
        """Flush any remaining buffered audio."""
        with self._lock:
            if len(self._input_buffer) > 0:
                block = self._input_buffer.copy()
                self._input_buffer = np.zeros(0, dtype=np.float32)
                
                # Pad to minimum size if needed
                if len(block) < self.input_block_size // 4:
                    return None  # Too small, discard
                
                return self._convert_block(block)
        return None
    
    def reset(self):
        """Reset all buffers."""
        with self._lock:
            self._input_buffer = np.zeros(0, dtype=np.float32)
            self._context_buffer = np.zeros(self.context_samples, dtype=np.float32)
            self._crossfade_buffer = np.zeros(self.crossfade_samples, dtype=np.float32)
            self._has_prev_output = False
        
        # Clear queues
        if self.config.use_async:
            while not self._input_queue.empty():
                try:
                    self._input_queue.get_nowait()
                except queue.Empty:
                    break
            while not self._output_queue.empty():
                try:
                    self._output_queue.get_nowait()
                except queue.Empty:
                    break
    
    def stop(self):
        """Stop the processing thread."""
        self._running = False
        if hasattr(self, '_thread') and self._thread.is_alive():
            self._thread.join(timeout=1.0)
    
    @property
    def process_time_ms(self) -> float:
        """Get the last block processing time in milliseconds."""
        return self._last_process_time
    
    @property
    def is_realtime(self) -> bool:
        """Check if processing is keeping up with real-time."""
        block_time_ms = self.config.block_time * 1000
        return self._last_process_time < block_time_ms
