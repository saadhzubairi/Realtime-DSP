"""
PyAudio I/O module for real-time audio streaming.
Handles device enumeration, stream setup, and callback management.
"""

import pyaudio
import numpy as np
import threading
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass

from .ringbuffer import RingBuffer
from utils.config import AudioConfig, DEFAULT_SAMPLE_RATE, DEFAULT_BUFFER_SIZE
from utils.logging_utils import audio_logger, ui_log_buffer


@dataclass
class AudioDevice:
    """Information about an audio device."""
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    is_input: bool
    is_output: bool


class AudioDeviceManager:
    """Manages audio device enumeration and selection."""
    
    def __init__(self):
        self._pa: Optional[pyaudio.PyAudio] = None
        self._devices: List[AudioDevice] = []
        self._initialize()
    
    def _initialize(self):
        """Initialize PyAudio and enumerate devices."""
        try:
            self._pa = pyaudio.PyAudio()
            self._enumerate_devices()
        except Exception as e:
            audio_logger.error(f"Failed to initialize PyAudio: {e}")
            ui_log_buffer.error(f"Audio initialization failed: {e}", "AudioDeviceManager")
    
    def _enumerate_devices(self):
        """Enumerate all available audio devices."""
        if self._pa is None:
            return
        
        self._devices = []
        
        for i in range(self._pa.get_device_count()):
            try:
                info = self._pa.get_device_info_by_index(i)
                device = AudioDevice(
                    index=i,
                    name=info['name'],
                    max_input_channels=int(info['maxInputChannels']),
                    max_output_channels=int(info['maxOutputChannels']),
                    default_sample_rate=info['defaultSampleRate'],
                    is_input=info['maxInputChannels'] > 0,
                    is_output=info['maxOutputChannels'] > 0
                )
                self._devices.append(device)
            except Exception as e:
                audio_logger.warning(f"Failed to get info for device {i}: {e}")
        
        audio_logger.info(f"Found {len(self._devices)} audio devices")
    
    def get_input_devices(self) -> List[AudioDevice]:
        """Get all input (microphone) devices."""
        return [d for d in self._devices if d.is_input]
    
    def get_output_devices(self) -> List[AudioDevice]:
        """Get all output (speaker) devices."""
        return [d for d in self._devices if d.is_output]
    
    def get_default_input_device(self) -> Optional[AudioDevice]:
        """Get the default input device."""
        if self._pa is None:
            return None
        try:
            idx = self._pa.get_default_input_device_info()['index']
            return next((d for d in self._devices if d.index == idx), None)
        except:
            inputs = self.get_input_devices()
            return inputs[0] if inputs else None
    
    def get_default_output_device(self) -> Optional[AudioDevice]:
        """Get the default output device."""
        if self._pa is None:
            return None
        try:
            idx = self._pa.get_default_output_device_info()['index']
            return next((d for d in self._devices if d.index == idx), None)
        except:
            outputs = self.get_output_devices()
            return outputs[0] if outputs else None
    
    def refresh(self):
        """Refresh device list."""
        self._enumerate_devices()
    
    def terminate(self):
        """Clean up PyAudio resources."""
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
    
    @property
    def pyaudio_instance(self) -> Optional[pyaudio.PyAudio]:
        return self._pa


class AudioStream:
    """
    Manages real-time audio I/O streams with callback processing.
    
    Uses PyAudio callback mode for low-latency operation.
    Interfaces with ring buffers for thread-safe data exchange.
    """
    
    def __init__(
        self,
        device_manager: AudioDeviceManager,
        config: AudioConfig,
        input_buffer: RingBuffer,
        output_buffer: RingBuffer
    ):
        """
        Initialize audio stream.
        
        Args:
            device_manager: AudioDeviceManager instance
            config: Audio configuration
            input_buffer: Ring buffer for input samples
            output_buffer: Ring buffer for output samples
        """
        self._device_manager = device_manager
        self._config = config
        self._input_buffer = input_buffer
        self._output_buffer = output_buffer
        
        self._stream: Optional[pyaudio.Stream] = None
        self._is_running = False
        
        # Metrics
        self._overrun_count = 0
        self._underrun_count = 0
        self._callback_count = 0
        
        # Passthrough mode (for testing)
        self._passthrough_mode = False
        
        # Lock for thread-safe access to metrics
        self._metrics_lock = threading.Lock()
        
        # Buffer safety management
        self._min_output_fill_samples = config.buffer_size * 2
        self._max_output_fill_samples = max(
            config.buffer_size * 8,
            output_buffer.capacity - config.buffer_size
        )
        self._needs_prefill = False
        self._adaptive_buffering = True
        self._underrun_adaptive_threshold = 3
    
    def _audio_callback(
        self,
        in_data: bytes,
        frame_count: int,
        time_info: dict,
        status: int
    ):
        """
        PyAudio callback function.
        
        Called from audio I/O thread - must be fast and non-blocking!
        """
        self._callback_count += 1
        
        # Check for stream errors
        if status:
            if status & pyaudio.paInputOverflow:
                with self._metrics_lock:
                    self._overrun_count += 1
            if status & pyaudio.paOutputUnderflow:
                with self._metrics_lock:
                    self._underrun_count += 1
        
        # Convert input bytes to numpy array
        input_samples = np.frombuffer(in_data, dtype=np.float32)
        
        # Push input samples to ring buffer
        pushed = self._input_buffer.push(input_samples)
        if pushed < len(input_samples):
            with self._metrics_lock:
                self._overrun_count += 1
        
        # Wait for output buffer to have a healthy fill before playback
        if not self._passthrough_mode and self._needs_prefill:
            if self._output_buffer.count >= self._min_output_fill_samples:
                self._needs_prefill = False
            else:
                # Feed silence until enough audio is prepared
                return (np.zeros(frame_count, dtype=np.float32).tobytes(), pyaudio.paContinue)
        
        # Get output samples
        if self._passthrough_mode:
            # Direct passthrough for testing
            output_samples = input_samples.copy()
        else:
            # Try to get processed samples from output buffer
            output_samples = self._output_buffer.pop(frame_count)
            
            if output_samples is None:
                # Buffer underrun - output silence or passthrough
                self._handle_underrun()
                output_samples = np.zeros(frame_count, dtype=np.float32)
        
        # Convert to bytes for output
        out_data = output_samples.astype(np.float32).tobytes()
        
        return (out_data, pyaudio.paContinue)
    
    def start(self, passthrough: bool = False):
        """
        Start the audio stream.
        
        Args:
            passthrough: If True, directly pass input to output (for testing)
        """
        if self._is_running:
            audio_logger.warning("Stream already running")
            return
        
        pa = self._device_manager.pyaudio_instance
        if pa is None:
            audio_logger.error("PyAudio not initialized")
            return
        
        self._passthrough_mode = passthrough
        
        # Clear buffers
        self._input_buffer.clear()
        self._output_buffer.clear()
        
        # Prefill output buffer with silence to avoid underruns at start
        if not passthrough:
            self._prefill_output_buffer(self._min_output_fill_samples)
            self._needs_prefill = True
        else:
            self._needs_prefill = False
        
        # Reset metrics
        with self._metrics_lock:
            self._overrun_count = 0
            self._underrun_count = 0
            self._callback_count = 0
        
        try:
            self._stream = pa.open(
                format=pyaudio.paFloat32,
                channels=self._config.channels,
                rate=self._config.sample_rate,
                input=True,
                output=True,
                input_device_index=self._config.input_device_index,
                output_device_index=self._config.output_device_index,
                frames_per_buffer=self._config.buffer_size,
                stream_callback=self._audio_callback
            )
            
            self._stream.start_stream()
            self._is_running = True
            
            mode = "passthrough" if passthrough else "processing"
            audio_logger.info(f"Audio stream started ({mode} mode)")
            ui_log_buffer.info(f"Audio stream started ({mode})", "AudioStream")
            
        except Exception as e:
            audio_logger.error(f"Failed to start audio stream: {e}")
            ui_log_buffer.error(f"Stream start failed: {e}", "AudioStream")
            raise
    
    def stop(self):
        """Stop the audio stream."""
        if not self._is_running:
            return
        
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                audio_logger.error(f"Error stopping stream: {e}")
            
            self._stream = None
        
        self._is_running = False
        audio_logger.info("Audio stream stopped")
        ui_log_buffer.info("Audio stream stopped", "AudioStream")
    
    @property
    def is_running(self) -> bool:
        return self._is_running
    
    @property
    def overrun_count(self) -> int:
        with self._metrics_lock:
            return self._overrun_count
    
    @property
    def underrun_count(self) -> int:
        with self._metrics_lock:
            return self._underrun_count
    
    @property
    def callback_count(self) -> int:
        return self._callback_count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current stream metrics."""
        with self._metrics_lock:
            return {
                'overrun_count': self._overrun_count,
                'underrun_count': self._underrun_count,
                'callback_count': self._callback_count,
                'is_running': self._is_running,
                'input_buffer_count': self._input_buffer.count,
                'output_buffer_count': self._output_buffer.count,
            }


    def _prefill_output_buffer(self, target_samples: int):
        """Fill the output buffer with silence up to the requested level."""
        samples_needed = max(0, min(target_samples, self._output_buffer.capacity) - self._output_buffer.count)
        if samples_needed > 0:
            silence = np.zeros(samples_needed, dtype=np.float32)
            self._output_buffer.push(silence)

    def _handle_underrun(self):
        """Record underrun and trigger adaptive buffering if needed."""
        request_prefill = False
        with self._metrics_lock:
            self._underrun_count += 1
            if (
                self._adaptive_buffering
                and self._underrun_count % self._underrun_adaptive_threshold == 0
            ):
                request_prefill = True
        
        if request_prefill:
            self._increase_prefill_window()
    
    def _increase_prefill_window(self):
        """Increase the minimum fill level as an adaptive response."""
        if self._passthrough_mode:
            return
        
        new_threshold = min(
            int(self._min_output_fill_samples * 1.5),
            self._max_output_fill_samples
        )
        
        if new_threshold > self._min_output_fill_samples:
            self._min_output_fill_samples = new_threshold
            self._needs_prefill = True
            # Allow DSP worker time to fill the larger buffer by padding with silence
            self._prefill_output_buffer(self._min_output_fill_samples)

def calculate_level_db(samples: np.ndarray, min_db: float = -60.0) -> float:
    """
    Calculate RMS level in dB.
    
    Args:
        samples: Audio samples
        min_db: Minimum dB value (for silence)
        
    Returns:
        Level in dB
    """
    if len(samples) == 0:
        return min_db
    
    rms = np.sqrt(np.mean(samples ** 2))
    
    if rms < 1e-10:
        return min_db
    
    db = 20 * np.log10(rms)
    return max(db, min_db)


def detect_clipping(samples: np.ndarray, threshold: float = 0.99) -> bool:
    """
    Detect if samples are clipping.
    
    Args:
        samples: Audio samples (assumed normalized to [-1, 1])
        threshold: Clipping threshold
        
    Returns:
        True if clipping detected
    """
    return np.any(np.abs(samples) >= threshold)
