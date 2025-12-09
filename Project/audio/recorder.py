"""
Simple audio recorder module using PyAudio blocking mode.
Records to WAV files for reliable, lag-free calibration.
"""

import pyaudio
import wave
import numpy as np
import threading
import time
import os
from typing import Optional, Callable
from dataclasses import dataclass

from utils.config import DEFAULT_SAMPLE_RATE, get_profiles_directory


@dataclass
class RecordingState:
    """State of current recording."""
    is_recording: bool = False
    elapsed_seconds: float = 0.0
    max_level: float = 0.0
    filepath: Optional[str] = None


class AudioRecorder:
    """
    Simple audio recorder using PyAudio blocking mode in a background thread.
    Records directly to WAV file - no ring buffer contention.
    """
    
    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = 1,
        chunk_size: int = 1024,
        device_index: Optional[int] = None
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index
        
        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._thread: Optional[threading.Thread] = None
        
        self.state = RecordingState()
        self._stop_flag = threading.Event()
        
        # Callbacks
        self.on_level_update: Optional[Callable[[float], None]] = None
        self.on_recording_done: Optional[Callable[[str], None]] = None
    
    def start_recording(self, filepath: str, duration_seconds: float = 5.0):
        """
        Start recording audio to a WAV file.
        
        Args:
            filepath: Path to save the WAV file
            duration_seconds: Maximum recording duration
        """
        if self.state.is_recording:
            return
        
        self.state = RecordingState(
            is_recording=True,
            elapsed_seconds=0.0,
            max_level=0.0,
            filepath=filepath
        )
        self._stop_flag.clear()
        
        # Start recording thread
        self._thread = threading.Thread(
            target=self._record_thread,
            args=(filepath, duration_seconds),
            daemon=True
        )
        self._thread.start()
    
    def stop_recording(self):
        """Stop the current recording."""
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def _record_thread(self, filepath: str, duration_seconds: float):
        """Recording thread - runs in background."""
        try:
            self._pa = pyaudio.PyAudio()
            
            # Open stream
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size
            )
            
            # Open WAV file
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            wf = wave.open(filepath, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            
            start_time = time.time()
            
            while not self._stop_flag.is_set():
                elapsed = time.time() - start_time
                self.state.elapsed_seconds = elapsed
                
                if elapsed >= duration_seconds:
                    break
                
                # Read audio chunk
                try:
                    data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                    wf.writeframes(data)
                    
                    # Calculate level for UI feedback
                    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    level = np.max(np.abs(samples))
                    self.state.max_level = max(self.state.max_level, level)
                    
                    if self.on_level_update:
                        self.on_level_update(level)
                        
                except Exception as e:
                    print(f"Read error: {e}")
                    break
            
            # Cleanup
            wf.close()
            self._stream.stop_stream()
            self._stream.close()
            self._pa.terminate()
            
            self.state.is_recording = False
            
            if self.on_recording_done:
                self.on_recording_done(filepath)
                
        except Exception as e:
            print(f"Recording error: {e}")
            self.state.is_recording = False


class AudioPlayer:
    """Simple audio player for WAV files."""
    
    def __init__(self, device_index: Optional[int] = None):
        self.device_index = device_index
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self.is_playing = False
    
    def play_file(self, filepath: str):
        """Play a WAV file in background thread."""
        if self.is_playing:
            self.stop()
        
        self._stop_flag.clear()
        self.is_playing = True
        
        self._thread = threading.Thread(
            target=self._play_thread,
            args=(filepath,),
            daemon=True
        )
        self._thread.start()
    
    def play_array(self, audio: np.ndarray, sample_rate: int):
        """Play a numpy array in background thread."""
        if self.is_playing:
            self.stop()
        
        self._stop_flag.clear()
        self.is_playing = True
        
        self._thread = threading.Thread(
            target=self._play_array_thread,
            args=(audio, sample_rate),
            daemon=True
        )
        self._thread.start()
    
    def stop(self):
        """Stop playback."""
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self.is_playing = False
    
    def _play_thread(self, filepath: str):
        """Playback thread for WAV files."""
        try:
            wf = wave.open(filepath, 'rb')
            pa = pyaudio.PyAudio()
            
            stream = pa.open(
                format=pa.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
                output_device_index=self.device_index
            )
            
            chunk_size = 1024
            data = wf.readframes(chunk_size)
            
            while data and not self._stop_flag.is_set():
                stream.write(data)
                data = wf.readframes(chunk_size)
            
            stream.stop_stream()
            stream.close()
            pa.terminate()
            wf.close()
            
        except Exception as e:
            print(f"Playback error: {e}")
        finally:
            self.is_playing = False
    
    def _play_array_thread(self, audio: np.ndarray, sample_rate: int):
        """Playback thread for numpy arrays."""
        try:
            pa = pyaudio.PyAudio()
            
            # Convert to int16
            audio_int16 = (audio * 32767).astype(np.int16)
            
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True,
                output_device_index=self.device_index
            )
            
            chunk_size = 1024
            for i in range(0, len(audio_int16), chunk_size):
                if self._stop_flag.is_set():
                    break
                chunk = audio_int16[i:i+chunk_size].tobytes()
                stream.write(chunk)
            
            stream.stop_stream()
            stream.close()
            pa.terminate()
            
        except Exception as e:
            print(f"Playback error: {e}")
        finally:
            self.is_playing = False


def load_wav_as_float(filepath: str) -> tuple:
    """
    Load a WAV file and return as float32 numpy array.
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    wf = wave.open(filepath, 'rb')
    sample_rate = wf.getframerate()
    n_frames = wf.getnframes()
    
    data = wf.readframes(n_frames)
    wf.close()
    
    # Convert to float32
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    
    return audio, sample_rate


def save_wav_from_float(filepath: str, audio: np.ndarray, sample_rate: int):
    """
    Save a float32 numpy array to WAV file.
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    
    wf = wave.open(filepath, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(sample_rate)
    wf.writeframes(audio_int16.tobytes())
    wf.close()
