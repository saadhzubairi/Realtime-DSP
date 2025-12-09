"""
Voice Transformation Application
================================

Real-time A→B voice transformation system.

Main entry point that coordinates:
- Audio I/O (PyAudio callback thread)
- DSP Worker (processing thread)
- UI (Tkinter main thread)

Architecture:
- Mic → Input Ring Buffer → DSP Worker → Output Ring Buffer → Speakers
- UI polls shared state via after() for updates
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import time
import numpy as np
from typing import Optional

# Project imports
from audio.ringbuffer import RingBuffer
from audio.pyaudio_io import (
    AudioDeviceManager, 
    AudioStream, 
    calculate_level_db,
    detect_clipping
)
from dsp.voice_profile import VoiceProfile
from dsp.transform import VoiceTransformPipeline
from ui.devices import DevicesTab
from ui.calibration import CalibrationTab
from ui.live import LiveTab
from ui.diagnostics import DiagnosticsTab
from utils.config import (
    AudioConfig,
    TransformConfig,
    AppState,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_BUFFER_SIZE,
    UI_UPDATE_INTERVAL_MS,
    samples_from_ms,
    FRAME_SIZE_MS,
    HOP_SIZE_MS,
)
from utils.timing import FrameTimer, LatencyEstimator
from utils.logging_utils import ui_log_buffer, audio_logger, dsp_logger


class DSPWorker(threading.Thread):
    """
    DSP Worker Thread.
    
    Pulls frames from input buffer, processes through transform pipeline,
    pushes results to output buffer.
    
    Uses simple FFT bin scaling for frequency shifting.
    """
    
    def __init__(
        self,
        input_buffer: RingBuffer,
        output_buffer: RingBuffer,
        audio_config: AudioConfig,
        transform_pipeline: VoiceTransformPipeline
    ):
        super().__init__(daemon=True)
        
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.audio_config = audio_config
        self.transform_pipeline = transform_pipeline
        
        self._running = False
        self._paused = True
        
        # Timing
        self.frame_timer = FrameTimer()
        
        # Metrics (thread-safe access via properties)
        self._metrics_lock = threading.Lock()
        self._current_f0 = 0.0
        self._is_voiced = False
        self._input_level_db = -60.0
        self._output_level_db = -60.0
        self._queue_depth = 0
    
    def run(self):
        """Main worker loop."""
        self._running = True
        dsp_logger.info("DSP Worker started")
        
        hop_size = self.audio_config.hop_size
        frame_size = self.audio_config.frame_size
        
        # Accumulator for input samples (for overlap)
        input_accumulator = np.zeros(0, dtype=np.float32)
        
        while self._running:
            if self._paused:
                time.sleep(0.01)
                continue
            
            # Check if we have enough input samples
            available = self.input_buffer.count
            
            if available >= hop_size:
                # Get hop_size samples
                samples = self.input_buffer.pop(hop_size)
                
                if samples is not None:
                    # Calculate input level
                    with self._metrics_lock:
                        self._input_level_db = calculate_level_db(samples)
                    
                    # Accumulate samples for frame processing
                    input_accumulator = np.concatenate([input_accumulator, samples])
                    
                    # Process if we have a full frame
                    if len(input_accumulator) >= frame_size:
                        # Extract frame
                        frame = input_accumulator[:frame_size]
                        
                        # Shift accumulator by hop (keep overlap)
                        input_accumulator = input_accumulator[hop_size:]
                        
                        # Time the processing
                        self.frame_timer.start()
                        
                        # Process through pipeline (includes OLA)
                        output_samples, metrics = self.transform_pipeline.process_with_overlap_add(frame)
                        
                        self.frame_timer.stop()
                        
                        # Update metrics
                        with self._metrics_lock:
                            self._current_f0 = metrics.get('f0', 0.0)
                            self._is_voiced = metrics.get('is_voiced', False)
                            self._output_level_db = calculate_level_db(output_samples)
                            self._queue_depth = self.output_buffer.count // hop_size
                        
                        # Push to output buffer
                        self.output_buffer.push(output_samples)
            else:
                # Wait a bit if no samples available
                time.sleep(0.001)
        
        dsp_logger.info("DSP Worker stopped")
    
    def start_processing(self):
        """Start processing."""
        self._paused = False
    
    def stop_processing(self):
        """Pause processing."""
        self._paused = True
        self.transform_pipeline.reset()
    
    def stop(self):
        """Stop the worker thread."""
        self._running = False
    
    @property
    def current_f0(self) -> float:
        with self._metrics_lock:
            return self._current_f0
    
    @property
    def is_voiced(self) -> bool:
        with self._metrics_lock:
            return self._is_voiced
    
    @property
    def input_level_db(self) -> float:
        with self._metrics_lock:
            return self._input_level_db
    
    @property
    def output_level_db(self) -> float:
        with self._metrics_lock:
            return self._output_level_db
    
    @property
    def queue_depth(self) -> int:
        with self._metrics_lock:
            return self._queue_depth
    
    @property
    def dsp_time_avg_ms(self) -> float:
        return self.frame_timer.avg_ms
    
    @property
    def dsp_time_max_ms(self) -> float:
        return self.frame_timer.max_ms


class VoiceTransformApp:
    """
    Main application class.
    
    Coordinates UI, audio I/O, and DSP processing.
    """
    
    def __init__(self):
        # Application state
        self.state = AppState()
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Voice Transform - Real-time A→B Conversion")
        self.root.geometry("800x700")
        self.root.minsize(700, 600)
        
        # Audio components
        self.device_manager = AudioDeviceManager()
        self.input_buffer: Optional[RingBuffer] = None
        self.output_buffer: Optional[RingBuffer] = None
        self.audio_stream: Optional[AudioStream] = None
        
        # DSP components
        self.transform_pipeline: Optional[VoiceTransformPipeline] = None
        self.dsp_worker: Optional[DSPWorker] = None
        
        # Latency estimator
        self.latency_estimator: Optional[LatencyEstimator] = None
        
        # Create UI
        self._create_ui()
        
        # Start UI update loop
        self._schedule_ui_update()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _create_ui(self):
        """Create the main UI with tabs."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Setup (Devices)
        self.devices_tab = DevicesTab(
            self.notebook,
            self.device_manager,
            self.state.audio_config,
            on_config_changed=self._on_audio_config_changed,
            on_test_loopback=self._on_test_loopback
        )
        self.notebook.add(self.devices_tab, text="1. Setup")
        
        # Tab 2: Calibration (simplified - handles recording internally)
        self.calibration_tab = CalibrationTab(
            self.notebook,
            self.state.audio_config,
            device_manager=self.device_manager
        )
        self.notebook.add(self.calibration_tab, text="2. Calibration")
        
        # Tab 3: Live Transform
        self.live_tab = LiveTab(
            self.notebook,
            on_start=self._on_start_transform,
            on_stop=self._on_stop_transform,
            on_config_changed=self._on_transform_config_changed
        )
        self.notebook.add(self.live_tab, text="3. Live Transform")
        
        # Tab 4: Diagnostics
        self.diagnostics_tab = DiagnosticsTab(
            self.notebook,
            on_bypass_changed=self._on_bypass_changed
        )
        self.notebook.add(self.diagnostics_tab, text="4. Diagnostics")
        
        # Status bar
        self.status_bar = ttk.Label(
            self.root, 
            text="Ready - Select audio devices in Setup tab",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    def _initialize_audio_system(self):
        """Initialize audio buffers and streams."""
        config = self.state.audio_config
        
        # Calculate buffer sizes
        # Ring buffer should hold several frames worth of samples
        buffer_samples = config.frame_size * 32
        
        self.input_buffer = RingBuffer(buffer_samples)
        self.output_buffer = RingBuffer(buffer_samples)
        
        # Create audio stream
        self.audio_stream = AudioStream(
            self.device_manager,
            config,
            self.input_buffer,
            self.output_buffer
        )
        
        # Create transform pipeline
        self.transform_pipeline = VoiceTransformPipeline(config)
        
        # Create DSP worker
        self.dsp_worker = DSPWorker(
            self.input_buffer,
            self.output_buffer,
            config,
            self.transform_pipeline
        )
        self.dsp_worker.start()
        
        # Create latency estimator
        self.latency_estimator = LatencyEstimator(
            config.hop_size,
            config.sample_rate
        )
        
        audio_logger.info("Audio system initialized")
        ui_log_buffer.info("Audio system initialized", "App")
    
    def _on_audio_config_changed(self, config: AudioConfig):
        """Handle audio configuration change."""
        self.state.audio_config = config
        if hasattr(self, 'status_bar'):
            self.status_bar.config(
                text=f"Config: {config.sample_rate} Hz, {config.buffer_size} samples buffer"
            )
    
    def _on_test_loopback(self, start: bool):
        """Handle loopback test toggle."""
        if start:
            self._initialize_audio_system()
            self.audio_stream.start(passthrough=True)
            self.status_bar.config(text="Loopback test active - speak into microphone")
        else:
            if self.audio_stream:
                self.audio_stream.stop()
            self.status_bar.config(text="Loopback test stopped")
    
    def _on_start_transform(self):
        """Start live transformation."""
        # Get profiles from calibration tab
        profile_a, profile_b = self.calibration_tab.get_profiles()
        
        if profile_a is None:
            messagebox.showwarning(
                "No Source Profile",
                "Please calibrate or load a source profile (Profile A) first."
            )
            self.live_tab._on_stop()
            return
        
        if profile_b is None:
            messagebox.showwarning(
                "No Target Profile",
                "Please calibrate or load a target profile (Profile B) first."
            )
            self.live_tab._on_stop()
            return
        
        # Initialize audio if needed
        if self.audio_stream is None:
            self._initialize_audio_system()
        
        # Set profiles
        self.transform_pipeline.set_profiles(profile_a, profile_b)
        self.live_tab.set_profiles(profile_a.name, profile_b.name)
        
        # Update transform config
        self.transform_pipeline.update_config(self.live_tab.get_config())
        
        # Start audio stream
        if not self.audio_stream.is_running:
            self.audio_stream.start(passthrough=False)
        
        # Start DSP processing
        self.dsp_worker.start_processing()
        
        # Lock device settings
        self.devices_tab.lock_settings()
        
        self.state.is_streaming = True
        self.status_bar.config(text="Live transform active")
        ui_log_buffer.info("Live transform started", "App")
    
    def _on_stop_transform(self):
        """Stop live transformation."""
        if self.dsp_worker:
            self.dsp_worker.stop_processing()
        
        if self.audio_stream and self.audio_stream.is_running:
            self.audio_stream.stop()
        
        # Unlock device settings
        self.devices_tab.unlock_settings()
        
        self.state.is_streaming = False
        self.status_bar.config(text="Transform stopped")
        ui_log_buffer.info("Live transform stopped", "App")
    
    def _on_transform_config_changed(self, config: TransformConfig):
        """Handle transform configuration change."""
        self.state.transform_config = config
        
        if self.transform_pipeline:
            self.transform_pipeline.update_config(config)
    
    def _on_bypass_changed(self, bypass_state: dict):
        """Handle debug bypass toggle changes."""
        if self.transform_pipeline:
            # Update transform config based on bypass
            config = self.live_tab.get_config()
            
            if bypass_state.get('bypass_pitch', False):
                config.pitch_strength = 0.0
            if bypass_state.get('bypass_envelope', False):
                config.envelope_strength = 0.0
            if bypass_state.get('bypass_formant', False):
                config.formant_strength = 0.0
            
            self.transform_pipeline.update_config(config)
    
    def _schedule_ui_update(self):
        """Schedule periodic UI updates."""
        self._update_ui()
        self.root.after(UI_UPDATE_INTERVAL_MS, self._schedule_ui_update)
    
    def _update_ui(self):
        """Update UI with current state."""
        # Update level meters if streaming
        if self.state.is_streaming and self.dsp_worker:
            # Update live tab meters
            self.live_tab.update_meters(
                input_level_db=self.dsp_worker.input_level_db,
                output_level_db=self.dsp_worker.output_level_db,
                f0=self.dsp_worker.current_f0,
                is_voiced=self.dsp_worker.is_voiced,
                underruns=self.audio_stream.underrun_count if self.audio_stream else 0,
                overruns=self.audio_stream.overrun_count if self.audio_stream else 0,
                queue_depth=self.dsp_worker.queue_depth
            )
            
            # Update diagnostics
            hop_time_ms = samples_from_ms(HOP_SIZE_MS, self.state.audio_config.sample_rate) / \
                         self.state.audio_config.sample_rate * 1000
            
            # Calculate latency
            if self.latency_estimator:
                self.latency_estimator.update(
                    self.input_buffer.count if self.input_buffer else 0,
                    self.output_buffer.count if self.output_buffer else 0,
                    self.state.audio_config.buffer_size
                )
                latency_ms = self.latency_estimator.estimated_latency_ms
            else:
                latency_ms = 0
            
            buffer_ms = self.state.audio_config.buffer_size / self.state.audio_config.sample_rate * 1000
            queue_ms = self.dsp_worker.queue_depth * hop_time_ms
            
            self.diagnostics_tab.update_metrics(
                latency_ms=latency_ms,
                input_buffer_ms=buffer_ms,
                queue_ms=queue_ms,
                output_buffer_ms=buffer_ms,
                dsp_avg_ms=self.dsp_worker.dsp_time_avg_ms,
                dsp_max_ms=self.dsp_worker.dsp_time_max_ms,
                hop_time_ms=hop_time_ms
            )
        
        # Update level meter in devices tab during loopback (only when not streaming)
        # During streaming, the DSP worker provides levels - avoid double-polling the ring buffer
        if self.audio_stream and self.audio_stream.is_running and not self.state.is_streaming:
            if self.input_buffer and self.input_buffer.count > 256:
                samples = self.input_buffer.peek(256)
                if samples is not None:
                    level = calculate_level_db(samples)
                    clipping = detect_clipping(samples)
                    self.devices_tab.update_level_meter(level, clipping)
        
        # Process log buffer
        messages = ui_log_buffer.get_all()
        if messages:
            self.diagnostics_tab.add_logs_from_buffer(messages)
    
    def _on_close(self):
        """Handle window close."""
        # Stop everything
        if self.dsp_worker:
            self.dsp_worker.stop()
            self.dsp_worker.join(timeout=1.0)
        
        if self.audio_stream:
            self.audio_stream.stop()
        
        self.device_manager.terminate()
        
        self.root.destroy()
    
    def run(self):
        """Run the application."""
        self.root.mainloop()


def main():
    """Main entry point."""
    app = VoiceTransformApp()
    app.run()


if __name__ == "__main__":
    main()
