"""
WORLD Voice Changer Application
================================

Real-time voice transformation using the WORLD vocoder.

WORLD provides excellent quality for pitch and formant shifting
while maintaining natural sound - much better than Griffin-Lim.

Usage:
1. Select audio devices
2. Adjust pitch shift and formant shift
3. Start streaming - your voice will be transformed in real-time!
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import numpy as np
from typing import Optional

# Audio I/O
from audio.ringbuffer import RingBuffer
from audio.pyaudio_io import (
    AudioDeviceManager,
    AudioStream,
    calculate_level_db,
)

# WORLD Vocoder
from dsp.world_vocoder import RealtimeWorldVocoder

# Utils
from utils.config import AudioConfig


class WorldWorker(threading.Thread):
    """
    Background worker for WORLD voice transformation.
    
    Pulls audio from input buffer, transforms through WORLD, pushes to output buffer.
    """
    
    def __init__(
        self,
        input_buffer: RingBuffer,
        output_buffer: RingBuffer,
        sample_rate: int = 16000,
        block_size: int = 3200  # 200ms at 16kHz - WORLD needs longer blocks
    ):
        super().__init__(daemon=True)
        
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.block_size = block_size
        
        # WORLD vocoder
        self.vocoder = RealtimeWorldVocoder(
            sample_rate=sample_rate,
            block_size=block_size,
            overlap=0.5
        )
        
        self._running = False
        self._paused = True
        self._bypass = False
        
        # Metrics
        self._lock = threading.Lock()
        self._input_level_db = -60.0
        self._output_level_db = -60.0
        self._process_time_ms = 0.0
    
    def run(self):
        """Main processing loop."""
        self._running = True
        
        while self._running:
            if self._paused:
                time.sleep(0.01)
                continue
            
            # Check for available input
            available = self.input_buffer.count
            
            if available >= self.block_size:
                samples = self.input_buffer.pop(self.block_size)
                
                if samples is not None:
                    # Update input level
                    with self._lock:
                        self._input_level_db = calculate_level_db(samples)
                    
                    if self._bypass:
                        # Bypass mode - direct passthrough
                        output = samples
                    else:
                        # Transform through WORLD
                        output = self.vocoder.process(samples)
                        
                        with self._lock:
                            self._process_time_ms = self.vocoder.process_time_ms
                    
                    # Update output level
                    with self._lock:
                        self._output_level_db = calculate_level_db(output)
                    
                    # Push to output
                    self.output_buffer.push(output)
            else:
                time.sleep(0.005)
    
    def start_processing(self):
        """Start processing."""
        self._paused = False
    
    def stop_processing(self):
        """Pause processing."""
        self._paused = True
        self.vocoder.reset()
    
    def set_bypass(self, bypass: bool):
        """Enable/disable bypass mode."""
        self._bypass = bypass
        self.vocoder.set_enabled(not bypass)
    
    def set_pitch_shift(self, semitones: float):
        """Set pitch shift in semitones."""
        self.vocoder.set_pitch_shift(semitones)
    
    def set_formant_shift(self, ratio: float):
        """Set formant shift ratio."""
        self.vocoder.set_formant_shift(ratio)
    
    def stop(self):
        """Stop the worker thread."""
        self._running = False
    
    @property
    def input_level_db(self) -> float:
        with self._lock:
            return self._input_level_db
    
    @property
    def output_level_db(self) -> float:
        with self._lock:
            return self._output_level_db
    
    @property
    def process_time_ms(self) -> float:
        with self._lock:
            return self._process_time_ms


class WorldVoiceChangerApp:
    """
    WORLD Voice Changer Application.
    
    Simple UI for real-time voice transformation.
    """
    
    def __init__(self):
        # Create main window
        self.root = tk.Tk()
        self.root.title("WORLD Voice Changer - High Quality Voice Transformation")
        self.root.geometry("650x550")
        self.root.minsize(550, 450)
        
        # Audio components
        self.device_manager = AudioDeviceManager()
        self.audio_config = AudioConfig(sample_rate=16000, buffer_size=512)
        
        self.input_buffer: Optional[RingBuffer] = None
        self.output_buffer: Optional[RingBuffer] = None
        self.audio_stream: Optional[AudioStream] = None
        self.worker: Optional[WorldWorker] = None
        
        # State
        self._is_streaming = False
        
        # Build UI
        self._create_ui()
        
        # Start UI update loop
        self._schedule_ui_update()
        
        # Handle close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _create_ui(self):
        """Create the UI."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = ttk.Label(
            main_frame,
            text="🎤 WORLD Voice Changer",
            font=('Helvetica', 16, 'bold')
        )
        title.pack(pady=(0, 5))
        
        subtitle = ttk.Label(
            main_frame,
            text="High-quality pitch & formant shifting using WORLD vocoder",
            foreground='gray'
        )
        subtitle.pack(pady=(0, 15))
        
        # Device section
        device_frame = ttk.LabelFrame(main_frame, text="Audio Devices", padding="10")
        device_frame.pack(fill=tk.X, pady=5)
        
        # Input device
        input_row = ttk.Frame(device_frame)
        input_row.pack(fill=tk.X, pady=2)
        ttk.Label(input_row, text="Microphone:", width=12).pack(side=tk.LEFT)
        
        self.input_var = tk.StringVar()
        input_devices = self.device_manager.get_input_devices()
        input_names = [d.name for d in input_devices]
        self.input_combo = ttk.Combobox(
            input_row,
            textvariable=self.input_var,
            values=input_names,
            state='readonly',
            width=50
        )
        self.input_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        if input_names:
            self.input_combo.current(0)
        
        # Output device
        output_row = ttk.Frame(device_frame)
        output_row.pack(fill=tk.X, pady=2)
        ttk.Label(output_row, text="Speakers:", width=12).pack(side=tk.LEFT)
        
        self.output_var = tk.StringVar()
        output_devices = self.device_manager.get_output_devices()
        output_names = [d.name for d in output_devices]
        self.output_combo = ttk.Combobox(
            output_row,
            textvariable=self.output_var,
            values=output_names,
            state='readonly',
            width=50
        )
        self.output_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        if output_names:
            self.output_combo.current(0)
        
        # Controls section
        controls_frame = ttk.LabelFrame(main_frame, text="Voice Transformation", padding="10")
        controls_frame.pack(fill=tk.X, pady=5)
        
        # Pitch shift
        pitch_row = ttk.Frame(controls_frame)
        pitch_row.pack(fill=tk.X, pady=5)
        ttk.Label(pitch_row, text="Pitch Shift:", width=12).pack(side=tk.LEFT)
        
        self.pitch_var = tk.DoubleVar(value=0.0)
        self.pitch_scale = ttk.Scale(
            pitch_row,
            from_=-12,
            to=12,
            orient=tk.HORIZONTAL,
            variable=self.pitch_var,
            command=self._on_pitch_changed
        )
        self.pitch_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.pitch_label = ttk.Label(pitch_row, text="0 semitones", width=15)
        self.pitch_label.pack(side=tk.LEFT)
        
        # Formant shift
        formant_row = ttk.Frame(controls_frame)
        formant_row.pack(fill=tk.X, pady=5)
        ttk.Label(formant_row, text="Formant Shift:", width=12).pack(side=tk.LEFT)
        
        self.formant_var = tk.DoubleVar(value=1.0)
        self.formant_scale = ttk.Scale(
            formant_row,
            from_=0.5,
            to=2.0,
            orient=tk.HORIZONTAL,
            variable=self.formant_var,
            command=self._on_formant_changed
        )
        self.formant_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.formant_label = ttk.Label(formant_row, text="1.00x", width=15)
        self.formant_label.pack(side=tk.LEFT)
        
        # Presets
        preset_row = ttk.Frame(controls_frame)
        preset_row.pack(fill=tk.X, pady=10)
        ttk.Label(preset_row, text="Presets:", width=12).pack(side=tk.LEFT)
        
        ttk.Button(
            preset_row,
            text="Male→Female",
            command=lambda: self._apply_preset(6, 1.15)
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            preset_row,
            text="Female→Male",
            command=lambda: self._apply_preset(-6, 0.85)
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            preset_row,
            text="Chipmunk",
            command=lambda: self._apply_preset(12, 1.5)
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            preset_row,
            text="Deep",
            command=lambda: self._apply_preset(-6, 0.7)
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            preset_row,
            text="Reset",
            command=lambda: self._apply_preset(0, 1.0)
        ).pack(side=tk.LEFT, padx=2)
        
        # Bypass toggle
        bypass_row = ttk.Frame(controls_frame)
        bypass_row.pack(fill=tk.X, pady=5)
        
        self.bypass_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            bypass_row,
            text="Bypass (passthrough - no processing)",
            variable=self.bypass_var,
            command=self._on_bypass_changed
        ).pack(side=tk.LEFT)
        
        # Start/Stop buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=15)
        
        self.start_btn = ttk.Button(
            button_frame,
            text="▶ Start",
            command=self._start_streaming
        )
        self.start_btn.pack(side=tk.LEFT, padx=10)
        
        self.stop_btn = ttk.Button(
            button_frame,
            text="■ Stop",
            command=self._stop_streaming,
            state='disabled'
        )
        self.stop_btn.pack(side=tk.LEFT, padx=10)
        
        self.status_label = ttk.Label(
            button_frame,
            text="● Stopped",
            foreground='gray'
        )
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Meters section
        meters_frame = ttk.LabelFrame(main_frame, text="Levels & Stats", padding="10")
        meters_frame.pack(fill=tk.X, pady=5)
        
        # Input meter
        input_meter_row = ttk.Frame(meters_frame)
        input_meter_row.pack(fill=tk.X, pady=2)
        ttk.Label(input_meter_row, text="Input:", width=10).pack(side=tk.LEFT)
        self.input_meter = ttk.Progressbar(
            input_meter_row,
            length=400,
            mode='determinate'
        )
        self.input_meter.pack(side=tk.LEFT, padx=5)
        self.input_db_label = ttk.Label(input_meter_row, text="-60 dB", width=10)
        self.input_db_label.pack(side=tk.LEFT)
        
        # Output meter
        output_meter_row = ttk.Frame(meters_frame)
        output_meter_row.pack(fill=tk.X, pady=2)
        ttk.Label(output_meter_row, text="Output:", width=10).pack(side=tk.LEFT)
        self.output_meter = ttk.Progressbar(
            output_meter_row,
            length=400,
            mode='determinate'
        )
        self.output_meter.pack(side=tk.LEFT, padx=5)
        self.output_db_label = ttk.Label(output_meter_row, text="-60 dB", width=10)
        self.output_db_label.pack(side=tk.LEFT)
        
        # Stats
        stats_row = ttk.Frame(meters_frame)
        stats_row.pack(fill=tk.X, pady=5)
        self.stats_label = ttk.Label(
            stats_row,
            text="Process time: -- ms | Block size: 200 ms",
            foreground='gray'
        )
        self.stats_label.pack(side=tk.LEFT)
        
        self.realtime_label = ttk.Label(stats_row, text="", foreground='gray')
        self.realtime_label.pack(side=tk.RIGHT)
        
        # Status bar
        self.status_bar = ttk.Label(
            self.root,
            text="Ready - Click Start to begin voice transformation",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    def _apply_preset(self, pitch: float, formant: float):
        """Apply a voice preset."""
        self.pitch_var.set(pitch)
        self.formant_var.set(formant)
        self._on_pitch_changed(pitch)
        self._on_formant_changed(formant)
    
    def _start_streaming(self):
        """Start real-time voice transformation."""
        # Get selected devices
        input_devices = self.device_manager.get_input_devices()
        output_devices = self.device_manager.get_output_devices()
        
        input_idx = self.input_combo.current()
        output_idx = self.output_combo.current()
        
        if input_idx >= 0 and input_idx < len(input_devices):
            self.audio_config.input_device_index = input_devices[input_idx].index
        if output_idx >= 0 and output_idx < len(output_devices):
            self.audio_config.output_device_index = output_devices[output_idx].index
        
        # Initialize audio system
        buffer_samples = self.audio_config.sample_rate * 2  # 2 seconds buffer
        self.input_buffer = RingBuffer(buffer_samples)
        self.output_buffer = RingBuffer(buffer_samples)
        
        self.audio_stream = AudioStream(
            self.device_manager,
            self.audio_config,
            self.input_buffer,
            self.output_buffer
        )
        
        # Create worker with 200ms blocks (WORLD needs longer blocks)
        block_size = int(0.2 * self.audio_config.sample_rate)  # 200ms
        self.worker = WorldWorker(
            self.input_buffer,
            self.output_buffer,
            sample_rate=self.audio_config.sample_rate,
            block_size=block_size
        )
        
        # Apply current settings
        self.worker.set_pitch_shift(self.pitch_var.get())
        self.worker.set_formant_shift(self.formant_var.get())
        self.worker.set_bypass(self.bypass_var.get())
        
        # Start
        self.worker.start()
        self.worker.start_processing()
        self.audio_stream.start(passthrough=False)
        
        self._is_streaming = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_label.config(text="● Running", foreground='green')
        self.status_bar.config(text="Streaming - speak into the microphone (use headphones!)")
    
    def _stop_streaming(self):
        """Stop streaming."""
        if self.worker:
            self.worker.stop_processing()
            self.worker.stop()
        
        if self.audio_stream:
            self.audio_stream.stop()
        
        self._is_streaming = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="● Stopped", foreground='gray')
        self.status_bar.config(text="Stopped")
    
    def _on_pitch_changed(self, value):
        """Handle pitch slider change."""
        pitch = float(value)
        self.pitch_label.config(text=f"{pitch:+.1f} semitones")
        
        if self.worker:
            self.worker.set_pitch_shift(pitch)
    
    def _on_formant_changed(self, value):
        """Handle formant slider change."""
        formant = float(value)
        self.formant_label.config(text=f"{formant:.2f}x")
        
        if self.worker:
            self.worker.set_formant_shift(formant)
    
    def _on_bypass_changed(self):
        """Handle bypass toggle."""
        if self.worker:
            self.worker.set_bypass(self.bypass_var.get())
    
    def _schedule_ui_update(self):
        """Schedule periodic UI updates."""
        self._update_ui()
        self.root.after(50, self._schedule_ui_update)
    
    def _update_ui(self):
        """Update UI with current state."""
        if self._is_streaming and self.worker:
            # Update meters
            input_db = self.worker.input_level_db
            output_db = self.worker.output_level_db
            
            input_pct = max(0, min(100, (input_db + 60) / 60 * 100))
            output_pct = max(0, min(100, (output_db + 60) / 60 * 100))
            
            self.input_meter['value'] = input_pct
            self.output_meter['value'] = output_pct
            
            self.input_db_label.config(text=f"{input_db:.1f} dB")
            self.output_db_label.config(text=f"{output_db:.1f} dB")
            
            # Update stats
            process_time = self.worker.process_time_ms
            block_time = 200  # 200ms blocks
            
            self.stats_label.config(
                text=f"Process time: {process_time:.1f} ms | Block size: {block_time} ms"
            )
            
            if process_time > 0:
                if process_time < block_time * 0.8:
                    self.realtime_label.config(text="✓ Real-time OK", foreground='green')
                elif process_time < block_time:
                    self.realtime_label.config(text="⚠ Marginal", foreground='orange')
                else:
                    self.realtime_label.config(text="✗ Too slow", foreground='red')
            else:
                self.realtime_label.config(text="○ Waiting", foreground='gray')
    
    def _on_close(self):
        """Handle window close."""
        self._stop_streaming()
        self.device_manager.terminate()
        self.root.destroy()
    
    def run(self):
        """Run the application."""
        self.root.mainloop()


def main():
    """Main entry point."""
    print("Starting WORLD Voice Changer...")
    print("Using WORLD vocoder for high-quality voice transformation")
    print("Tip: Use headphones to prevent feedback!")
    app = WorldVoiceChangerApp()
    app.run()


if __name__ == "__main__":
    main()
