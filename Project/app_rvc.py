"""
RVC Voice Changer Application
==============================

Real-time voice conversion using RVC (Retrieval-based Voice Conversion).

Key differences from the DSP-based approach:
- Only needs target voice model (Person B) - no source profile required
- Uses neural network for high-quality voice conversion
- ~50-100ms latency with GPU acceleration
- Much higher audio quality

Usage:
1. Load an RVC voice model (.pth file)
2. Select audio devices
3. Start streaming - your voice will be converted in real-time!
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import numpy as np
from typing import Optional
from pathlib import Path

# Audio I/O
from audio.ringbuffer import RingBuffer
from audio.pyaudio_io import (
    AudioDeviceManager,
    AudioStream,
    calculate_level_db,
)

# RVC
from rvc.inference import RVCVoiceConverter, RVCConfig, list_available_models, VoiceModel
from rvc.realtime import RealtimeRVCPipeline, RealtimeConfig

# Utils
from utils.config import AudioConfig, DEFAULT_SAMPLE_RATE
from utils.logging_utils import audio_logger


class RVCWorker(threading.Thread):
    """
    Background worker for RVC voice conversion.
    
    Pulls audio from input buffer, converts through RVC, pushes to output buffer.
    """
    
    def __init__(
        self,
        input_buffer: RingBuffer,
        output_buffer: RingBuffer,
        pipeline: RealtimeRVCPipeline,
        block_size: int = 1600  # 100ms at 16kHz
    ):
        super().__init__(daemon=True)
        
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.pipeline = pipeline
        self.block_size = block_size
        
        self._running = False
        self._paused = True
        self._bypass = False
        
        # Metrics
        self._lock = threading.Lock()
        self._input_level_db = -60.0
        self._output_level_db = -60.0
        self._process_time_ms = 0.0
        self._is_converting = False
    
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
                        # Convert through RVC
                        output = self.pipeline.push_audio(samples)
                        
                        if output is None:
                            # No output ready yet (buffering)
                            output = np.zeros(self.block_size, dtype=np.float32)
                            with self._lock:
                                self._is_converting = False
                        else:
                            with self._lock:
                                self._is_converting = True
                                self._process_time_ms = self.pipeline.process_time_ms
                    
                    # Update output level
                    with self._lock:
                        self._output_level_db = calculate_level_db(output)
                    
                    # Push to output
                    self.output_buffer.push(output)
            else:
                time.sleep(0.001)
    
    def start_processing(self):
        """Start processing."""
        self._paused = False
    
    def stop_processing(self):
        """Pause processing."""
        self._paused = True
        self.pipeline.reset()
    
    def set_bypass(self, bypass: bool):
        """Enable/disable bypass mode."""
        self._bypass = bypass
    
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
    
    @property
    def is_converting(self) -> bool:
        with self._lock:
            return self._is_converting


class RVCApp:
    """
    RVC Voice Changer Application.
    
    Simple UI for real-time voice conversion using RVC models.
    """
    
    def __init__(self):
        # Create main window
        self.root = tk.Tk()
        self.root.title("RVC Voice Changer - Real-time Voice Conversion")
        self.root.geometry("700x600")
        self.root.minsize(600, 500)
        
        # Audio components
        self.device_manager = AudioDeviceManager()
        self.audio_config = AudioConfig(sample_rate=16000, buffer_size=512)
        
        self.input_buffer: Optional[RingBuffer] = None
        self.output_buffer: Optional[RingBuffer] = None
        self.audio_stream: Optional[AudioStream] = None
        
        # RVC components
        self.rvc_config = RVCConfig(
            device='cuda:0',
            f0_method='rmvpe',
            pitch_shift=0
        )
        self.rvc_converter: Optional[RVCVoiceConverter] = None
        self.rvc_pipeline: Optional[RealtimeRVCPipeline] = None
        self.worker: Optional[RVCWorker] = None
        
        # State
        self._is_streaming = False
        self._model_loaded = False
        
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
            text="🎤 RVC Voice Changer",
            font=('Helvetica', 16, 'bold')
        )
        title.pack(pady=(0, 10))
        
        subtitle = ttk.Label(
            main_frame,
            text="Real-time voice conversion using AI",
            foreground='gray'
        )
        subtitle.pack(pady=(0, 15))
        
        # Model section
        model_frame = ttk.LabelFrame(main_frame, text="Voice Model", padding="10")
        model_frame.pack(fill=tk.X, pady=5)
        
        model_row = ttk.Frame(model_frame)
        model_row.pack(fill=tk.X)
        
        self.model_label = ttk.Label(model_row, text="No model loaded", width=40)
        self.model_label.pack(side=tk.LEFT)
        
        ttk.Button(
            model_row,
            text="Load Model...",
            command=self._load_model
        ).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(
            model_row,
            text="Browse Models",
            command=self._browse_models
        ).pack(side=tk.RIGHT)
        
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
        controls_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        controls_frame.pack(fill=tk.X, pady=5)
        
        # Pitch shift
        pitch_row = ttk.Frame(controls_frame)
        pitch_row.pack(fill=tk.X, pady=5)
        ttk.Label(pitch_row, text="Pitch Shift:", width=12).pack(side=tk.LEFT)
        
        self.pitch_var = tk.IntVar(value=0)
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
        
        # F0 method
        f0_row = ttk.Frame(controls_frame)
        f0_row.pack(fill=tk.X, pady=5)
        ttk.Label(f0_row, text="Pitch Method:", width=12).pack(side=tk.LEFT)
        
        self.f0_var = tk.StringVar(value='rmvpe')
        f0_methods = ['rmvpe', 'harvest', 'crepe', 'pm']
        self.f0_combo = ttk.Combobox(
            f0_row,
            textvariable=self.f0_var,
            values=f0_methods,
            state='readonly',
            width=15
        )
        self.f0_combo.pack(side=tk.LEFT, padx=5)
        self.f0_combo.bind('<<ComboboxSelected>>', self._on_f0_changed)
        
        ttk.Label(
            f0_row,
            text="rmvpe=best quality, pm=fastest",
            foreground='gray'
        ).pack(side=tk.LEFT, padx=10)
        
        # Bypass toggle
        bypass_row = ttk.Frame(controls_frame)
        bypass_row.pack(fill=tk.X, pady=5)
        
        self.bypass_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            bypass_row,
            text="Bypass (passthrough)",
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
        meters_frame = ttk.LabelFrame(main_frame, text="Levels", padding="10")
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
            text="Process time: -- ms | Latency: -- ms",
            foreground='gray'
        )
        self.stats_label.pack(side=tk.LEFT)
        
        self.realtime_label = ttk.Label(stats_row, text="", foreground='gray')
        self.realtime_label.pack(side=tk.RIGHT)
        
        # Status bar
        self.status_bar = ttk.Label(
            self.root,
            text="Ready - Load a voice model to begin",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    def _load_model(self):
        """Load an RVC model file."""
        filepath = filedialog.askopenfilename(
            title="Select RVC Model",
            filetypes=[
                ("RVC Models", "*.pth"),
                ("All Files", "*.*")
            ],
            initialdir="rvc_models"
        )
        
        if not filepath:
            return
        
        self._load_model_file(filepath)
    
    def _load_model_file(self, filepath: str):
        """Load a model from a file path."""
        self.status_bar.config(text=f"Loading model: {Path(filepath).name}...")
        self.root.update()
        
        try:
            # Initialize RVC if needed
            if self.rvc_converter is None:
                self.rvc_converter = RVCVoiceConverter(self.rvc_config)
            
            # Look for index file
            model_dir = Path(filepath).parent
            index_path = None
            for idx_file in model_dir.glob("*.index"):
                index_path = str(idx_file)
                break
            
            # Load model
            success = self.rvc_converter.load_model(filepath, index_path)
            
            if success:
                model_name = Path(filepath).stem
                self.model_label.config(text=f"✓ {model_name}")
                self._model_loaded = True
                self.status_bar.config(text=f"Model loaded: {model_name}")
                
                # Initialize pipeline
                rt_config = RealtimeConfig(
                    input_sample_rate=self.audio_config.sample_rate,
                    output_sample_rate=self.audio_config.sample_rate,
                    block_time=0.1,
                    use_async=True
                )
                self.rvc_pipeline = RealtimeRVCPipeline(self.rvc_converter, rt_config)
            else:
                messagebox.showerror("Error", "Failed to load model")
                self.status_bar.config(text="Failed to load model")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")
            self.status_bar.config(text=f"Error: {e}")
    
    def _browse_models(self):
        """Open the models directory."""
        models_dir = Path("rvc_models")
        models_dir.mkdir(exist_ok=True)
        
        # List available models
        models = list_available_models(str(models_dir))
        
        if not models:
            messagebox.showinfo(
                "No Models Found",
                f"No models found in {models_dir.absolute()}\n\n"
                "To add models:\n"
                "1. Create a folder for each model\n"
                "2. Place the .pth file inside\n"
                "3. Optionally add a .index file for better quality"
            )
            return
        
        # Show model selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Voice Model")
        dialog.geometry("400x300")
        
        ttk.Label(dialog, text="Available Models:").pack(pady=10)
        
        listbox = tk.Listbox(dialog, width=50, height=10)
        listbox.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        for model in models:
            has_index = "✓" if model.index_path else ""
            listbox.insert(tk.END, f"{model.name} {has_index}")
        
        def on_select():
            selection = listbox.curselection()
            if selection:
                model = models[selection[0]]
                dialog.destroy()
                self._load_model_file(model.path)
        
        ttk.Button(dialog, text="Load", command=on_select).pack(pady=10)
    
    def _start_streaming(self):
        """Start real-time voice conversion."""
        if not self._model_loaded:
            messagebox.showwarning(
                "No Model",
                "Please load a voice model first."
            )
            return
        
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
        
        # Create worker
        block_size = int(0.1 * self.audio_config.sample_rate)  # 100ms blocks
        self.worker = RVCWorker(
            self.input_buffer,
            self.output_buffer,
            self.rvc_pipeline,
            block_size=block_size
        )
        
        # Start
        self.worker.start()
        self.worker.start_processing()
        self.audio_stream.start(passthrough=False)
        
        self._is_streaming = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_label.config(text="● Running", foreground='green')
        self.status_bar.config(text="Streaming - speak into the microphone")
    
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
        pitch = int(float(value))
        self.pitch_label.config(text=f"{pitch:+d} semitones")
        
        if self.rvc_converter:
            self.rvc_converter.set_pitch_shift(pitch)
    
    def _on_f0_changed(self, event):
        """Handle F0 method change."""
        method = self.f0_var.get()
        if self.rvc_converter:
            try:
                self.rvc_converter.set_f0_method(method)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to set F0 method: {e}")
    
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
            block_time = 100  # 100ms blocks
            
            self.stats_label.config(
                text=f"Process time: {process_time:.1f} ms | Block size: {block_time} ms"
            )
            
            if self.worker.is_converting:
                if process_time < block_time:
                    self.realtime_label.config(text="✓ Real-time", foreground='green')
                else:
                    self.realtime_label.config(text="⚠ Slow", foreground='orange')
            else:
                self.realtime_label.config(text="○ Buffering", foreground='gray')
    
    def _on_close(self):
        """Handle window close."""
        self._stop_streaming()
        
        if self.rvc_pipeline:
            self.rvc_pipeline.stop()
        
        self.device_manager.terminate()
        self.root.destroy()
    
    def run(self):
        """Run the application."""
        self.root.mainloop()


def main():
    """Main entry point."""
    app = RVCApp()
    app.run()


if __name__ == "__main__":
    main()
