"""
Device selection UI tab.
Input/output device selection, sample rate, and buffer size configuration.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, List

from audio.pyaudio_io import AudioDeviceManager, AudioDevice
from utils.config import AudioConfig, BUFFER_SIZES, DEFAULT_BUFFER_SIZE, SAMPLE_RATE_LOW, SAMPLE_RATE_HIGH


class DevicesTab(ttk.Frame):
    """
    Tab 1: Setup - Device configuration tab.
    
    Provides:
    - Input device dropdown
    - Output device dropdown  
    - Sample rate selector
    - Buffer size slider
    - Mic level meter with clip indicator
    - Test loopback button
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        device_manager: AudioDeviceManager,
        config: AudioConfig,
        on_config_changed: Optional[Callable[[AudioConfig], None]] = None,
        on_test_loopback: Optional[Callable[[bool], None]] = None
    ):
        """
        Initialize devices tab.
        
        Args:
            parent: Parent widget
            device_manager: Audio device manager
            config: Current audio configuration
            on_config_changed: Callback when config changes
            on_test_loopback: Callback for loopback test (bool = start/stop)
        """
        super().__init__(parent)
        
        self.device_manager = device_manager
        self.config = config
        self.on_config_changed = on_config_changed
        self.on_test_loopback = on_test_loopback
        
        self._is_locked = False
        self._loopback_active = False
        
        # Device lists
        self._input_devices: List[AudioDevice] = []
        self._output_devices: List[AudioDevice] = []
        
        self._create_widgets()
        self._refresh_devices()
    
    def _create_widgets(self):
        """Create all widgets for the tab."""
        # Main container with padding
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = ttk.Label(main_frame, text="Audio Device Setup", font=('Helvetica', 14, 'bold'))
        title.pack(pady=(0, 15))
        
        # Device selection frame
        device_frame = ttk.LabelFrame(main_frame, text="Device Selection", padding="10")
        device_frame.pack(fill=tk.X, pady=5)
        
        # Input device
        input_frame = ttk.Frame(device_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="Input Device:", width=15).pack(side=tk.LEFT)
        self.input_var = tk.StringVar()
        self.input_combo = ttk.Combobox(input_frame, textvariable=self.input_var, state='readonly', width=50)
        self.input_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.input_combo.bind('<<ComboboxSelected>>', self._on_input_changed)
        
        # Output device
        output_frame = ttk.Frame(device_frame)
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_frame, text="Output Device:", width=15).pack(side=tk.LEFT)
        self.output_var = tk.StringVar()
        self.output_combo = ttk.Combobox(output_frame, textvariable=self.output_var, state='readonly', width=50)
        self.output_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.output_combo.bind('<<ComboboxSelected>>', self._on_output_changed)
        
        # Refresh button
        refresh_btn = ttk.Button(device_frame, text="Refresh Devices", command=self._refresh_devices)
        refresh_btn.pack(pady=5)
        
        # Audio settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Audio Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=10)
        
        # Sample rate
        rate_frame = ttk.Frame(settings_frame)
        rate_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(rate_frame, text="Sample Rate:", width=15).pack(side=tk.LEFT)
        self.rate_var = tk.StringVar(value=str(self.config.sample_rate))
        self.rate_combo = ttk.Combobox(
            rate_frame, 
            textvariable=self.rate_var, 
            values=[str(SAMPLE_RATE_LOW), str(SAMPLE_RATE_HIGH)],
            state='readonly',
            width=15
        )
        self.rate_combo.pack(side=tk.LEFT, padx=5)
        self.rate_combo.bind('<<ComboboxSelected>>', self._on_rate_changed)
        ttk.Label(rate_frame, text="Hz").pack(side=tk.LEFT)
        
        # Buffer size
        buffer_frame = ttk.Frame(settings_frame)
        buffer_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(buffer_frame, text="Buffer Size:", width=15).pack(side=tk.LEFT)
        self.buffer_var = tk.IntVar(value=self.config.buffer_size)
        self.buffer_scale = ttk.Scale(
            buffer_frame,
            from_=0,
            to=len(BUFFER_SIZES) - 1,
            orient=tk.HORIZONTAL,
            command=self._on_buffer_changed
        )
        self.buffer_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.buffer_label = ttk.Label(buffer_frame, text=f"{self.config.buffer_size} samples", width=15)
        self.buffer_label.pack(side=tk.LEFT)
        
        # Set initial buffer scale position
        try:
            idx = BUFFER_SIZES.index(self.config.buffer_size)
            self.buffer_scale.set(idx)
        except ValueError:
            self.buffer_scale.set(1)  # Default to 256
        
        # Channel info
        channel_frame = ttk.Frame(settings_frame)
        channel_frame.pack(fill=tk.X, pady=5)
        ttk.Label(channel_frame, text="Channels:", width=15).pack(side=tk.LEFT)
        ttk.Label(channel_frame, text="Mono (1 channel)").pack(side=tk.LEFT)
        
        # Level meter frame
        meter_frame = ttk.LabelFrame(main_frame, text="Input Level", padding="10")
        meter_frame.pack(fill=tk.X, pady=10)
        
        # Level bar
        self.level_bar = ttk.Progressbar(meter_frame, length=400, mode='determinate')
        self.level_bar.pack(fill=tk.X, pady=5)
        
        # Level labels
        level_labels = ttk.Frame(meter_frame)
        level_labels.pack(fill=tk.X)
        ttk.Label(level_labels, text="-60 dB").pack(side=tk.LEFT)
        self.level_value_label = ttk.Label(level_labels, text="-60.0 dB")
        self.level_value_label.pack(side=tk.LEFT, expand=True)
        ttk.Label(level_labels, text="0 dB").pack(side=tk.RIGHT)
        
        # Clip indicator
        self.clip_indicator = ttk.Label(meter_frame, text="CLIP", foreground='gray')
        self.clip_indicator.pack(pady=5)
        
        # Test controls frame
        test_frame = ttk.LabelFrame(main_frame, text="Test", padding="10")
        test_frame.pack(fill=tk.X, pady=10)
        
        self.loopback_btn = ttk.Button(
            test_frame, 
            text="Test Loopback",
            command=self._toggle_loopback
        )
        self.loopback_btn.pack(pady=5)
        
        ttk.Label(
            test_frame, 
            text="Routes microphone directly to speakers for testing",
            foreground='gray'
        ).pack()
        
        # Status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=10)
        
        self.status_label = ttk.Label(status_frame, text="Ready", foreground='green')
        self.status_label.pack()
        
        self.lock_label = ttk.Label(status_frame, text="", foreground='orange')
        self.lock_label.pack()
    
    def _refresh_devices(self):
        """Refresh the device lists."""
        self.device_manager.refresh()
        
        self._input_devices = self.device_manager.get_input_devices()
        self._output_devices = self.device_manager.get_output_devices()
        
        # Update input combo
        input_names = [f"{d.index}: {d.name}" for d in self._input_devices]
        self.input_combo['values'] = input_names
        
        # Select default input
        default_input = self.device_manager.get_default_input_device()
        if default_input:
            for i, d in enumerate(self._input_devices):
                if d.index == default_input.index:
                    self.input_combo.current(i)
                    self.config.input_device_index = d.index
                    break
        
        # Update output combo
        output_names = [f"{d.index}: {d.name}" for d in self._output_devices]
        self.output_combo['values'] = output_names
        
        # Select default output
        default_output = self.device_manager.get_default_output_device()
        if default_output:
            for i, d in enumerate(self._output_devices):
                if d.index == default_output.index:
                    self.output_combo.current(i)
                    self.config.output_device_index = d.index
                    break
        
        self._update_status(f"Found {len(self._input_devices)} input, {len(self._output_devices)} output devices")
    
    def _on_input_changed(self, event):
        """Handle input device selection change."""
        if self._is_locked:
            return
        
        idx = self.input_combo.current()
        if 0 <= idx < len(self._input_devices):
            self.config.input_device_index = self._input_devices[idx].index
            self._notify_config_changed()
    
    def _on_output_changed(self, event):
        """Handle output device selection change."""
        if self._is_locked:
            return
        
        idx = self.output_combo.current()
        if 0 <= idx < len(self._output_devices):
            self.config.output_device_index = self._output_devices[idx].index
            self._notify_config_changed()
    
    def _on_rate_changed(self, event):
        """Handle sample rate change."""
        if self._is_locked:
            return
        
        try:
            self.config.sample_rate = int(self.rate_var.get())
            self._notify_config_changed()
        except ValueError:
            pass
    
    def _on_buffer_changed(self, value):
        """Handle buffer size change."""
        if self._is_locked:
            return
        
        idx = int(float(value))
        if 0 <= idx < len(BUFFER_SIZES):
            self.config.buffer_size = BUFFER_SIZES[idx]
            self.buffer_label.config(text=f"{self.config.buffer_size} samples")
            self._notify_config_changed()
    
    def _toggle_loopback(self):
        """Toggle loopback test."""
        self._loopback_active = not self._loopback_active
        
        if self._loopback_active:
            self.loopback_btn.config(text="Stop Loopback")
            self._update_status("Loopback active - speak into microphone", "blue")
        else:
            self.loopback_btn.config(text="Test Loopback")
            self._update_status("Loopback stopped", "green")
        
        if self.on_test_loopback:
            self.on_test_loopback(self._loopback_active)
    
    def _notify_config_changed(self):
        """Notify that config has changed."""
        if self.on_config_changed:
            self.on_config_changed(self.config)
    
    def _update_status(self, message: str, color: str = 'green'):
        """Update status label."""
        self.status_label.config(text=message, foreground=color)
    
    def lock_settings(self):
        """Lock settings (when audio is streaming)."""
        self._is_locked = True
        self.input_combo.config(state='disabled')
        self.output_combo.config(state='disabled')
        self.rate_combo.config(state='disabled')
        self.buffer_scale.config(state='disabled')
        self.lock_label.config(text="Settings locked while streaming")
    
    def unlock_settings(self):
        """Unlock settings."""
        self._is_locked = False
        self.input_combo.config(state='readonly')
        self.output_combo.config(state='readonly')
        self.rate_combo.config(state='readonly')
        self.buffer_scale.config(state='normal')
        self.lock_label.config(text="")
    
    def update_level_meter(self, level_db: float, is_clipping: bool = False):
        """
        Update the level meter display.
        
        Args:
            level_db: Input level in dB
            is_clipping: Whether clipping is detected
        """
        # Map -60..0 dB to 0..100
        level_pct = max(0, min(100, (level_db + 60) / 60 * 100))
        self.level_bar['value'] = level_pct
        
        self.level_value_label.config(text=f"{level_db:.1f} dB")
        
        if is_clipping:
            self.clip_indicator.config(foreground='red')
        else:
            self.clip_indicator.config(foreground='gray')
