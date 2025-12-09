"""
Live transform UI tab.
Real-time voice transformation controls and monitoring.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from utils.config import TransformConfig


class LiveTab(ttk.Frame):
    """
    Tab 3: Live Transform - Real-time voice transformation controls.
    
    Provides:
    - Start/Stop stream buttons
    - Profile selection dropdowns
    - Wet/Dry slider
    - Pitch map strength slider
    - Formant map strength slider
    - Envelope match strength slider
    - Unvoiced handling dropdown
    - Real-time meters (input/output level, F0, underrun/overrun)
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        on_start: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
        on_config_changed: Optional[Callable[[TransformConfig], None]] = None
    ):
        """
        Initialize live tab.
        
        Args:
            parent: Parent widget
            on_start: Callback when start is pressed
            on_stop: Callback when stop is pressed
            on_config_changed: Callback when transform config changes
        """
        super().__init__(parent)
        
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_config_changed = on_config_changed
        
        self.config = TransformConfig()
        self._is_streaming = False
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create all widgets for the tab."""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = ttk.Label(main_frame, text="Live Voice Transform", font=('Helvetica', 14, 'bold'))
        title.pack(pady=(0, 15))
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = ttk.Button(
            control_frame,
            text="▶ Start Stream",
            command=self._on_start
        )
        self.start_btn.pack(side=tk.LEFT, padx=10)
        
        self.stop_btn = ttk.Button(
            control_frame,
            text="■ Stop Stream",
            command=self._on_stop,
            state='disabled'
        )
        self.stop_btn.pack(side=tk.LEFT, padx=10)
        
        self.status_indicator = ttk.Label(
            control_frame,
            text="● Stopped",
            foreground='gray'
        )
        self.status_indicator.pack(side=tk.LEFT, padx=20)
        
        # Profile selection
        profile_frame = ttk.LabelFrame(main_frame, text="Profiles", padding="10")
        profile_frame.pack(fill=tk.X, pady=10)
        
        source_frame = ttk.Frame(profile_frame)
        source_frame.pack(fill=tk.X, pady=2)
        ttk.Label(source_frame, text="Source Profile:", width=15).pack(side=tk.LEFT)
        self.source_var = tk.StringVar(value="(Not loaded)")
        self.source_label = ttk.Label(source_frame, textvariable=self.source_var)
        self.source_label.pack(side=tk.LEFT)
        
        target_frame = ttk.Frame(profile_frame)
        target_frame.pack(fill=tk.X, pady=2)
        ttk.Label(target_frame, text="Target Profile:", width=15).pack(side=tk.LEFT)
        self.target_var = tk.StringVar(value="(Not loaded)")
        self.target_label = ttk.Label(target_frame, textvariable=self.target_var)
        self.target_label.pack(side=tk.LEFT)
        
        # Transform controls
        controls_frame = ttk.LabelFrame(main_frame, text="Transform Controls", padding="10")
        controls_frame.pack(fill=tk.X, pady=10)
        
        # Wet/Dry
        self._create_slider(
            controls_frame,
            "Wet/Dry Mix:",
            "wet_dry",
            0.0, 1.0, 1.0,
            "0% = Original, 100% = Transformed"
        )
        
        # Pitch strength
        self._create_slider(
            controls_frame,
            "Pitch Mapping:",
            "pitch_strength",
            0.0, 1.0, 0.0,
            "Strength of pitch transformation"
        )
        
        # Formant strength
        self._create_slider(
            controls_frame,
            "Formant Mapping:",
            "formant_strength",
            0.0, 1.0, 0.0,
            "Strength of formant warping"
        )
        
        # Envelope strength
        self._create_slider(
            controls_frame,
            "Envelope Match:",
            "envelope_strength",
            0.0, 1.0, 0.0,
            "Strength of spectral envelope matching"
        )
        
        # Unvoiced handling
        unvoiced_frame = ttk.Frame(controls_frame)
        unvoiced_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(unvoiced_frame, text="Unvoiced Handling:", width=20).pack(side=tk.LEFT)
        self.unvoiced_var = tk.StringVar(value='bypass')
        unvoiced_combo = ttk.Combobox(
            unvoiced_frame,
            textvariable=self.unvoiced_var,
            values=['bypass', 'noise_shaped'],
            state='readonly',
            width=15
        )
        unvoiced_combo.pack(side=tk.LEFT, padx=5)
        unvoiced_combo.bind('<<ComboboxSelected>>', self._on_unvoiced_changed)
        ttk.Label(unvoiced_frame, text="How to process unvoiced segments", foreground='gray').pack(side=tk.LEFT, padx=10)
        
        # Real-time meters
        meters_frame = ttk.LabelFrame(main_frame, text="Real-time Meters", padding="10")
        meters_frame.pack(fill=tk.X, pady=10)
        
        # Input level
        input_meter_frame = ttk.Frame(meters_frame)
        input_meter_frame.pack(fill=tk.X, pady=2)
        ttk.Label(input_meter_frame, text="Input Level:", width=15).pack(side=tk.LEFT)
        self.input_level_bar = ttk.Progressbar(input_meter_frame, length=300, mode='determinate')
        self.input_level_bar.pack(side=tk.LEFT, padx=5)
        self.input_level_label = ttk.Label(input_meter_frame, text="-60 dB", width=10)
        self.input_level_label.pack(side=tk.LEFT)
        
        # Output level
        output_meter_frame = ttk.Frame(meters_frame)
        output_meter_frame.pack(fill=tk.X, pady=2)
        ttk.Label(output_meter_frame, text="Output Level:", width=15).pack(side=tk.LEFT)
        self.output_level_bar = ttk.Progressbar(output_meter_frame, length=300, mode='determinate')
        self.output_level_bar.pack(side=tk.LEFT, padx=5)
        self.output_level_label = ttk.Label(output_meter_frame, text="-60 dB", width=10)
        self.output_level_label.pack(side=tk.LEFT)
        
        # F0 display
        f0_frame = ttk.Frame(meters_frame)
        f0_frame.pack(fill=tk.X, pady=2)
        ttk.Label(f0_frame, text="Estimated F0:", width=15).pack(side=tk.LEFT)
        self.f0_label = ttk.Label(f0_frame, text="-- Hz", font=('Helvetica', 12))
        self.f0_label.pack(side=tk.LEFT, padx=5)
        self.voiced_indicator = ttk.Label(f0_frame, text="", foreground='gray')
        self.voiced_indicator.pack(side=tk.LEFT, padx=10)
        
        # Error counters
        errors_frame = ttk.Frame(meters_frame)
        errors_frame.pack(fill=tk.X, pady=5)
        ttk.Label(errors_frame, text="Buffer Status:", width=15).pack(side=tk.LEFT)
        self.underrun_label = ttk.Label(errors_frame, text="Underruns: 0")
        self.underrun_label.pack(side=tk.LEFT, padx=10)
        self.overrun_label = ttk.Label(errors_frame, text="Overruns: 0")
        self.overrun_label.pack(side=tk.LEFT, padx=10)
        
        # Queue depth
        queue_frame = ttk.Frame(meters_frame)
        queue_frame.pack(fill=tk.X, pady=2)
        ttk.Label(queue_frame, text="Queue Depth:", width=15).pack(side=tk.LEFT)
        self.queue_bar = ttk.Progressbar(queue_frame, length=200, mode='determinate', maximum=100)
        self.queue_bar.pack(side=tk.LEFT, padx=5)
        self.queue_label = ttk.Label(queue_frame, text="0 frames")
        self.queue_label.pack(side=tk.LEFT)
    
    def _create_slider(
        self,
        parent: ttk.Frame,
        label: str,
        attr: str,
        min_val: float,
        max_val: float,
        default: float,
        tooltip: str
    ):
        """Create a labeled slider."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(frame, text=label, width=20).pack(side=tk.LEFT)
        
        var = tk.DoubleVar(value=default)
        setattr(self, f'{attr}_var', var)
        
        scale = ttk.Scale(
            frame,
            from_=min_val,
            to=max_val,
            orient=tk.HORIZONTAL,
            variable=var,
            command=lambda v, a=attr: self._on_slider_changed(a, float(v))
        )
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        value_label = ttk.Label(frame, text=f"{default*100:.0f}%", width=8)
        value_label.pack(side=tk.LEFT)
        setattr(self, f'{attr}_label', value_label)
        
        ttk.Label(frame, text=tooltip, foreground='gray').pack(side=tk.LEFT, padx=5)
    
    def _on_slider_changed(self, attr: str, value: float):
        """Handle slider change."""
        # Update label
        label = getattr(self, f'{attr}_label')
        label.config(text=f"{value*100:.0f}%")
        
        # Update config
        setattr(self.config, attr, value)
        
        self._notify_config_changed()
    
    def _on_unvoiced_changed(self, event):
        """Handle unvoiced mode change."""
        self.config.unvoiced_mode = self.unvoiced_var.get()
        self._notify_config_changed()
    
    def _on_start(self):
        """Handle start button press."""
        self._is_streaming = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_indicator.config(text="● Running", foreground='green')
        
        if self.on_start:
            self.on_start()
    
    def _on_stop(self):
        """Handle stop button press."""
        self._is_streaming = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_indicator.config(text="● Stopped", foreground='gray')
        
        if self.on_stop:
            self.on_stop()
    
    def _notify_config_changed(self):
        """Notify that config has changed."""
        if self.on_config_changed:
            self.on_config_changed(self.config)
    
    def set_profiles(self, source_name: str, target_name: str):
        """Update profile display."""
        self.source_var.set(source_name or "(Not loaded)")
        self.target_var.set(target_name or "(Not loaded)")
    
    def update_meters(
        self,
        input_level_db: float = -60.0,
        output_level_db: float = -60.0,
        f0: float = 0.0,
        is_voiced: bool = False,
        underruns: int = 0,
        overruns: int = 0,
        queue_depth: int = 0
    ):
        """Update real-time meters (optimized - only updates changed values)."""
        # Cache previous values to avoid unnecessary UI updates
        if not hasattr(self, '_prev_meters'):
            self._prev_meters = {}
        
        # Input level - only update if changed by more than 1dB
        input_pct = max(0, min(100, (input_level_db + 60) / 60 * 100))
        if abs(self._prev_meters.get('input_db', -999) - input_level_db) > 1:
            self.input_level_bar['value'] = input_pct
            self.input_level_label.config(text=f"{input_level_db:.1f} dB")
            self._prev_meters['input_db'] = input_level_db
        
        # Output level - only update if changed by more than 1dB
        output_pct = max(0, min(100, (output_level_db + 60) / 60 * 100))
        if abs(self._prev_meters.get('output_db', -999) - output_level_db) > 1:
            self.output_level_bar['value'] = output_pct
            self.output_level_label.config(text=f"{output_level_db:.1f} dB")
            self._prev_meters['output_db'] = output_level_db
        
        # F0 - only update if voiced state changed or f0 changed significantly
        prev_voiced = self._prev_meters.get('voiced', None)
        prev_f0 = self._prev_meters.get('f0', 0)
        
        if is_voiced != prev_voiced or (is_voiced and abs(f0 - prev_f0) > 5):
            if f0 > 0:
                self.f0_label.config(text=f"{f0:.1f} Hz")
                self.voiced_indicator.config(text="VOICED", foreground='green')
            else:
                self.f0_label.config(text="-- Hz")
                self.voiced_indicator.config(text="unvoiced", foreground='gray')
            self._prev_meters['voiced'] = is_voiced
            self._prev_meters['f0'] = f0
        
        # Error counters - only update if changed
        if self._prev_meters.get('underruns', -1) != underruns:
            self.underrun_label.config(
                text=f"Underruns: {underruns}",
                foreground='red' if underruns > 0 else 'black'
            )
            self._prev_meters['underruns'] = underruns
        
        if self._prev_meters.get('overruns', -1) != overruns:
            self.overrun_label.config(
                text=f"Overruns: {overruns}",
                foreground='red' if overruns > 0 else 'black'
            )
            self._prev_meters['overruns'] = overruns
        
        # Queue depth - only update if changed
        if self._prev_meters.get('queue', -1) != queue_depth:
            self.queue_bar['value'] = min(100, queue_depth * 5)
            self.queue_label.config(text=f"{queue_depth} frames")
            self._prev_meters['queue'] = queue_depth
    
    def get_config(self) -> TransformConfig:
        """Get current transform config."""
        return self.config
