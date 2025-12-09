"""
Diagnostics UI tab.
Performance monitoring, latency estimation, and debug controls.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, List
import time


class DiagnosticsTab(ttk.Frame):
    """
    Tab 4: Diagnostics - Performance monitoring and debug controls.
    
    Provides:
    - Latency estimate display
    - CPU time per frame (moving avg + max)
    - Spectrogram / log-magnitude plot (optional)
    - Debug toggles: bypass pitch / bypass envelope / bypass formant
    - Log viewer
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        on_bypass_changed: Optional[Callable[[dict], None]] = None
    ):
        """
        Initialize diagnostics tab.
        
        Args:
            parent: Parent widget
            on_bypass_changed: Callback when bypass toggles change
        """
        super().__init__(parent)
        
        self.on_bypass_changed = on_bypass_changed
        
        # Bypass state
        self.bypass_pitch = False
        self.bypass_envelope = False
        self.bypass_formant = False
        
        # Performance history
        self._dsp_times: List[float] = []
        self._latency_history: List[float] = []
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create all widgets for the tab."""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = ttk.Label(main_frame, text="Diagnostics & Performance", font=('Helvetica', 14, 'bold'))
        title.pack(pady=(0, 15))
        
        # Performance metrics
        perf_frame = ttk.LabelFrame(main_frame, text="Performance Metrics", padding="10")
        perf_frame.pack(fill=tk.X, pady=10)
        
        # Latency estimate
        latency_frame = ttk.Frame(perf_frame)
        latency_frame.pack(fill=tk.X, pady=5)
        ttk.Label(latency_frame, text="Estimated Latency:", width=20).pack(side=tk.LEFT)
        self.latency_label = ttk.Label(latency_frame, text="-- ms", font=('Helvetica', 12, 'bold'))
        self.latency_label.pack(side=tk.LEFT)
        
        # Latency breakdown
        breakdown_frame = ttk.Frame(perf_frame)
        breakdown_frame.pack(fill=tk.X, pady=2)
        ttk.Label(breakdown_frame, text="", width=20).pack(side=tk.LEFT)  # Spacer
        self.latency_breakdown = ttk.Label(
            breakdown_frame, 
            text="Input buffer: -- ms | Queue: -- ms | Output buffer: -- ms",
            foreground='gray'
        )
        self.latency_breakdown.pack(side=tk.LEFT)
        
        # DSP time
        dsp_frame = ttk.Frame(perf_frame)
        dsp_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dsp_frame, text="DSP Time per Frame:", width=20).pack(side=tk.LEFT)
        self.dsp_avg_label = ttk.Label(dsp_frame, text="Avg: -- ms")
        self.dsp_avg_label.pack(side=tk.LEFT, padx=10)
        self.dsp_max_label = ttk.Label(dsp_frame, text="Max: -- ms")
        self.dsp_max_label.pack(side=tk.LEFT, padx=10)
        
        # DSP time bar
        dsp_bar_frame = ttk.Frame(perf_frame)
        dsp_bar_frame.pack(fill=tk.X, pady=2)
        ttk.Label(dsp_bar_frame, text="", width=20).pack(side=tk.LEFT)
        self.dsp_time_bar = ttk.Progressbar(dsp_bar_frame, length=300, mode='determinate', maximum=20)
        self.dsp_time_bar.pack(side=tk.LEFT)
        ttk.Label(dsp_bar_frame, text="(target: < hop time)", foreground='gray').pack(side=tk.LEFT, padx=5)
        
        # CPU usage estimate
        cpu_frame = ttk.Frame(perf_frame)
        cpu_frame.pack(fill=tk.X, pady=5)
        ttk.Label(cpu_frame, text="DSP CPU Load:", width=20).pack(side=tk.LEFT)
        self.cpu_label = ttk.Label(cpu_frame, text="--%")
        self.cpu_label.pack(side=tk.LEFT)
        self.cpu_bar = ttk.Progressbar(cpu_frame, length=200, mode='determinate')
        self.cpu_bar.pack(side=tk.LEFT, padx=10)
        
        # Debug toggles
        debug_frame = ttk.LabelFrame(main_frame, text="Debug Bypass Controls", padding="10")
        debug_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(
            debug_frame,
            text="Toggle individual processing stages for debugging:",
            foreground='gray'
        ).pack(anchor=tk.W)
        
        toggles_frame = ttk.Frame(debug_frame)
        toggles_frame.pack(fill=tk.X, pady=5)
        
        self.bypass_pitch_var = tk.BooleanVar(value=False)
        pitch_cb = ttk.Checkbutton(
            toggles_frame,
            text="Bypass Pitch Mapping",
            variable=self.bypass_pitch_var,
            command=self._on_bypass_changed
        )
        pitch_cb.pack(side=tk.LEFT, padx=10)
        
        self.bypass_envelope_var = tk.BooleanVar(value=False)
        envelope_cb = ttk.Checkbutton(
            toggles_frame,
            text="Bypass Envelope Match",
            variable=self.bypass_envelope_var,
            command=self._on_bypass_changed
        )
        envelope_cb.pack(side=tk.LEFT, padx=10)
        
        self.bypass_formant_var = tk.BooleanVar(value=False)
        formant_cb = ttk.Checkbutton(
            toggles_frame,
            text="Bypass Formant Warp",
            variable=self.bypass_formant_var,
            command=self._on_bypass_changed
        )
        formant_cb.pack(side=tk.LEFT, padx=10)
        
        # Spectrogram placeholder
        spec_frame = ttk.LabelFrame(main_frame, text="Spectrogram (Live)", padding="10")
        spec_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.spectrogram_canvas = tk.Canvas(spec_frame, height=150, bg='black')
        self.spectrogram_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Frequency labels
        freq_labels = ttk.Frame(spec_frame)
        freq_labels.pack(fill=tk.X)
        ttk.Label(freq_labels, text="0 Hz").pack(side=tk.LEFT)
        ttk.Label(freq_labels, text="8 kHz").pack(side=tk.RIGHT)
        
        # Log viewer
        log_frame = ttk.LabelFrame(main_frame, text="Event Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Log text with scrollbar
        log_scroll = ttk.Scrollbar(log_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(log_frame, height=8, state='disabled', yscrollcommand=log_scroll.set)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.log_text.yview)
        
        # Clear log button
        clear_btn = ttk.Button(log_frame, text="Clear Log", command=self._clear_log)
        clear_btn.pack(pady=5)
    
    def _on_bypass_changed(self):
        """Handle bypass toggle change."""
        self.bypass_pitch = self.bypass_pitch_var.get()
        self.bypass_envelope = self.bypass_envelope_var.get()
        self.bypass_formant = self.bypass_formant_var.get()
        
        if self.on_bypass_changed:
            self.on_bypass_changed({
                'bypass_pitch': self.bypass_pitch,
                'bypass_envelope': self.bypass_envelope,
                'bypass_formant': self.bypass_formant
            })
    
    def _clear_log(self):
        """Clear the log display."""
        self.log_text.config(state='normal')
        self.log_text.delete('1.0', tk.END)
        self.log_text.config(state='disabled')
    
    def update_metrics(
        self,
        latency_ms: float = 0.0,
        input_buffer_ms: float = 0.0,
        queue_ms: float = 0.0,
        output_buffer_ms: float = 0.0,
        dsp_avg_ms: float = 0.0,
        dsp_max_ms: float = 0.0,
        hop_time_ms: float = 10.0
    ):
        """Update performance metrics display (optimized - caches values)."""
        # Cache previous values
        if not hasattr(self, '_prev_metrics'):
            self._prev_metrics = {}
        
        # Only update latency if changed by > 1ms
        if abs(self._prev_metrics.get('latency', -999) - latency_ms) > 1:
            self.latency_label.config(text=f"{latency_ms:.1f} ms")
            # Color code latency
            if latency_ms > 50:
                self.latency_label.config(foreground='red')
            elif latency_ms > 30:
                self.latency_label.config(foreground='orange')
            else:
                self.latency_label.config(foreground='green')
            self._prev_metrics['latency'] = latency_ms
        
        # Breakdown - update less frequently
        breakdown_key = f"{input_buffer_ms:.0f}_{queue_ms:.0f}_{output_buffer_ms:.0f}"
        if self._prev_metrics.get('breakdown') != breakdown_key:
            self.latency_breakdown.config(
                text=f"Input buffer: {input_buffer_ms:.1f} ms | Queue: {queue_ms:.1f} ms | Output buffer: {output_buffer_ms:.1f} ms"
            )
            self._prev_metrics['breakdown'] = breakdown_key
        
        # DSP time - only update if changed by > 0.1ms
        if abs(self._prev_metrics.get('dsp_avg', -999) - dsp_avg_ms) > 0.1:
            self.dsp_avg_label.config(text=f"Avg: {dsp_avg_ms:.2f} ms")
            self.dsp_time_bar['value'] = min(20, dsp_avg_ms)
            # Color code if too slow
            if dsp_avg_ms > hop_time_ms:
                self.dsp_avg_label.config(foreground='red')
            else:
                self.dsp_avg_label.config(foreground='green')
            self._prev_metrics['dsp_avg'] = dsp_avg_ms
        
        if abs(self._prev_metrics.get('dsp_max', -999) - dsp_max_ms) > 0.1:
            self.dsp_max_label.config(text=f"Max: {dsp_max_ms:.2f} ms")
            self._prev_metrics['dsp_max'] = dsp_max_ms
        
        # CPU load estimate
        cpu_pct = (dsp_avg_ms / hop_time_ms) * 100 if hop_time_ms > 0 else 0
        if abs(self._prev_metrics.get('cpu', -999) - cpu_pct) > 1:
            self.cpu_label.config(text=f"{cpu_pct:.1f}%")
            self.cpu_bar['value'] = min(100, cpu_pct)
            if cpu_pct > 80:
                self.cpu_label.config(foreground='red')
            elif cpu_pct > 50:
                self.cpu_label.config(foreground='orange')
            else:
                self.cpu_label.config(foreground='green')
            self._prev_metrics['cpu'] = cpu_pct
    
    def update_spectrogram(self, spectrum: Optional[list] = None):
        """
        Update spectrogram display with new spectrum data.
        
        Args:
            spectrum: List of magnitude values (log scale)
        """
        if spectrum is None:
            return
        
        canvas = self.spectrogram_canvas
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            return
        
        # Shift existing content left
        canvas.move('all', -2, 0)
        
        # Delete items that moved off screen
        canvas.delete(canvas.find_overlapping(-100, 0, 0, height))
        
        # Draw new column
        n_bins = len(spectrum)
        if n_bins == 0:
            return
        
        bin_height = height / n_bins
        
        for i, val in enumerate(spectrum):
            # Map value to color (assume val is in dB, -80 to 0)
            intensity = max(0, min(255, int((val + 80) / 80 * 255)))
            color = f'#{intensity:02x}{intensity//2:02x}00'  # Orange-ish
            
            y = height - (i + 1) * bin_height
            canvas.create_rectangle(
                width - 2, y,
                width, y + bin_height,
                fill=color, outline=''
            )
    
    def add_log(self, message: str, level: str = 'INFO'):
        """Add a message to the log display."""
        timestamp = time.strftime('%H:%M:%S')
        
        self.log_text.config(state='normal')
        
        # Add with color tag
        if level == 'ERROR':
            tag = 'error'
        elif level == 'WARNING':
            tag = 'warning'
        else:
            tag = 'info'
        
        self.log_text.insert(tk.END, f"[{timestamp}] {level}: {message}\n", tag)
        
        # Configure tags
        self.log_text.tag_config('error', foreground='red')
        self.log_text.tag_config('warning', foreground='orange')
        self.log_text.tag_config('info', foreground='black')
        
        # Auto-scroll to bottom
        self.log_text.see(tk.END)
        
        self.log_text.config(state='disabled')
    
    def add_logs_from_buffer(self, messages: list):
        """Add multiple log messages from buffer."""
        for msg in messages:
            self.add_log(msg.message, msg.level)
    
    def get_bypass_state(self) -> dict:
        """Get current bypass toggle states."""
        return {
            'bypass_pitch': self.bypass_pitch,
            'bypass_envelope': self.bypass_envelope,
            'bypass_formant': self.bypass_formant
        }
