"""
Calibration UI tab - Simplified version.
Uses WAV file recording for reliability and auto-extracts profiles.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Callable, Optional
import numpy as np
import threading
import os

from utils.config import AudioConfig, CALIBRATION_DURATION_S, get_profiles_directory
from dsp.voice_profile import VoiceProfile, extract_profile, save_profile, load_profile
from audio.recorder import AudioRecorder, AudioPlayer, load_wav_as_float


class RecordingOverlay(tk.Toplevel):
    """
    Simple recording overlay window.
    Shows countdown and recording timer with level meter.
    """
    
    def __init__(self, parent, profile_name: str, duration: float = 5.0):
        super().__init__(parent)
        
        self.duration = duration
        self.profile_name = profile_name
        
        # Window setup
        self.title("Recording")
        self.geometry("400x250")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()  # Modal
        
        # Center on parent
        self.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() // 2) - 200
        y = parent.winfo_rooty() + (parent.winfo_height() // 2) - 125
        self.geometry(f"+{x}+{y}")
        
        self._create_widgets()
        
    def _create_widgets(self):
        """Create overlay widgets."""
        main = ttk.Frame(self, padding=20)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(
            main, 
            text=f"Recording {self.profile_name}",
            font=('Helvetica', 16, 'bold')
        ).pack(pady=(0, 20))
        
        # Big timer/countdown display
        self.timer_label = ttk.Label(
            main,
            text="3",
            font=('Helvetica', 48, 'bold'),
            foreground='#FF4444'
        )
        self.timer_label.pack(pady=10)
        
        # Status text
        self.status_label = ttk.Label(
            main,
            text="Get ready to speak...",
            font=('Helvetica', 12)
        )
        self.status_label.pack(pady=5)
        
        # Level meter (simple progress bar)
        self.level_meter = ttk.Progressbar(
            main,
            length=300,
            mode='determinate',
            maximum=100
        )
        self.level_meter.pack(pady=15)
        
        # Cancel button
        self.cancel_btn = ttk.Button(
            main,
            text="Cancel",
            command=self.on_cancel
        )
        self.cancel_btn.pack(pady=10)
        
        # Callback handlers
        self.on_cancel_callback: Optional[Callable] = None
    
    def on_cancel(self):
        """Handle cancel button."""
        if self.on_cancel_callback:
            self.on_cancel_callback()
        self.destroy()
    
    def update_countdown(self, seconds: int):
        """Update countdown display."""
        self.timer_label.config(text=str(seconds), foreground='#FF4444')
        self.status_label.config(text="Get ready to speak...")
    
    def update_recording(self, elapsed: float, level: float):
        """Update recording display."""
        remaining = max(0, self.duration - elapsed)
        self.timer_label.config(
            text=f"{remaining:.1f}s",
            foreground='#44AA44'
        )
        self.status_label.config(text="Speak now!")
        
        # Update level meter (scale level 0-1 to 0-100)
        self.level_meter['value'] = min(100, level * 100)
    
    def update_processing(self):
        """Update to show processing state."""
        self.timer_label.config(text="⏳", foreground='#4444FF')
        self.status_label.config(text="Extracting profile features...")
        self.level_meter['value'] = 0
        self.cancel_btn.config(state='disabled')


class CalibrationTab(ttk.Frame):
    """
    Tab 2: Calibration - Simplified voice profile recording.
    
    Features:
    - Clean recording overlay with countdown
    - WAV file based recording (no ring buffer issues)
    - Automatic profile extraction after recording
    - Playback and save/load functionality
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        config: AudioConfig,
        device_manager=None
    ):
        super().__init__(parent)
        
        self.config = config
        self.device_manager = device_manager
        
        # Profile data
        self.profile_a: Optional[VoiceProfile] = None
        self.profile_b: Optional[VoiceProfile] = None
        self.audio_a: Optional[np.ndarray] = None
        self.audio_b: Optional[np.ndarray] = None
        self.wav_path_a: Optional[str] = None
        self.wav_path_b: Optional[str] = None
        
        # Recorder and player
        self.recorder = AudioRecorder(
            sample_rate=config.sample_rate,
            channels=1,
            chunk_size=1024
        )
        self.player = AudioPlayer()
        
        # Recording state
        self._overlay: Optional[RecordingOverlay] = None
        self._current_profile: Optional[str] = None
        self._countdown_value = 0
        
        self._create_widgets()
    
    def set_input_device(self, device_index: int):
        """Set input device for recording."""
        self.recorder.device_index = device_index
    
    def set_output_device(self, device_index: int):
        """Set output device for playback."""
        self.player.device_index = device_index
    
    def _create_widgets(self):
        """Create all widgets."""
        main = ttk.Frame(self, padding=15)
        main.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(
            main,
            text="Voice Calibration",
            font=('Helvetica', 16, 'bold')
        ).pack(pady=(0, 10))
        
        # Instructions
        ttk.Label(
            main,
            text="Record 5 seconds of natural speech for each voice profile.\n"
                 "Speak clearly and avoid background noise.",
            justify=tk.CENTER,
            foreground='gray'
        ).pack(pady=(0, 15))
        
        # Profile panels in two columns
        profiles_frame = ttk.Frame(main)
        profiles_frame.pack(fill=tk.BOTH, expand=True)
        
        # Profile A
        self.panel_a = self._create_profile_panel(
            profiles_frame, 
            "Profile A (Your Voice)", 
            'A'
        )
        self.panel_a.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Profile B
        self.panel_b = self._create_profile_panel(
            profiles_frame,
            "Profile B (Target Voice)",
            'B'
        )
        self.panel_b.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to calibrate")
        status = ttk.Label(main, textvariable=self.status_var)
        status.pack(pady=(15, 0))
    
    def _create_profile_panel(self, parent, title: str, profile_id: str) -> ttk.LabelFrame:
        """Create a panel for one profile."""
        panel = ttk.LabelFrame(parent, text=title, padding=10)
        
        # Record button - prominent
        record_btn = ttk.Button(
            panel,
            text="🎤 Record",
            command=lambda: self._start_recording(profile_id)
        )
        record_btn.pack(fill=tk.X, pady=5)
        setattr(self, f'record_btn_{profile_id.lower()}', record_btn)
        
        # Status indicator
        status = ttk.Label(panel, text="Not recorded", foreground='gray')
        status.pack(pady=5)
        setattr(self, f'status_{profile_id.lower()}', status)
        
        # Action buttons (horizontal)
        btn_frame = ttk.Frame(panel)
        btn_frame.pack(fill=tk.X, pady=5)
        
        play_btn = ttk.Button(
            btn_frame,
            text="▶ Play",
            command=lambda: self._play_audio(profile_id),
            state='disabled'
        )
        play_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        setattr(self, f'play_btn_{profile_id.lower()}', play_btn)
        
        save_btn = ttk.Button(
            btn_frame,
            text="💾 Save",
            command=lambda: self._save_profile(profile_id),
            state='disabled'
        )
        save_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        setattr(self, f'save_btn_{profile_id.lower()}', save_btn)
        
        load_btn = ttk.Button(
            btn_frame,
            text="📂 Load",
            command=lambda: self._load_profile(profile_id)
        )
        load_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        # Feature summary
        summary_frame = ttk.LabelFrame(panel, text="Features", padding=5)
        summary_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        summary = tk.Text(
            summary_frame, 
            height=10, 
            width=30, 
            state='disabled',
            font=('Consolas', 9)
        )
        summary.pack(fill=tk.BOTH, expand=True)
        setattr(self, f'summary_{profile_id.lower()}', summary)
        
        return panel
    
    def _start_recording(self, profile_id: str):
        """Start the recording process."""
        self._current_profile = profile_id
        
        # Create overlay
        self._overlay = RecordingOverlay(
            self.winfo_toplevel(),
            f"Profile {profile_id}",
            duration=CALIBRATION_DURATION_S
        )
        self._overlay.on_cancel_callback = self._cancel_recording
        
        # Start countdown
        self._countdown_value = 3
        self._do_countdown()
    
    def _do_countdown(self):
        """Handle countdown before recording."""
        if not self._overlay:
            return
        
        if self._countdown_value > 0:
            self._overlay.update_countdown(self._countdown_value)
            self._countdown_value -= 1
            self.after(1000, self._do_countdown)
        else:
            self._begin_recording()
    
    def _begin_recording(self):
        """Start actual recording."""
        profile_id = self._current_profile
        if not profile_id or not self._overlay:
            return
        
        # Determine WAV path
        profiles_dir = get_profiles_directory()
        os.makedirs(profiles_dir, exist_ok=True)
        wav_path = os.path.join(profiles_dir, f"recording_{profile_id.lower()}.wav")
        
        # Set callbacks
        def on_level(level):
            if self._overlay:
                elapsed = self.recorder.state.elapsed_seconds
                self._overlay.update_recording(elapsed, level)
        
        def on_done(filepath):
            # Schedule UI update on main thread
            self.after(0, lambda: self._on_recording_done(filepath))
        
        self.recorder.on_level_update = on_level
        self.recorder.on_recording_done = on_done
        
        # Start recording
        self.recorder.start_recording(wav_path, CALIBRATION_DURATION_S)
        
        # Update overlay
        self._update_recording_display()
    
    def _update_recording_display(self):
        """Update overlay during recording."""
        if not self.recorder.state.is_recording:
            return
        
        if self._overlay:
            elapsed = self.recorder.state.elapsed_seconds
            level = self.recorder.state.max_level
            self._overlay.update_recording(elapsed, level)
        
        self.after(100, self._update_recording_display)
    
    def _on_recording_done(self, filepath: str):
        """Handle recording completion."""
        profile_id = self._current_profile
        if not profile_id:
            return
        
        # Store WAV path
        if profile_id == 'A':
            self.wav_path_a = filepath
        else:
            self.wav_path_b = filepath
        
        # Update overlay for processing
        if self._overlay:
            self._overlay.update_processing()
        
        # Load audio and extract profile
        self._extract_profile_from_wav(profile_id, filepath)
    
    def _extract_profile_from_wav(self, profile_id: str, filepath: str):
        """Extract profile from WAV file in background thread."""
        def do_extract():
            try:
                # Load audio
                audio, sr = load_wav_as_float(filepath)
                
                # Extract profile
                profile = extract_profile(
                    audio,
                    self.config.sample_rate,
                    name=f"Profile {profile_id}"
                )
                
                # Update UI on main thread
                self.after(0, lambda: self._on_extraction_done(profile_id, audio, profile))
                
            except Exception as e:
                self.after(0, lambda: self._on_extraction_error(str(e)))
        
        threading.Thread(target=do_extract, daemon=True).start()
    
    def _on_extraction_done(self, profile_id: str, audio: np.ndarray, profile: VoiceProfile):
        """Handle profile extraction completion."""
        # Store data
        if profile_id == 'A':
            self.audio_a = audio
            self.profile_a = profile
        else:
            self.audio_b = audio
            self.profile_b = profile
        
        # Close overlay
        if self._overlay:
            self._overlay.destroy()
            self._overlay = None
        
        # Update UI
        status = getattr(self, f'status_{profile_id.lower()}')
        status.config(text=f"✓ Recorded ({len(audio)/self.config.sample_rate:.1f}s)", foreground='green')
        
        play_btn = getattr(self, f'play_btn_{profile_id.lower()}')
        save_btn = getattr(self, f'save_btn_{profile_id.lower()}')
        play_btn.config(state='normal')
        save_btn.config(state='normal')
        
        # Update summary
        summary = getattr(self, f'summary_{profile_id.lower()}')
        summary.config(state='normal')
        summary.delete('1.0', tk.END)
        summary.insert('1.0', profile.get_summary())
        summary.config(state='disabled')
        
        self.status_var.set(f"Profile {profile_id} calibrated successfully")
        self._current_profile = None
    
    def _on_extraction_error(self, error: str):
        """Handle extraction error."""
        if self._overlay:
            self._overlay.destroy()
            self._overlay = None
        
        messagebox.showerror("Extraction Error", f"Failed to extract profile: {error}")
        self.status_var.set("Extraction failed")
        self._current_profile = None
    
    def _cancel_recording(self):
        """Cancel recording."""
        self.recorder.stop_recording()
        self._overlay = None
        self._current_profile = None
        self.status_var.set("Recording cancelled")
    
    def _play_audio(self, profile_id: str):
        """Play recorded audio."""
        audio = self.audio_a if profile_id == 'A' else self.audio_b
        
        if audio is not None:
            self.player.play_array(audio, self.config.sample_rate)
            self.status_var.set(f"Playing Profile {profile_id}...")
    
    def _save_profile(self, profile_id: str):
        """Save profile to file."""
        profile = self.profile_a if profile_id == 'A' else self.profile_b
        
        if profile is None:
            messagebox.showwarning("No Profile", "No profile to save.")
            return
        
        filepath = filedialog.asksaveasfilename(
            initialdir=get_profiles_directory(),
            title=f"Save Profile {profile_id}",
            defaultextension="",
            initialfile=f"profile_{profile_id.lower()}"
        )
        
        if filepath:
            try:
                save_profile(profile, filepath)
                self.status_var.set(f"Profile saved: {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Save Error", str(e))
    
    def _load_profile(self, profile_id: str):
        """Load profile from file."""
        filepath = filedialog.askopenfilename(
            initialdir=get_profiles_directory(),
            title=f"Load Profile {profile_id}",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            # Handle .json extension
            base_path = filepath
            if filepath.endswith('.json'):
                base_path = filepath[:-5]
            
            profile = load_profile(base_path)
            
            if profile_id == 'A':
                self.profile_a = profile
            else:
                self.profile_b = profile
            
            # Update UI
            status = getattr(self, f'status_{profile_id.lower()}')
            status.config(text=f"✓ Loaded: {profile.name}", foreground='blue')
            
            save_btn = getattr(self, f'save_btn_{profile_id.lower()}')
            save_btn.config(state='normal')
            
            summary = getattr(self, f'summary_{profile_id.lower()}')
            summary.config(state='normal')
            summary.delete('1.0', tk.END)
            summary.insert('1.0', profile.get_summary())
            summary.config(state='disabled')
            
            self.status_var.set(f"Loaded profile: {profile.name}")
            
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
    
    def get_profiles(self):
        """Get current profiles (A, B)."""
        return self.profile_a, self.profile_b
    
    def has_both_profiles(self) -> bool:
        """Check if both profiles are available."""
        return self.profile_a is not None and self.profile_b is not None
