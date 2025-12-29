import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import numpy as np
from typing import Optional
from pathlib import Path
import os

from audio.ringbuffer import RingBuffer
from audio.pyaudio_io import AudioDeviceManager, AudioStream, calculate_level_db
from audio.recorder import AudioRecorder, AudioPlayer, load_wav_as_float

from dsp.world_vocoder import RealtimeWorldVocoder
from dsp.voice_profile import VoiceProfile, extract_profile, save_profile, load_profile


from utils.config import AudioConfig, get_profiles_directory
from utils.logging_utils import audio_logger, dsp_logger, ui_log_buffer


def calculate_transform_params(profile_a: VoiceProfile, profile_b: VoiceProfile) -> dict:
    """Calculate pitch and formant shift from two voice profiles."""
    if profile_a.f0_median_hz > 0 and profile_b.f0_median_hz > 0:
        f0_ratio = profile_b.f0_median_hz / profile_a.f0_median_hz
        pitch_shift = 12 * np.log2(f0_ratio)
    else:
        pitch_shift = 0.0
    
    formants_a = []
    formants_b = []
    
    if profile_a.formant_f1_median > 0 and profile_b.formant_f1_median > 0:
        formants_a.append(profile_a.formant_f1_median)
        formants_b.append(profile_b.formant_f1_median)
    
    if profile_a.formant_f2_median > 0 and profile_b.formant_f2_median > 0:
        formants_a.append(profile_a.formant_f2_median)
        formants_b.append(profile_b.formant_f2_median)
    
    if formants_a and formants_b:
        avg_formant_a = np.mean(formants_a)
        avg_formant_b = np.mean(formants_b)
        formant_shift = avg_formant_b / avg_formant_a
    else:
        formant_shift = 1.0
    
    pitch_shift = np.clip(pitch_shift, -12, 12)
    formant_shift = np.clip(formant_shift, 0.5, 2.0)
    
    return {
        'pitch_shift': float(pitch_shift),
        'formant_shift': float(formant_shift)
    }


class WorldWorker(threading.Thread):
    """Background worker for WORLD voice transformation."""
    
    def __init__(
        self,
        input_buffer: RingBuffer,
        output_buffer: RingBuffer,
        sample_rate: int = 14400,
        block_size: int = 1024
    ):
        super().__init__(daemon=True)
        
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.block_size = block_size
        
        self.vocoder = RealtimeWorldVocoder(
            sample_rate=sample_rate,
            block_size=block_size,
            overlap=0
        )
        
        self._running = False
        self._paused = True
        self._bypass = False
        
        self._lock = threading.Lock()
        self._input_level_db = -60.0
        self._output_level_db = -60.0
        self._process_time_ms = 0.0
    
    def run(self):
        self._running = True
        
        while self._running:
            if self._paused:
                time.sleep(0.01)
                continue
            
            available = self.input_buffer.count
            
            if available >= self.block_size:
                samples = self.input_buffer.pop(self.block_size)
                
                if samples is not None:
                    with self._lock:
                        self._input_level_db = calculate_level_db(samples)
                    
                    if self._bypass:
                        output = samples
                    else:
                        output = self.vocoder.process(samples)
                        with self._lock:
                            self._process_time_ms = self.vocoder.process_time_ms
                    
                    with self._lock:
                        self._output_level_db = calculate_level_db(output)
                    
                    self.output_buffer.push(output)
            else:
                time.sleep(0.005)
    
    def start_processing(self):
        self._paused = False
    
    def stop_processing(self):
        self._paused = True
        self.vocoder.reset()
    
    def set_bypass(self, bypass: bool):
        self._bypass = bypass
        self.vocoder.set_enabled(not bypass)
    
    def set_pitch_shift(self, semitones: float):
        self.vocoder.set_pitch_shift(semitones)
    
    def set_formant_shift(self, ratio: float):
        self.vocoder.set_formant_shift(ratio)
    
    def stop(self):
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


class CalibrationTab(ttk.Frame):
    """Tab for recording and managing voice profiles."""
    
    def __init__(self, parent, config: AudioConfig, on_profiles_changed=None):
        super().__init__(parent)
        
        self.config = config
        self.on_profiles_changed = on_profiles_changed
        
        self.profile_a: Optional[VoiceProfile] = None
        self.profile_b: Optional[VoiceProfile] = None
        self.audio_a: Optional[np.ndarray] = None
        self.audio_b: Optional[np.ndarray] = None
        
        self.recorder = AudioRecorder(
            sample_rate=config.sample_rate,
            channels=1,
            chunk_size=1024
        )
        self.player = AudioPlayer()
        
        self._recording = False
        self._current_profile = None
        self._init_complete = False
        
        self._create_widgets()
    
    def set_input_device(self, device_index: int):
        self.recorder.device_index = device_index
    
    def set_output_device(self, device_index: int):
        self.player.device_index = device_index
    
    def _create_widgets(self):
        main = ttk.Frame(self, padding=15)
        main.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(
            main,
            text="Voice Calibration",
            font=('Helvetica', 16, 'bold')
        ).pack(pady=(0, 5))
        
        ttk.Label(
            main,
            text="Record your voice (A) and the target voice (B) to calculate transformation parameters.",
            foreground='gray'
        ).pack(pady=(0, 15))
        
        profiles_frame = ttk.Frame(main)
        profiles_frame.pack(fill=tk.BOTH, expand=True)
        
        self.panel_a = self._create_profile_panel(profiles_frame, "Profile A (Your Voice)", 'A')
        self.panel_a.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.panel_b = self._create_profile_panel(profiles_frame, "Profile B (Target Voice)", 'B')
        self.panel_b.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        params_frame = ttk.LabelFrame(main, text="Calculated Transform Parameters", padding=10)
        params_frame.pack(fill=tk.X, pady=15)
        
        self.params_label = ttk.Label(
            params_frame,
            text="Record both profiles to calculate parameters",
            foreground='gray'
        )
        self.params_label.pack()
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main, textvariable=self.status_var).pack()
    
    def _create_profile_panel(self, parent, title: str, profile_id: str) -> ttk.LabelFrame:
        panel = ttk.LabelFrame(parent, text=title, padding=10)
        
        record_btn = ttk.Button(
            panel,
            text="🎤 Record (5 sec)",
            command=lambda: self._start_recording(profile_id)
        )
        record_btn.pack(fill=tk.X, pady=5)
        setattr(self, f'record_btn_{profile_id.lower()}', record_btn)
        
        status = ttk.Label(panel, text="Not recorded", foreground='gray')
        status.pack(pady=5)
        setattr(self, f'status_{profile_id.lower()}', status)
        
        info = ttk.Label(panel, text="", font=('Courier', 9))
        info.pack(pady=5)
        setattr(self, f'info_{profile_id.lower()}', info)
        
        btn_frame = ttk.Frame(panel)
        btn_frame.pack(fill=tk.X, pady=5)
        
        play_btn = ttk.Button(
            btn_frame,
            text="▶ Play",
            command=lambda: self._play_audio(profile_id),
            state='disabled',
            width=8
        )
        play_btn.pack(side=tk.LEFT, padx=2)
        setattr(self, f'play_btn_{profile_id.lower()}', play_btn)
        
        load_btn = ttk.Button(
            btn_frame,
            text="📂 Load",
            command=lambda: self._load_profile(profile_id),
            width=8
        )
        load_btn.pack(side=tk.LEFT, padx=2)
        
        save_btn = ttk.Button(
            btn_frame,
            text="💾 Save",
            command=lambda: self._save_profile(profile_id),
            state='disabled',
            width=8
        )
        save_btn.pack(side=tk.LEFT, padx=2)
        setattr(self, f'save_btn_{profile_id.lower()}', save_btn)
        
        return panel
    
    def load_existing_profiles(self):
        """Try to load existing profiles on startup."""
        self._init_complete = True
        profiles_dir = get_profiles_directory()
        
        profile_a_path = os.path.join(profiles_dir, 'profile_a')
        if os.path.exists(profile_a_path + '.json'):
            try:
                self.profile_a = load_profile(profile_a_path)
                self._update_profile_display('A')
            except Exception:
                pass
        
        profile_b_path = os.path.join(profiles_dir, 'profile_b')
        if os.path.exists(profile_b_path + '.json'):
            try:
                self.profile_b = load_profile(profile_b_path)
                self._update_profile_display('B')
            except Exception:
                pass
        
        self._update_params_display()
    
    def _start_recording(self, profile_id: str):
        """Start recording for a profile."""
        if self._recording:
            return
        
        self._recording = True
        self._current_profile = profile_id
        
        btn = getattr(self, f'record_btn_{profile_id.lower()}')
        status = getattr(self, f'status_{profile_id.lower()}')
        
        btn.config(state='disabled')
        status.config(text="Recording...", foreground='red')
        self.status_var.set(f"Recording Profile {profile_id}...")
        audio_logger.info(f"Start recording for Profile {profile_id}")
        
        def record_thread():
            try:
                for i in range(3, 0, -1):
                    self.after(0, lambda x=i: status.config(text=f"Starting in {x}..."))
                    time.sleep(1)
                
                self.after(0, lambda: status.config(text="🔴 SPEAK NOW!", foreground='red'))
                
                profiles_dir = get_profiles_directory()
                os.makedirs(profiles_dir, exist_ok=True)
                wav_path = os.path.join(profiles_dir, f'temp_{profile_id.lower()}.wav')
                
                self.recorder.start_recording(wav_path, duration_seconds=5.0)
                
                while self.recorder.state.is_recording:
                    time.sleep(0.1)
                
                self.after(0, lambda: status.config(text="Processing...", foreground='blue'))
                
                audio, sr = load_wav_as_float(wav_path)
                
                if profile_id == 'A':
                    self.audio_a = audio
                    self.profile_a = extract_profile(audio, sr, name=f"Profile {profile_id}")
                else:
                    self.audio_b = audio
                    self.profile_b = extract_profile(audio, sr, name=f"Profile {profile_id}")
                
                dsp_logger.info(f"Extracted profile for {profile_id}")
                self.after(0, lambda: self._update_profile_display(profile_id))
                self.after(0, self._update_params_display)
                self.after(0, lambda: self.status_var.set("Recording complete!"))
                audio_logger.info(f"Recording and extraction complete for Profile {profile_id}")
            except Exception as e:
                audio_logger.error(f"Recording failed for Profile {profile_id}: {e}")
                self.after(0, lambda: messagebox.showerror("Error", f"Recording failed: {e}"))
                self.after(0, lambda: status.config(text="Error", foreground='red'))
            finally:
                self._recording = False
                self.after(0, lambda: btn.config(state='normal'))
        
        threading.Thread(target=record_thread, daemon=True).start()
    
    def _update_profile_display(self, profile_id: str):
        """Update the display for a profile."""
        profile = self.profile_a if profile_id == 'A' else self.profile_b
        status = getattr(self, f'status_{profile_id.lower()}')
        info = getattr(self, f'info_{profile_id.lower()}')
        play_btn = getattr(self, f'play_btn_{profile_id.lower()}')
        save_btn = getattr(self, f'save_btn_{profile_id.lower()}')
        
        if profile and profile.is_valid:
            status.config(text="✓ Recorded", foreground='green')
            info.config(text=f"F0: {profile.f0_median_hz:.0f} Hz\n"
                            f"F1: {profile.formant_f1_median:.0f} Hz\n"
                            f"F2: {profile.formant_f2_median:.0f} Hz")
            play_btn.config(state='normal')
            save_btn.config(state='normal')
        else:
            status.config(text="Not recorded", foreground='gray')
            info.config(text="")
            play_btn.config(state='disabled')
            save_btn.config(state='disabled')
    
    def _update_params_display(self):
        """Update the calculated parameters display."""
        if self.profile_a and self.profile_a.is_valid and self.profile_b and self.profile_b.is_valid:
            params = calculate_transform_params(self.profile_a, self.profile_b)
            
            self.params_label.config(
                text=f"Pitch Shift: {params['pitch_shift']:+.1f} semitones  |  "
                     f"Formant Shift: {params['formant_shift']:.2f}x",
                foreground='black',
                font=('Helvetica', 12, 'bold')
            )
            
            if self.on_profiles_changed and self._init_complete:
                self.on_profiles_changed(params)
        else:
            self.params_label.config(
                text="Record both profiles to calculate parameters",
                foreground='gray',
                font=('Helvetica', 10)
            )
    
    def _play_audio(self, profile_id: str):
        """Play recorded audio."""
        audio = self.audio_a if profile_id == 'A' else self.audio_b
        if audio is not None:
            self.player.play_array(audio, self.config.sample_rate)
    
    def _save_profile(self, profile_id: str):
        """Save profile to disk."""
        profile = self.profile_a if profile_id == 'A' else self.profile_b
        if not profile:
            return
        
        profiles_dir = get_profiles_directory()
        os.makedirs(profiles_dir, exist_ok=True)
        
        filepath = os.path.join(profiles_dir, f'profile_{profile_id.lower()}')
        save_profile(profile, filepath)
        self.status_var.set(f"Profile {profile_id} saved!")
    
    def _load_profile(self, profile_id: str):
        """Load profile from disk."""
        filepath = filedialog.askopenfilename(
            title=f"Load Profile {profile_id}",
            filetypes=[("Profile JSON", "*.json")],
            initialdir=get_profiles_directory()
        )
        
        if not filepath:
            return
        
        try:
            base_path = filepath.rsplit('.', 1)[0]
            profile = load_profile(base_path)
            
            if profile_id == 'A':
                self.profile_a = profile
            else:
                self.profile_b = profile
            
            self._update_profile_display(profile_id)
            self._update_params_display()
            self.status_var.set(f"Profile {profile_id} loaded!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load profile: {e}")
    
    def get_transform_params(self) -> Optional[dict]:
        """Get current transform parameters if both profiles are valid."""
        if self.profile_a and self.profile_a.is_valid and self.profile_b and self.profile_b.is_valid:
            return calculate_transform_params(self.profile_a, self.profile_b)
        return None


class LiveTab(ttk.Frame):
    """Tab for real-time voice transformation."""
    
    def __init__(self, parent, config: AudioConfig, device_manager: AudioDeviceManager):
        super().__init__(parent)
        
        self.config = config
        self.device_manager = device_manager
        
        self.input_buffer: Optional[RingBuffer] = None
        self.output_buffer: Optional[RingBuffer] = None
        self.audio_stream: Optional[AudioStream] = None
        self.worker: Optional[WorldWorker] = None
        
        self._is_streaming = False
        self._auto_params = None
        
        self._create_widgets()
    
    def _create_widgets(self):
        main = ttk.Frame(self, padding=15)
        main.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(
            main,
            text="Live Voice Transformation",
            font=('Helvetica', 16, 'bold')
        ).pack(pady=(0, 5))
        
        ttk.Label(
            main,
            text="Transform your voice in real-time using WORLD vocoder",
            foreground='gray'
        ).pack(pady=(0, 15))
        
        self.calib_frame = ttk.LabelFrame(main, text="Calibration Status", padding=10)
        self.calib_frame.pack(fill=tk.X, pady=5)
        
        self.calib_label = ttk.Label(
            self.calib_frame,
            text="⚠ No calibration - using manual controls",
            foreground='orange'
        )
        self.calib_label.pack()
        
        self.apply_calib_btn = ttk.Button(
            self.calib_frame,
            text="Apply Calibration",
            command=self._apply_calibration,
            state='disabled'
        )
        self.apply_calib_btn.pack(pady=5)
        
        controls_frame = ttk.LabelFrame(main, text="Transform Controls", padding=10)
        controls_frame.pack(fill=tk.X, pady=5)
        
        pitch_row = ttk.Frame(controls_frame)
        pitch_row.pack(fill=tk.X, pady=5)
        ttk.Label(pitch_row, text="Pitch Shift:", width=12).pack(side=tk.LEFT)
        
        self.pitch_var = tk.DoubleVar(value=0.0)
        self.pitch_scale = ttk.Scale(
            pitch_row, from_=-12, to=12, orient=tk.HORIZONTAL,
            variable=self.pitch_var, command=self._on_pitch_changed
        )
        self.pitch_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.pitch_label = ttk.Label(pitch_row, text="0.0 st", width=10)
        self.pitch_label.pack(side=tk.LEFT)
        
        formant_row = ttk.Frame(controls_frame)
        formant_row.pack(fill=tk.X, pady=5)
        ttk.Label(formant_row, text="Formant Shift:", width=12).pack(side=tk.LEFT)
        
        self.formant_var = tk.DoubleVar(value=1.0)
        self.formant_scale = ttk.Scale(
            formant_row, from_=0.5, to=2.0, orient=tk.HORIZONTAL,
            variable=self.formant_var, command=self._on_formant_changed
        )
        self.formant_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.formant_label = ttk.Label(formant_row, text="1.00x", width=10)
        self.formant_label.pack(side=tk.LEFT)
        
        self.bypass_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls_frame, text="Bypass (passthrough)",
            variable=self.bypass_var, command=self._on_bypass_changed
        ).pack(anchor=tk.W, pady=5)
        
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=15)
        
        self.start_btn = ttk.Button(btn_frame, text="▶ Start", command=self._start)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="■ Stop", command=self._stop, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(btn_frame, text="● Stopped", foreground='gray')
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        meters_frame = ttk.LabelFrame(main, text="Levels", padding=10)
        meters_frame.pack(fill=tk.X, pady=5)
        
        in_row = ttk.Frame(meters_frame)
        in_row.pack(fill=tk.X, pady=2)
        ttk.Label(in_row, text="Input:", width=8).pack(side=tk.LEFT)
        self.input_meter = ttk.Progressbar(in_row, length=350, mode='determinate')
        self.input_meter.pack(side=tk.LEFT, padx=5)
        self.input_db = ttk.Label(in_row, text="-60 dB", width=8)
        self.input_db.pack(side=tk.LEFT)
        
        out_row = ttk.Frame(meters_frame)
        out_row.pack(fill=tk.X, pady=2)
        ttk.Label(out_row, text="Output:", width=8).pack(side=tk.LEFT)
        self.output_meter = ttk.Progressbar(out_row, length=350, mode='determinate')
        self.output_meter.pack(side=tk.LEFT, padx=5)
        self.output_db = ttk.Label(out_row, text="-60 dB", width=8)
        self.output_db.pack(side=tk.LEFT)
        
        self.stats_label = ttk.Label(meters_frame, text="Process: -- ms", foreground='gray')
        self.stats_label.pack(anchor=tk.W, pady=5)
    
    def set_calibration_params(self, params: dict):
        """Set calibration parameters from CalibrationTab."""
        self._auto_params = params
        self.calib_label.config(
            text=f"✓ Calibration: pitch {params['pitch_shift']:+.1f} st, formant {params['formant_shift']:.2f}x",
            foreground='green'
        )
        self.apply_calib_btn.config(state='normal')
    
    def _apply_calibration(self):
        """Apply calibration parameters to controls."""
        if self._auto_params:
            self.pitch_var.set(self._auto_params['pitch_shift'])
            self.formant_var.set(self._auto_params['formant_shift'])
            self._on_pitch_changed(self._auto_params['pitch_shift'])
            self._on_formant_changed(self._auto_params['formant_shift'])
    
    def _on_pitch_changed(self, value):
        pitch = float(value)
        self.pitch_label.config(text=f"{pitch:+.1f} st")
        if self.worker:
            self.worker.set_pitch_shift(pitch)
    
    def _on_formant_changed(self, value):
        formant = float(value)
        self.formant_label.config(text=f"{formant:.2f}x")
        if self.worker:
            self.worker.set_formant_shift(formant)
    
    def _on_bypass_changed(self):
        if self.worker:
            self.worker.set_bypass(self.bypass_var.get())
    
    def _start(self):
        buffer_samples = self.config.sample_rate * 2
        self.input_buffer = RingBuffer(buffer_samples)
        self.output_buffer = RingBuffer(buffer_samples)
        
        self.audio_stream = AudioStream(
            self.device_manager,
            self.config,
            self.input_buffer,
            self.output_buffer
        )
        
        block_size = int(0.2 * self.config.sample_rate)
        self.worker = WorldWorker(
            self.input_buffer, self.output_buffer,
            sample_rate=self.config.sample_rate,
            block_size=block_size
        )
        
        self.worker.set_pitch_shift(self.pitch_var.get())
        self.worker.set_formant_shift(self.formant_var.get())
        self.worker.set_bypass(self.bypass_var.get())
        
        self.worker.start()
        self.worker.start_processing()
        self.audio_stream.start(passthrough=False)
        
        self._is_streaming = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_label.config(text="● Running", foreground='green')
    
    def _stop(self):
        if self.worker:
            self.worker.stop_processing()
            self.worker.stop()
        if self.audio_stream:
            self.audio_stream.stop()
        
        self._is_streaming = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="● Stopped", foreground='gray')
    
    def update_meters(self):
        """Update level meters."""
        if self._is_streaming and self.worker:
            in_db = self.worker.input_level_db
            out_db = self.worker.output_level_db
            
            self.input_meter['value'] = max(0, min(100, (in_db + 60) / 60 * 100))
            self.output_meter['value'] = max(0, min(100, (out_db + 60) / 60 * 100))
            
            self.input_db.config(text=f"{in_db:.0f} dB")
            self.output_db.config(text=f"{out_db:.0f} dB")
            
            proc = self.worker.process_time_ms
            self.stats_label.config(text=f"Process: {proc:.0f} ms")


class VoiceTransformerApp:
    """Main application with calibration and live transformation tabs."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Voice Transformer - Calibrated WORLD Vocoder")
        self.root.geometry("700x600")
        self.root.minsize(650, 550)
        
        self.device_manager = AudioDeviceManager()
        self.config = AudioConfig(sample_rate=16000, buffer_size=512)
        
        self._create_ui()
        self._schedule_update()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _create_ui(self):
        device_frame = ttk.LabelFrame(self.root, text="Audio Devices", padding=5)
        device_frame.pack(fill=tk.X, padx=10, pady=5)
        
        in_row = ttk.Frame(device_frame)
        in_row.pack(fill=tk.X, pady=2)
        ttk.Label(in_row, text="Input:", width=8).pack(side=tk.LEFT)
        
        self.input_var = tk.StringVar()
        input_devices = self.device_manager.get_input_devices()
        input_names = [d.name for d in input_devices]
        self.input_combo = ttk.Combobox(
            in_row, textvariable=self.input_var,
            values=input_names, state='readonly', width=60
        )
        self.input_combo.pack(side=tk.LEFT, padx=5)
        if input_names:
            self.input_combo.current(0)
        self.input_combo.bind('<<ComboboxSelected>>', self._on_input_changed)
        
        out_row = ttk.Frame(device_frame)
        out_row.pack(fill=tk.X, pady=2)
        ttk.Label(out_row, text="Output:", width=8).pack(side=tk.LEFT)
        
        self.output_var = tk.StringVar()
        output_devices = self.device_manager.get_output_devices()
        output_names = [d.name for d in output_devices]
        self.output_combo = ttk.Combobox(
            out_row, textvariable=self.output_var,
            values=output_names, state='readonly', width=60
        )
        self.output_combo.pack(side=tk.LEFT, padx=5)
        if output_names:
            self.output_combo.current(0)
        self.output_combo.bind('<<ComboboxSelected>>', self._on_output_changed)
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.calib_tab = CalibrationTab(
            self.notebook, self.config,
            on_profiles_changed=self._on_profiles_changed
        )
        self.notebook.add(self.calib_tab, text="📊 Calibration")
        
        self.live_tab = LiveTab(self.notebook, self.config, self.device_manager)
        self.notebook.add(self.live_tab, text="🎤 Live")
        
        self._on_input_changed(None)
        self._on_output_changed(None)
        
        self.status = ttk.Label(
            self.root,
            text="Ready - Record voice profiles in Calibration tab",
            relief=tk.SUNKEN
        )
        self.status.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.calib_tab.load_existing_profiles()
    
    def _on_input_changed(self, event):
        input_devices = self.device_manager.get_input_devices()
        idx = self.input_combo.current()
        if 0 <= idx < len(input_devices):
            device_idx = input_devices[idx].index
            self.config.input_device_index = device_idx
            self.calib_tab.set_input_device(device_idx)
    
    def _on_output_changed(self, event):
        output_devices = self.device_manager.get_output_devices()
        idx = self.output_combo.current()
        if 0 <= idx < len(output_devices):
            device_idx = output_devices[idx].index
            self.config.output_device_index = device_idx
            self.calib_tab.set_output_device(device_idx)
    
    def _on_profiles_changed(self, params: dict):
        """Called when calibration profiles are updated."""
        self.live_tab.set_calibration_params(params)
        self.status.config(
            text=f"Calibration ready! Pitch: {params['pitch_shift']:+.1f} st, "
                 f"Formant: {params['formant_shift']:.2f}x"
        )
    
    def _schedule_update(self):
        self.live_tab.update_meters()
        self.root.after(50, self._schedule_update)
    
    def _on_close(self):
        self.live_tab._stop()
        self.device_manager.terminate()
        self.root.destroy()
    
    def run(self):
        self.root.mainloop()


def main():
    print("Voice Transformer with Calibration")
    print("=" * 40)
    print("1. Go to Calibration tab")
    print("2. Record your voice (Profile A)")
    print("3. Record target voice (Profile B)")
    print("4. Go to Live tab and click 'Apply Calibration'")
    print("5. Click Start to transform your voice!")
    print()
    
    app = VoiceTransformerApp()
    app.run()


if __name__ == "__main__":
    main()
