# Voice Transform - Real-time A→B Voice Conversion

A real-time voice transformation system that converts one person's voice to sound like another's using pitch shifting based on voice profile analysis.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## Project Structure

```
Project/
├── app.py                 # Main application entry point
│                          # - VoiceTransformApp: Main Tkinter window with 4 tabs
│                          # - DSPWorker: Background thread for audio processing
│
├── requirements.txt       # Python dependencies (numpy, scipy, pyaudio)
├── profiles/             # Saved voice profiles (.npz + .json)
│
├── audio/                # Audio I/O layer
│   ├── pyaudio_io.py     # AudioDeviceManager: Device enumeration
│   │                     # AudioStream: PyAudio callback-based streaming
│   ├── ringbuffer.py     # RingBuffer: Thread-safe circular buffer
│   └── recorder.py       # AudioRecorder: WAV file recording
│
├── dsp/                  # Digital Signal Processing modules
│   ├── phase_vocoder.py  # PhaseVocoderPitchShifter: STFT-based pitch shifting
│   ├── transform.py      # VoiceTransformPipeline: Main DSP pipeline
│   ├── voice_profile.py  # VoiceProfile: Feature extraction and storage
│   ├── pitch.py          # YINPitchDetector: F0 estimation using YIN algorithm
│   ├── lpc.py            # LPC analysis for spectral envelope
│   ├── formant.py        # FormantTracker: Formant frequency estimation
│   └── stft.py           # STFT/ISTFT utilities
│
├── ui/                   # Tkinter UI tabs
│   ├── devices.py        # Tab 1: Device selection, buffer configuration
│   ├── calibration.py    # Tab 2: Voice recording & profile extraction
│   ├── live.py           # Tab 3: Real-time transform controls
│   └── diagnostics.py    # Tab 4: Performance monitoring
│
└── utils/                # Utilities
    ├── config.py         # AudioConfig, TransformConfig, constants
    ├── logging_utils.py  # Thread-safe logging utilities
    └── timing.py         # FrameTimer, LatencyEstimator
```

## Architecture

### Threading Model

The application uses three threads to achieve real-time audio processing:

```
┌──────────────────────────────────────────────────────────────────────┐
│                      MAIN THREAD (Tkinter UI)                        │
│  • Handles user interaction (buttons, sliders)                       │
│  • Polls shared state every 50ms via after() callbacks               │
│  • Updates level meters, status displays                             │
└──────────────────────────────────────────────────────────────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
┌────────────────────────────┐         ┌────────────────────────────────┐
│  AUDIO I/O THREAD          │         │     DSP WORKER THREAD          │
│  (PyAudio callback)        │         │                                │
│                            │         │  while running:                │
│  def callback():           │         │    samples = input_buf.pop()   │
│    # MUST be fast!         │         │    output = pipeline.process() │
│    input_buf.push(data)    │         │    output_buf.push(output)     │
│    return output_buf.pop() │         │                                │
└────────────────────────────┘         └────────────────────────────────┘
              │                                       ▲
              │     ┌─────────────────────────┐       │
              └────►│   INPUT RING BUFFER     │───────┘
                    └─────────────────────────┘
                    ┌─────────────────────────┐
              ┌────►│   OUTPUT RING BUFFER    │◄──────┐
              │     └─────────────────────────┘       │
              └───────────────────────────────────────┘
```

### Data Flow (Simplified)

```
Microphone
    │
    ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Input RingBuf   │────►│  DSP Pipeline   │────►│ Output RingBuf  │
│ (thread-safe)   │     │                 │     │ (thread-safe)   │
└─────────────────┘     │  Pitch Shift    │     └─────────────────┘
                        │  via Resampling │              │
                        └─────────────────┘              ▼
                                                    Speakers
```

## Core Classes

### Main Application (`app.py`)

#### `VoiceTransformApp`
The main Tkinter application window. Creates and manages:
- Audio device manager and stream
- Ring buffers for inter-thread communication
- DSP worker thread
- Four UI tabs (Devices, Calibration, Live, Diagnostics)

#### `DSPWorker(threading.Thread)`
Background thread that processes audio:
```python
def run(self):
    while running:
        samples = input_buffer.pop(hop_size)      # Get audio chunk
        output, metrics = pipeline.process_direct(samples)  # Process
        output_buffer.push(output)                # Send to output
```

---

### Audio Layer (`audio/`)

#### `RingBuffer` (`ringbuffer.py`)
Thread-safe circular buffer for passing audio between threads:
- `push(data)` - Add samples to buffer
- `pop(count)` - Remove and return samples
- `count` - Number of samples available
- Uses numpy arrays for efficiency

#### `AudioDeviceManager` (`pyaudio_io.py`)
Enumerates and manages audio devices:
- `get_input_devices()` - List microphones
- `get_output_devices()` - List speakers
- `get_default_input_device()` / `get_default_output_device()`

#### `AudioStream` (`pyaudio_io.py`)
PyAudio stream wrapper with callback mode:
- Runs in separate thread (PyAudio manages this)
- Callback pushes input to ring buffer, pops output from ring buffer
- Supports passthrough mode (direct loopback) or processing mode

#### `AudioRecorder` (`recorder.py`)
Records audio to WAV file in a separate thread:
- Used by calibration tab for profile recording
- Non-blocking recording with callback for completion

---

### DSP Layer (`dsp/`)

#### `PhaseVocoderPitchShifter` (`phase_vocoder.py`)
Smoother real-time pitch shifting via STFT + phase vocoder:

```python
def process(self, samples):
    # Phase vocoder analysis/resynthesis
    stretch = 1.0 / pitch_ratio
    for frame in stft_frames(samples):
        inst_phase = expected + principal(phase_diff)
        phase_acc += inst_phase * stretch
        frame_out = istft(mag * exp(j*phase_acc))
        stretched_buffer.append(frame_out_hop)
    
    # Resample stretched audio back to original hop duration
    return render_resampled_block(len(samples))
```

#### `VoiceTransformPipeline` (`transform.py`)
Coordinates the transformation:
1. Computes pitch ratio from source/target profiles
2. Applies pitch strength scaling
3. Calls pitch shifter
4. Applies wet/dry mix

```python
def process_direct(self, samples):
    ratio = target_f0 / source_f0  # From profiles
    ratio = 1.0 + (ratio - 1.0) * strength  # Apply strength
    
    self.pitch_shifter.set_pitch_ratio(ratio)
    shifted = self.pitch_shifter.process(samples)
    
    output = wet_dry * shifted + (1 - wet_dry) * samples
    return output
```

#### `VoiceProfile` (`voice_profile.py`)
Stores voice characteristics extracted during calibration:
- `f0_median_hz` - Median fundamental frequency
- `f0_p05_hz`, `f0_p95_hz` - Pitch range (5th/95th percentile)
- `formant_f1/f2/f3_median` - Formant frequencies
- `envelope_log_mag` - Spectral envelope
- `lpc_coefficients` - LPC analysis coefficients

#### `extract_profile()` (`voice_profile.py`)
Analyzes recorded audio to create a VoiceProfile:
1. Process audio frame by frame
2. Detect pitch using YIN algorithm (optimized with FFT)
3. Extract formants using LPC analysis
4. Compute statistics (median, percentiles)
5. Store averaged spectral envelope

#### `YINPitchDetector` (`pitch.py`)
Fundamental frequency (F0) estimation using the YIN algorithm:
- Uses FFT-based autocorrelation for efficiency
- Returns F0 in Hz and confidence score
- Distinguishes voiced vs unvoiced frames

#### `compute_lpc()` (`lpc.py`)
Linear Predictive Coding for spectral envelope:
- Uses Levinson-Durbin recursion
- FFT-based autocorrelation for speed
- Returns LPC coefficients and gain

---

### UI Layer (`ui/`)

#### `DevicesTab` (`devices.py`)
**Tab 1: Setup**
- Input/output device dropdowns
- Buffer size configuration
- Sample rate selection
- Microphone level meter
- Test loopback button

#### `CalibrationTab` (`calibration.py`)
**Tab 2: Calibration**
- Record Profile A (your voice) and Profile B (target voice)
- 5-second recordings with countdown overlay
- Automatic feature extraction after recording
- Displays extracted features (F0, formants, etc.)
- Save/load profiles to disk

#### `LiveTab` (`live.py`)
**Tab 3: Live Transform**
- Start/Stop stream buttons
- Enable Transform toggle
- Pitch strength slider (0-100%)
- Wet/dry mix slider
- Real-time level meters
- Underrun/overrun counters

#### `DiagnosticsTab` (`diagnostics.py`)
**Tab 4: Diagnostics**
- Latency estimation
- CPU usage per frame
- Debug log viewer
- Performance metrics

---

### Utilities (`utils/`)

#### `AudioConfig` (`config.py`)
Audio system configuration:
```python
@dataclass
class AudioConfig:
    sample_rate: int = 16000
    buffer_size: int = 256
    frame_size: int = 320   # 20ms at 16kHz
    hop_size: int = 160     # 10ms at 16kHz
```

#### `TransformConfig` (`config.py`)
Transform parameters (adjustable via UI):
```python
@dataclass
class TransformConfig:
    wet_dry: float = 1.0           # 0=dry, 1=wet
    pitch_strength: float = 1.0    # 0=bypass, 1=full transform
    formant_strength: float = 0.0  # Not currently used
    envelope_strength: float = 0.0 # Not currently used
```

## Configuration Constants

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_SAMPLE_RATE` | 16000 Hz | Audio sample rate |
| `DEFAULT_BUFFER_SIZE` | 256 samples | PyAudio callback buffer |
| `FRAME_SIZE_MS` | 20 ms | Analysis frame size |
| `HOP_SIZE_MS` | 10 ms | Frame hop (50% overlap) |
| `LPC_ORDER` | 16 | LPC analysis order |
| `YIN_THRESHOLD` | 0.15 | Voiced/unvoiced threshold |
| `F0_MIN` / `F0_MAX` | 50-500 Hz | Pitch detection range |

## UI Tabs

### Tab 1: Setup (Devices)
- Select input device (microphone)
- Select output device (speakers/headphones)
- Adjust buffer size (lower = less latency, higher = more stable)
- Test with loopback (hear your mic through speakers)
- Monitor input level with clip indicator

### Tab 2: Calibration
- **Record Profile A**: Record 5 seconds of your natural voice
- **Record Profile B**: Record 5 seconds of the target voice
- Automatic feature extraction (pitch, formants, spectral envelope)
- View extracted features summary
- Save/load profiles for later use

### Tab 3: Live Transform
- **Start/Stop Stream**: Enable real-time audio processing
- **Enable Transform**: Toggle voice transformation on/off
- **Pitch Strength**: How much to shift pitch (0% = no change, 100% = full A→B)
- **Wet/Dry Mix**: Blend between original and transformed voice
- Real-time monitoring:
  - Input/output level meters
  - Current pitch display
  - Buffer underrun/overrun counters

### Tab 4: Diagnostics
- Performance metrics (latency, CPU usage)
- Debug logging
- System status

## How the Pitch Shifting Works

The pitch shifter uses **direct resampling** for zero-latency operation:

### Algorithm
1. **Input**: Block of audio samples (e.g., 160 samples = 10ms at 16kHz)
2. **Calculate read indices**: `out_indices = [0, ratio, 2*ratio, 3*ratio, ...]`
3. **Linear interpolation**: Read samples at fractional positions
4. **Crossfade**: Blend with previous block's tail (32 samples) to avoid clicks
5. **Output**: Same number of samples as input (no buffering delay)

### Example
- Source voice: 120 Hz (male)
- Target voice: 200 Hz (female)
- Pitch ratio: 200/120 = 1.67
- Effect: Voice pitch raised by ~9 semitones

```
ratio > 1.0  →  Higher pitch (read faster, compress waveform)
ratio < 1.0  →  Lower pitch (read slower, stretch waveform)
ratio = 1.0  →  No change (bypass)
```

## Voice Profile Features

Each VoiceProfile contains:

| Feature | Description |
|---------|-------------|
| `f0_median_hz` | Median fundamental frequency (pitch) |
| `f0_p05_hz` / `f0_p95_hz` | 5th/95th percentile F0 (pitch range) |
| `f0_std_log` | F0 standard deviation (log domain) |
| `voiced_ratio` | Proportion of voiced frames |
| `envelope_log_mag` | Average spectral envelope (log magnitude) |
| `lpc_coefficients` | Average LPC coefficients |
| `formant_f1/f2/f3_median` | Median formant frequencies |

## Usage Workflow

1. **Setup**: Select your microphone and speakers, test with loopback
2. **Record Profile A**: Record yourself speaking naturally for 5 seconds
3. **Record Profile B**: Record the target voice (or use a saved profile)
4. **Go Live**: Switch to Live tab, click "Start Stream", enable transform
5. **Adjust**: Use the pitch strength slider to control how much transformation is applied

## Requirements

- Python 3.8+
- NumPy
- SciPy
- PyAudio (requires PortAudio system library)
- tkinter (usually bundled with Python)

### Installing PyAudio

**Windows:**
```bash
pip install pyaudio
```

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Linux:**
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| No audio output | Wrong device selected | Check Tab 1, select correct output |
| High latency | Buffer too large | Reduce buffer size in Tab 1 |
| Audio crackling | Buffer too small | Increase buffer size |
| No pitch change | Profiles not set | Record/load both Profile A and B |
| "Chipmunk" sound | Pitch too extreme | Reduce pitch strength slider |

## File Formats

### Voice Profiles
Profiles are saved as two files:
- `profile_name.npz` - NumPy arrays (envelope, LPC coefficients)
- `profile_name.json` - Metadata (name, F0 stats, formants)

## Future Improvements

- [ ] Formant-preserving pitch shift (PSOLA or phase vocoder)
- [ ] Spectral envelope matching  
- [ ] Real-time spectrogram display
- [ ] Preset management
- [ ] Audio file input/output
