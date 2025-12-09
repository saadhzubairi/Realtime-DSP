# Voice Transform - Real-time A→B Voice Conversion

A real-time voice transformation system that converts one person's voice to sound like another's using pitch mapping, spectral envelope matching, and formant warping.

## Project Structure

```
Project/
├── app.py                 # Main application entry point (Tkinter + threading)
├── requirements.txt       # Python dependencies
├── profiles/             # Saved voice profiles
│
├── audio/                # Audio I/O layer
│   ├── __init__.py
│   ├── pyaudio_io.py     # PyAudio stream, device enumeration, callback
│   └── ringbuffer.py     # Lock-free ring buffers for thread communication
│
├── dsp/                  # Signal processing modules
│   ├── __init__.py
│   ├── stft.py          # STFT/ISTFT, windowing, overlap-add
│   ├── pitch.py         # YIN pitch detection, pitch tracking & mapping
│   ├── lpc.py           # LPC analysis, spectral envelope estimation
│   ├── formant.py       # Formant estimation & frequency warping
│   ├── voice_profile.py # Profile extraction & serialization
│   └── transform.py     # Live A→B transform pipeline
│
├── ui/                   # Tkinter UI tabs
│   ├── __init__.py
│   ├── devices.py       # Tab 1: Device selection, buffer config
│   ├── calibration.py   # Tab 2: Voice recording & profile extraction
│   ├── live.py          # Tab 3: Real-time transform controls
│   └── diagnostics.py   # Tab 4: Performance monitoring, debug
│
└── utils/                # Utilities
    ├── __init__.py
    ├── config.py        # Configuration constants & dataclasses
    ├── logging_utils.py # Thread-safe logging
    └── timing.py        # High-resolution timing & latency estimation
```

## Architecture

### Threading Model

```
┌─────────────────────────────────────────────────────────────────┐
│                     Main Thread (Tkinter UI)                     │
│  - Handles user interaction                                      │
│  - Polls shared state every 30-60ms via after()                 │
│  - Updates meters, plots, logs                                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
┌─────────────────────────────┐ ┌─────────────────────────────────┐
│   Audio I/O Thread          │ │    DSP Worker Thread            │
│   (PyAudio callback)        │ │                                 │
│                             │ │  - Pulls frames from input      │
│  - Must be FAST             │ │    ring buffer                  │
│  - Copy input → ring buffer │ │  - Runs transform pipeline      │
│  - Pop output if available  │ │  - Pushes to output ring buffer │
│  - Else play silence        │ │                                 │
└─────────────────────────────┘ └─────────────────────────────────┘
         │                                  ▲
         │       ┌─────────────────────┐    │
         └──────►│  Input Ring Buffer  │────┘
                 └─────────────────────┘
                 ┌─────────────────────┐
         ┌──────►│ Output Ring Buffer  │◄────┐
         │       └─────────────────────┘     │
         └───────────────────────────────────┘
```

### Data Flow

```
Mic → Input Ring Buffer → Frame Extraction (overlap)
                              ↓
                    ┌─────────────────────┐
                    │   DSP Pipeline      │
                    │                     │
                    │  1. STFT Analysis   │
                    │  2. Pitch Detection │
                    │  3. Pitch Mapping   │
                    │  4. Envelope Match  │
                    │  5. Formant Warp    │
                    │  6. ISTFT Synthesis │
                    └─────────────────────┘
                              ↓
           Overlap-Add Reconstruction → Output Ring Buffer → Speakers
```

## Configuration

### Target Constraints (in `utils/config.py`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| Sample Rate | 16 kHz | CPU-light; 48 kHz for higher fidelity |
| Frame Size | 20 ms (320 samples @ 16kHz) | Analysis frame |
| Hop Size | 10 ms (160 samples @ 16kHz) | 50% overlap |
| Buffer Size | 256 samples | PyAudio callback buffer |
| LPC Order | 16 | ~fs/1000 + 4 |
| FFT Size | 512 | For spectral analysis |

## UI Tabs

### Tab 1: Setup
- Input/output device selection
- Sample rate (16/48 kHz)
- Buffer size slider (128-1024)
- Mic level meter with clip indicator
- Test loopback button

### Tab 2: Calibration
- Record Profile A (source voice)
- Record Profile B (target voice)
- 5-10 second recordings with countdown
- Waveform preview and playback
- Feature extraction and summary:
  - F0 median, 5th/95th percentile range
  - Formant estimates (F1, F2, F3)
  - Spectral envelope
- Save/load profiles (.npz + .json)

### Tab 3: Live Transform
- Start/Stop stream
- Profile display
- Transform controls (with EMA smoothing):
  - Wet/Dry mix (0-100%)
  - Pitch mapping strength (0-100%)
  - Formant mapping strength (0-100%)
  - Envelope match strength (0-100%)
- Unvoiced handling mode (bypass / noise-shaped)
- Real-time meters:
  - Input/output levels (dB)
  - Estimated F0 (Hz)
  - Underrun/overrun counters
  - Queue depth

### Tab 4: Diagnostics
- Latency estimate (ms)
- CPU time per frame (avg/max)
- DSP CPU load percentage
- Debug bypass toggles:
  - Bypass pitch mapping
  - Bypass envelope matching
  - Bypass formant warping
- Live spectrogram
- Event log viewer

## Voice Profile Features

Each profile contains:

| Feature | Description |
|---------|-------------|
| `f0_median_hz` | Median fundamental frequency |
| `f0_p05_hz` / `f0_p95_hz` | 5th/95th percentile F0 |
| `f0_std_log` | F0 standard deviation (log domain) |
| `voiced_ratio` | Proportion of voiced frames |
| `envelope_log_mag` | Average spectral envelope (log magnitude) |
| `lpc_coefficients` | Average LPC coefficients |
| `formant_f1/f2/f3_median` | Median formant frequencies |

## DSP Pipeline Stages

### Stage A: STFT/ISTFT Base Plumbing
- Overlap-add with 50% overlap
- Hann window for analysis and synthesis
- Proper COLA normalization

### Stage B: Pitch Mapping
- YIN algorithm for F0 estimation
- Median filtering for smoothing
- Linear mapping in log-F0 domain:
  ```
  p' = μB + (σB/σA) * (p - μA)
  p_out = (1-strength)*p + strength*p'
  ```

### Stage C: Spectral Envelope Matching
- LPC-based envelope estimation per frame
- Precomputed target envelope from calibration
- Apply envelope ratio:
  ```
  |Y(f)| = |X(f)| * (E_target(f) / E_source(f))^strength
  ```
- EMA smoothing to prevent artifacts

### Stage D: Formant Warping
- Frequency warping of target envelope
- Warp factor from F1 ratio: `s = F1_target / F1_source`
- Applied via interpolation: `E_warp(f) = E(f / s)`

### Stage E: Voiced/Unvoiced Handling
- Bypass processing for unvoiced frames (or reduced strength)
- Prevents noise amplification

## Milestone Build Order

1. ✅ Device selection UI + loopback passthrough
2. ✅ Ring buffers + callback stability + meters
3. ✅ STFT/ISTFT no-op pipeline
4. ✅ Calibration recording + profile extraction
5. ✅ Envelope match only
6. ✅ Pitch mapping only
7. ✅ Combined pitch + envelope
8. ✅ Formant warp as envelope warping
9. ⬜ Diagnostics + profiling + hardening

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Running

```bash
python app.py
```

### Workflow

1. **Setup**: Select input/output devices, test with loopback
2. **Calibrate**: Record 5-10s of your voice (Profile A), then target voice (Profile B)
3. **Extract**: Click "Extract Profile Features" for each
4. **Transform**: Go to Live tab, adjust strengths, click Start

## Requirements

- Python 3.8+
- NumPy
- SciPy
- PyAudio (requires PortAudio)
- tkinter (usually bundled with Python)

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Robotic" sound | Too-strong envelope ratio | Reduce envelope strength, increase smoothing |
| "Chipmunk" sound | Pitch shift without formant compensation | Enable formant warping |
| "Buzzy pumping" | Envelope ratio not smoothed | Increase EMA smoothing alpha |
| Buffer underruns | DSP too slow | Increase buffer size, reduce sample rate |
| High latency | Large buffers or queue depth | Reduce buffer size, optimize DSP |

### Validation Checklist

- [ ] Queue depth stable over time (no drift)
- [ ] 10-20 minute stress test (no buffer growth/leaks)
- [ ] A/B test: envelope-only vs pitch-only vs both
- [ ] Unvoiced bypass vs processed comparison

## Future Improvements

- [ ] Phase vocoder with identity phase locking
- [ ] Real-time LPC envelope (replace smoothing)
- [ ] De-essing and tilt EQ
- [ ] Output limiter
- [ ] WORLD vocoder integration for higher quality
