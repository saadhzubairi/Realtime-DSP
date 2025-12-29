# Real-Time Voice Transformer

A real-time voice transformation application using the WORLD vocoder for high-quality pitch and formant shifting.

## Features

- **Voice Calibration**: Record your voice and a target voice to automatically calculate transformation parameters
- **Real-Time Processing**: Transform your voice in real-time with low latency
- **WORLD Vocoder**: High-quality voice synthesis with independent pitch and formant control
- **Simple GUI**: Easy-to-use tkinter interface with audio device selection

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python app.py
```

### Quick Start

1. **Calibration Tab**: Record your voice (Profile A) and the target voice (Profile B)
2. **Live Tab**: Click "Apply Calibration" to set the calculated parameters
3. Click **Start** to begin real-time voice transformation

## Project Structure

```
Project Final/
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── audio/              # Audio I/O modules
│   ├── pyaudio_io.py   # PyAudio streaming
│   ├── recorder.py     # Audio recording/playback
│   └── ringbuffer.py   # Circular buffers
├── dsp/                # DSP processing modules
│   ├── pitch.py        # YIN pitch detection
│   ├── lpc.py          # Linear prediction
│   ├── formant.py      # Formant tracking
│   ├── voice_profile.py # Profile extraction
│   └── world_vocoder.py # WORLD vocoder
├── utils/              # Utilities
│   ├── config.py       # Configuration
│   └── logging_utils.py # Logging
└── profiles/           # Saved voice profiles
```

## Requirements

- Python 3.8+
- PyAudio
- NumPy
- SciPy
- PyWorld
