# RVC Voice Changer - Real-time Voice Conversion

## Overview

This is a **real-time voice changer** using **RVC (Retrieval-based Voice Conversion)**, the state-of-the-art technology for voice conversion. RVC provides much higher audio quality than traditional DSP approaches while maintaining real-time performance.

## Key Differences from DSP Approach

| Feature | Old DSP Approach | New RVC Approach |
|---------|------------------|------------------|
| Quality | Robotic, choppy | Natural, high-fidelity |
| Profiles needed | Both A (source) and B (target) | Only target voice model |
| Training | Record both voices | Use pre-trained models |
| Latency | ~10-50ms | ~50-100ms (with GPU) |
| Technology | PSOLA + Formant shifting | Neural network inference |

## Installation

### 1. Install PyTorch with CUDA (Recommended)

For GPU acceleration (much faster):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CPU only (slower):
```bash
pip install torch torchvision torchaudio
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download RVC Models

You need RVC voice models (.pth files) to convert your voice. 

**Where to get models:**
- [weights.gg](https://weights.gg/) - Large collection of RVC models
- [RVC Models](https://huggingface.co/spaces/QuickWick/Music-AI-Voices) - Hugging Face collection
- Train your own using [RVC WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

### 4. Add Models to Project

1. Create a folder in `rvc_models/` for each model
2. Place the `.pth` file inside
3. Optionally add the `.index` file for better quality

```
rvc_models/
├── singer_a/
│   ├── model.pth
│   └── added_IVF512_Flat_nprobe_1.index  (optional)
├── character_b/
│   └── model.pth
└── ...
```

## Usage

### Run the RVC Voice Changer

```bash
python app_rvc.py
```

### Interface

1. **Load Model**: Select an RVC voice model file (.pth)
2. **Select Devices**: Choose your microphone and speakers
3. **Pitch Shift**: Adjust pitch up/down by semitones (e.g., +12 for male→female)
4. **Pitch Method**: 
   - `rmvpe` - Best quality (recommended)
   - `harvest` - Good quality, slower
   - `crepe` - High quality, GPU required
   - `pm` - Fastest, lower quality
5. **Start**: Begin real-time voice conversion

### Tips

- **Use headphones** to prevent feedback loops
- **Speak clearly** into the microphone
- For **male to female**: use pitch shift +6 to +12
- For **female to male**: use pitch shift -6 to -12
- Better models = better results (model quality matters!)

## Architecture

```
┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  Microphone  │───▶│ Input Buffer  │───▶│  RVC Worker  │
└──────────────┘    └───────────────┘    └──────┬───────┘
                                                 │
                    ┌───────────────┐    ┌───────▼───────┐
                    │ Output Buffer │◀───│ RVC Pipeline  │
                    └───────┬───────┘    └───────────────┘
                            │                    │
                    ┌───────▼───────┐    ┌───────▼───────┐
                    │   Speakers    │    │ - F0 Extract  │
                    └───────────────┘    │ - HuBERT enc  │
                                         │ - RVC infer   │
                                         └───────────────┘
```

### Files

- `app_rvc.py` - Main application with UI
- `rvc/inference.py` - RVC model wrapper
- `rvc/realtime.py` - Real-time streaming pipeline

## Requirements

- **Python 3.8+**
- **PyTorch 2.0+** (CUDA recommended)
- **16GB RAM** minimum
- **NVIDIA GPU** with 4GB+ VRAM (recommended for real-time)

## Troubleshooting

### "Audio is delayed"
- Reduce block size in settings
- Use a faster pitch method (pm)
- Ensure GPU is being used

### "No audio output"
- Check that the model is loaded (green checkmark)
- Check device selection
- Try bypass mode to test audio path

### "Model failed to load"
- Ensure the .pth file is a valid RVC model
- Check if dependencies are installed
- Check console for error messages

### "Out of memory"
- Close other applications
- Use a smaller model
- Reduce buffer size

## Training Your Own Models

To create a custom voice model from recordings:

1. Install [RVC WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
2. Record 10-15 minutes of the target voice
3. Train the model (takes 30-60 min with GPU)
4. Export the .pth file
5. Copy to `rvc_models/`

## Credits

- [RVC Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) - Original RVC implementation
- [rvc-python](https://pypi.org/project/rvc-python/) - Python wrapper library
- [RMVPE](https://github.com/Dream-High/RMVPE) - Robust pitch extraction
