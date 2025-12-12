"""
RVC Inference wrapper for voice conversion.

Wraps the rvc-python library for easy integration.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from pathlib import Path
import threading
import os

# Try to import rvc_python
try:
    from rvc_python.infer import RVCInference
    _RVC_AVAILABLE = True
except ImportError:
    RVCInference = None
    _RVC_AVAILABLE = False


@dataclass
class RVCConfig:
    """Configuration for RVC voice conversion."""
    
    # Model settings
    model_path: Optional[str] = None
    index_path: Optional[str] = None
    
    # Pitch settings
    pitch_shift: int = 0  # Semitones (-12 to +12)
    
    # Pitch extraction method
    # Options: 'rmvpe' (best), 'harvest', 'crepe', 'pm' (fastest)
    f0_method: str = 'rmvpe'
    
    # Quality settings
    index_rate: float = 0.5  # Feature search ratio (0.0-1.0)
    filter_radius: int = 3   # Median filtering radius for pitch
    resample_sr: int = 0     # Output resampling (0 = no resampling)
    rms_mix_rate: float = 0.25  # Volume envelope mix rate
    protect: float = 0.33    # Protection for voiceless consonants
    
    # Device settings
    device: str = 'cuda:0'  # 'cpu', 'cuda:0', etc.
    
    # Real-time settings
    block_time: float = 0.25  # Processing block time in seconds
    crossfade_time: float = 0.05  # Crossfade duration in seconds
    extra_time: float = 2.0  # Extra context time for pitch extraction


@dataclass
class VoiceModel:
    """Represents a loaded RVC voice model."""
    name: str
    path: str
    index_path: Optional[str] = None
    sample_rate: int = 40000  # RVC models typically use 40kHz or 48kHz
    

class RVCVoiceConverter:
    """
    RVC-based voice converter.
    
    Converts input audio to sound like a target voice using a pre-trained
    RVC model. Only requires the target voice model - no source profile needed.
    """
    
    def __init__(self, config: Optional[RVCConfig] = None):
        """
        Initialize the RVC voice converter.
        
        Args:
            config: RVC configuration. If None, uses defaults.
        """
        if not _RVC_AVAILABLE:
            raise RuntimeError(
                "RVC is not available. Please install rvc-python:\n"
                "  pip install rvc-python\n"
                "  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118"
            )
        
        self.config = config or RVCConfig()
        self._lock = threading.Lock()
        
        # Initialize RVC inference engine
        self._rvc: Optional[RVCInference] = None
        self._model_loaded = False
        self._current_model: Optional[VoiceModel] = None
        
        # Initialize the inference engine
        self._initialize()
    
    def _initialize(self):
        """Initialize the RVC inference engine."""
        try:
            self._rvc = RVCInference(device=self.config.device)
            self._apply_config()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize RVC: {e}")
    
    def _apply_config(self):
        """Apply current configuration to the RVC engine."""
        if self._rvc is None:
            return
        
        # Set parameters
        self._rvc.set_params(
            f0method=self.config.f0_method,
            f0up_key=self.config.pitch_shift,
            index_rate=self.config.index_rate,
            filter_radius=self.config.filter_radius,
            resample_sr=self.config.resample_sr,
            rms_mix_rate=self.config.rms_mix_rate,
            protect=self.config.protect,
        )
    
    def load_model(self, model_path: str, index_path: Optional[str] = None) -> bool:
        """
        Load an RVC voice model.
        
        Args:
            model_path: Path to the .pth model file
            index_path: Optional path to the .index file for better quality
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model not found: {model_path}")
                
                self._rvc.load_model(model_path, index_path=index_path)
                
                # Store model info
                model_name = Path(model_path).stem
                self._current_model = VoiceModel(
                    name=model_name,
                    path=model_path,
                    index_path=index_path
                )
                self._model_loaded = True
                
                return True
                
            except Exception as e:
                print(f"Failed to load RVC model: {e}")
                self._model_loaded = False
                return False
    
    def unload_model(self):
        """Unload the current model."""
        with self._lock:
            if self._rvc is not None:
                self._rvc.unload_model()
            self._model_loaded = False
            self._current_model = None
    
    @property
    def is_ready(self) -> bool:
        """Check if a model is loaded and ready for conversion."""
        return self._model_loaded and self._rvc is not None
    
    @property
    def current_model(self) -> Optional[VoiceModel]:
        """Get info about the currently loaded model."""
        return self._current_model
    
    def set_pitch_shift(self, semitones: int):
        """
        Set pitch shift in semitones.
        
        Args:
            semitones: Pitch shift (-12 to +12, 0 = no shift)
        """
        self.config.pitch_shift = max(-12, min(12, semitones))
        if self._rvc is not None:
            self._rvc.set_params(f0up_key=self.config.pitch_shift)
    
    def set_f0_method(self, method: str):
        """
        Set the pitch extraction method.
        
        Args:
            method: 'rmvpe' (best), 'harvest', 'crepe', 'pm' (fastest)
        """
        valid_methods = ['rmvpe', 'harvest', 'crepe', 'pm']
        if method not in valid_methods:
            raise ValueError(f"Invalid f0 method. Use one of: {valid_methods}")
        
        self.config.f0_method = method
        if self._rvc is not None:
            self._rvc.set_params(f0method=method)
    
    def convert_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> Tuple[np.ndarray, int]:
        """
        Convert audio to the target voice.
        
        Args:
            audio: Input audio samples (float32, mono)
            sample_rate: Sample rate of input audio
            
        Returns:
            Tuple of (converted audio, output sample rate)
        """
        if not self.is_ready:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        with self._lock:
            # RVC expects float32 audio
            audio = audio.astype(np.float32)
            
            # Convert using RVC
            output = self._rvc.infer(
                audio,
                input_sr=sample_rate
            )
            
            # Get output sample rate (usually 40000 or 48000)
            output_sr = self._rvc.output_sr if hasattr(self._rvc, 'output_sr') else 40000
            
            return output, output_sr
    
    def convert_file(
        self,
        input_path: str,
        output_path: str
    ) -> bool:
        """
        Convert an audio file to the target voice.
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output audio file
            
        Returns:
            True if successful
        """
        if not self.is_ready:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        with self._lock:
            try:
                self._rvc.infer_file(input_path, output_path)
                return True
            except Exception as e:
                print(f"Conversion failed: {e}")
                return False


def list_available_models(models_dir: str = "rvc_models") -> List[VoiceModel]:
    """
    List available RVC models in a directory.
    
    Args:
        models_dir: Directory containing RVC models
        
    Returns:
        List of VoiceModel objects
    """
    models = []
    models_path = Path(models_dir)
    
    if not models_path.exists():
        return models
    
    # Look for .pth files
    for pth_file in models_path.rglob("*.pth"):
        # Look for matching index file
        index_file = None
        for idx in pth_file.parent.glob("*.index"):
            index_file = str(idx)
            break
        
        models.append(VoiceModel(
            name=pth_file.stem,
            path=str(pth_file),
            index_path=index_file
        ))
    
    return models
