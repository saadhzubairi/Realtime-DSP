"""Formant estimation and manipulation module."""

import numpy as np
from typing import List
from dataclasses import dataclass
from .lpc import compute_lpc


@dataclass
class Formant:
    """A single formant with frequency and bandwidth."""
    frequency: float
    bandwidth: float
    amplitude: float


@dataclass
class FormantEstimate:
    """Result of formant estimation."""
    formants: List[Formant]
    is_valid: bool


def lpc_roots_to_formants(
    lpc_coeffs: np.ndarray,
    sample_rate: int,
    max_formants: int = 3,
    min_freq: float = 90.0,
    max_freq: float = 5000.0,
    max_bandwidth: float = 400.0
) -> FormantEstimate:
    """Extract formants from LPC coefficients via root finding."""
    poly_coeffs = lpc_coeffs.copy()
    roots = np.roots(poly_coeffs)
    
    formant_candidates = []
    
    for root in roots:
        if np.imag(root) <= 0:
            continue
        
        root_mag = np.abs(root)
        if root_mag > 1.0:
            continue
        
        angle = np.angle(root)
        freq = angle * sample_rate / (2 * np.pi)
        
        if root_mag > 0:
            bandwidth = -np.log(root_mag) * sample_rate / np.pi
        else:
            bandwidth = float('inf')
        
        if min_freq <= freq <= max_freq and bandwidth <= max_bandwidth:
            formant_candidates.append(Formant(
                frequency=freq,
                bandwidth=bandwidth,
                amplitude=1.0 / (1.0 - root_mag + 1e-10)
            ))
    
    formant_candidates.sort(key=lambda f: f.frequency)
    formants = formant_candidates[:max_formants]
    is_valid = len(formants) >= 2
    
    return FormantEstimate(formants=formants, is_valid=is_valid)


def estimate_formants(
    frame: np.ndarray,
    sample_rate: int,
    lpc_order: int = 16,
    max_formants: int = 3,
    pre_emphasis: float = 0.97
) -> FormantEstimate:
    """Estimate formants from an audio frame."""
    windowed = frame * np.hanning(len(frame))
    lpc_result = compute_lpc(windowed, lpc_order, pre_emphasis)
    
    return lpc_roots_to_formants(
        lpc_result.coefficients,
        sample_rate,
        max_formants
    )


class FormantTracker:
    """Tracks formants over time with smoothing."""
    
    def __init__(
        self,
        sample_rate: int,
        lpc_order: int = 16,
        max_formants: int = 3,
        smoothing_alpha: float = 0.3
    ):
        self.sample_rate = sample_rate
        self.lpc_order = lpc_order
        self.max_formants = max_formants
        self.smoothing_alpha = smoothing_alpha
        
        self._smoothed_formants = [Formant(0, 0, 0) for _ in range(max_formants)]
        self._formant_history = [[] for _ in range(max_formants)]
    
    def process_frame(self, frame: np.ndarray) -> FormantEstimate:
        estimate = estimate_formants(
            frame,
            self.sample_rate,
            self.lpc_order,
            self.max_formants
        )
        
        if not estimate.is_valid:
            return estimate
        
        smoothed = []
        for i, formant in enumerate(estimate.formants):
            if i < len(self._smoothed_formants):
                prev = self._smoothed_formants[i]
                
                if prev.frequency > 0:
                    new_freq = self.smoothing_alpha * formant.frequency + \
                              (1 - self.smoothing_alpha) * prev.frequency
                    new_bw = self.smoothing_alpha * formant.bandwidth + \
                            (1 - self.smoothing_alpha) * prev.bandwidth
                else:
                    new_freq = formant.frequency
                    new_bw = formant.bandwidth
                
                smoothed_formant = Formant(new_freq, new_bw, formant.amplitude)
                self._smoothed_formants[i] = smoothed_formant
                smoothed.append(smoothed_formant)
                
                if i < len(self._formant_history):
                    self._formant_history[i].append(new_freq)
            else:
                smoothed.append(formant)
        
        return FormantEstimate(formants=smoothed, is_valid=True)
    
    def get_statistics(self) -> dict:
        stats = {}
        
        for i, history in enumerate(self._formant_history):
            if history:
                arr = np.array(history)
                stats[f'F{i+1}_median'] = float(np.median(arr))
                stats[f'F{i+1}_std'] = float(np.std(arr))
            else:
                stats[f'F{i+1}_median'] = 0.0
                stats[f'F{i+1}_std'] = 0.0
        
        return stats
    
    def reset(self):
        self._smoothed_formants = [Formant(0, 0, 0) for _ in range(self.max_formants)]
        self._formant_history = [[] for _ in range(self.max_formants)]
