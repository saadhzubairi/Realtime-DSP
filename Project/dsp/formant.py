"""
Formant estimation and manipulation module.
Provides formant tracking and frequency warping for vocal tract length modification.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .lpc import compute_lpc, lpc_to_spectrum


@dataclass
class Formant:
    """A single formant with frequency and bandwidth."""
    frequency: float   # Hz
    bandwidth: float   # Hz
    amplitude: float   # Linear amplitude (optional)


@dataclass
class FormantEstimate:
    """Result of formant estimation."""
    formants: List[Formant]  # List of detected formants (F1, F2, F3, ...)
    is_valid: bool           # Whether estimation is reliable


def lpc_roots_to_formants(
    lpc_coeffs: np.ndarray,
    sample_rate: int,
    max_formants: int = 3,
    min_freq: float = 90.0,
    max_freq: float = 5000.0,
    max_bandwidth: float = 400.0
) -> FormantEstimate:
    """
    Extract formants from LPC coefficients via root finding.
    
    Formants correspond to complex conjugate pole pairs of the LPC polynomial.
    
    Args:
        lpc_coeffs: LPC coefficients [1, -a1, -a2, ...]
        sample_rate: Audio sample rate
        max_formants: Maximum number of formants to return
        min_freq: Minimum formant frequency
        max_freq: Maximum formant frequency  
        max_bandwidth: Maximum acceptable bandwidth
        
    Returns:
        FormantEstimate with list of detected formants
    """
    # Find roots of LPC polynomial
    # A(z) = 1 - a1*z^-1 - a2*z^-2 - ... = 0
    # Multiply by z^p: z^p - a1*z^{p-1} - a2*z^{p-2} - ... = 0
    
    # Convert to standard polynomial form for root finding
    # coeffs are [1, -a1, -a2, ...] so polynomial is 1 - a1*z^-1 - ...
    # For np.roots, we need [a_n, a_{n-1}, ..., a_1, a_0]
    
    poly_coeffs = lpc_coeffs.copy()
    roots = np.roots(poly_coeffs)
    
    # Filter to keep only stable roots inside unit circle
    # and with positive imaginary part (to avoid conjugate duplicates)
    formant_candidates = []
    
    for root in roots:
        # Only consider roots with positive imaginary part
        if np.imag(root) <= 0:
            continue
        
        # Root should be inside (or on) unit circle for stability
        root_mag = np.abs(root)
        if root_mag > 1.0:
            continue
        
        # Convert to frequency and bandwidth
        # Frequency: angle of root
        angle = np.angle(root)
        freq = angle * sample_rate / (2 * np.pi)
        
        # Bandwidth: related to distance from unit circle
        # BW = -ln(|root|) * fs / pi
        if root_mag > 0:
            bandwidth = -np.log(root_mag) * sample_rate / np.pi
        else:
            bandwidth = float('inf')
        
        # Filter by frequency and bandwidth constraints
        if min_freq <= freq <= max_freq and bandwidth <= max_bandwidth:
            formant_candidates.append(Formant(
                frequency=freq,
                bandwidth=bandwidth,
                amplitude=1.0 / (1.0 - root_mag + 1e-10)  # Rough amplitude estimate
            ))
    
    # Sort by frequency
    formant_candidates.sort(key=lambda f: f.frequency)
    
    # Take up to max_formants
    formants = formant_candidates[:max_formants]
    
    # Validity check
    is_valid = len(formants) >= 2  # At least F1 and F2
    
    return FormantEstimate(formants=formants, is_valid=is_valid)


def estimate_formants(
    frame: np.ndarray,
    sample_rate: int,
    lpc_order: int = 16,
    max_formants: int = 3,
    pre_emphasis: float = 0.97
) -> FormantEstimate:
    """
    Estimate formants from an audio frame.
    
    Args:
        frame: Audio frame
        sample_rate: Sample rate
        lpc_order: LPC order (should be ~2-4 per expected formant + 4)
        max_formants: Maximum number of formants
        pre_emphasis: Pre-emphasis coefficient
        
    Returns:
        FormantEstimate
    """
    # Apply window
    windowed = frame * np.hanning(len(frame))
    
    # Compute LPC
    lpc_result = compute_lpc(windowed, lpc_order, pre_emphasis)
    
    # Extract formants from roots
    return lpc_roots_to_formants(
        lpc_result.coefficients,
        sample_rate,
        max_formants
    )


def compute_formant_warp_factor(
    source_formants: FormantEstimate,
    target_formants: FormantEstimate,
    formant_index: int = 0,  # 0 = F1, 1 = F2, etc.
    clamp_range: Tuple[float, float] = (0.8, 1.25)
) -> float:
    """
    Compute vocal tract length warp factor from formant ratio.
    
    Args:
        source_formants: Source voice formants
        target_formants: Target voice formants
        formant_index: Which formant to use (0=F1, 1=F2)
        clamp_range: Min/max warp factor
        
    Returns:
        Warp factor (target/source ratio, clamped)
    """
    if not source_formants.is_valid or not target_formants.is_valid:
        return 1.0
    
    if formant_index >= len(source_formants.formants):
        return 1.0
    if formant_index >= len(target_formants.formants):
        return 1.0
    
    source_f = source_formants.formants[formant_index].frequency
    target_f = target_formants.formants[formant_index].frequency
    
    if source_f <= 0:
        return 1.0
    
    warp = target_f / source_f
    
    # Clamp to reasonable range
    warp = max(clamp_range[0], min(clamp_range[1], warp))
    
    return warp


def warp_envelope(
    envelope: np.ndarray,
    warp_factor: float,
    sample_rate: int
) -> np.ndarray:
    """
    Warp a spectral envelope by a frequency scaling factor.
    
    This effectively simulates vocal tract length modification.
    
    E_warp(f) = E(f / warp_factor)
    
    Args:
        envelope: Spectral envelope (magnitude, rfft bins)
        warp_factor: Frequency warp factor (>1 = shift up, <1 = shift down)
        sample_rate: Sample rate
        
    Returns:
        Warped envelope
    """
    n_bins = len(envelope)
    
    if abs(warp_factor - 1.0) < 1e-6:
        return envelope.copy()
    
    # Create warped frequency axis
    # Original bins: [0, 1, 2, ..., n_bins-1]
    # Warped bins: [0, 1/w, 2/w, ..., (n_bins-1)/w]
    
    original_bins = np.arange(n_bins)
    warped_bins = original_bins / warp_factor
    
    # Interpolate envelope at warped positions
    warped_envelope = np.interp(
        original_bins,
        warped_bins,
        envelope,
        left=envelope[0],
        right=envelope[-1]
    )
    
    return warped_envelope.astype(np.float32)


def bilinear_warp_envelope(
    envelope: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Warp envelope using bilinear frequency warping.
    
    This is a more sophisticated warping that better matches
    human perception (similar to mel scale warping).
    
    Args:
        envelope: Spectral envelope
        alpha: Warp coefficient (-1 < alpha < 1)
               Positive = expand low frequencies
               Negative = compress low frequencies
               
    Returns:
        Warped envelope
    """
    n_bins = len(envelope)
    
    if abs(alpha) < 1e-6:
        return envelope.copy()
    
    # Bilinear warping: w' = arctan((1-a^2)sin(w) / ((1+a^2)cos(w) - 2a))
    # Simplified for frequency bins
    
    original_bins = np.arange(n_bins)
    omega = np.pi * original_bins / (n_bins - 1)  # Normalized frequency [0, pi]
    
    # Bilinear transform
    numerator = (1 - alpha**2) * np.sin(omega)
    denominator = (1 + alpha**2) * np.cos(omega) - 2 * alpha
    
    omega_warped = np.arctan2(numerator, denominator)
    omega_warped = np.mod(omega_warped + np.pi, 2*np.pi) - np.pi  # Wrap to [-pi, pi]
    omega_warped[omega_warped < 0] += np.pi  # Ensure positive
    
    warped_bins = omega_warped * (n_bins - 1) / np.pi
    
    # Interpolate
    warped_envelope = np.interp(
        original_bins,
        warped_bins,
        envelope,
        left=envelope[0],
        right=envelope[-1]
    )
    
    return warped_envelope.astype(np.float32)


class FormantTracker:
    """
    Tracks formants over time with smoothing.
    """
    
    def __init__(
        self,
        sample_rate: int,
        lpc_order: int = 16,
        max_formants: int = 3,
        smoothing_alpha: float = 0.3
    ):
        """
        Initialize formant tracker.
        
        Args:
            sample_rate: Audio sample rate
            lpc_order: LPC order for analysis
            max_formants: Number of formants to track
            smoothing_alpha: EMA smoothing factor
        """
        self.sample_rate = sample_rate
        self.lpc_order = lpc_order
        self.max_formants = max_formants
        self.smoothing_alpha = smoothing_alpha
        
        # Smoothed formant values
        self._smoothed_formants = [Formant(0, 0, 0) for _ in range(max_formants)]
        
        # Statistics
        self._formant_history = [[] for _ in range(max_formants)]
    
    def process_frame(self, frame: np.ndarray) -> FormantEstimate:
        """
        Process a frame and return smoothed formants.
        
        Args:
            frame: Audio frame
            
        Returns:
            Smoothed formant estimate
        """
        # Get raw estimate
        estimate = estimate_formants(
            frame,
            self.sample_rate,
            self.lpc_order,
            self.max_formants
        )
        
        if not estimate.is_valid:
            return estimate
        
        # Smooth each formant
        smoothed = []
        for i, formant in enumerate(estimate.formants):
            if i < len(self._smoothed_formants):
                # EMA smoothing
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
                
                # Update history
                if i < len(self._formant_history):
                    self._formant_history[i].append(new_freq)
            else:
                smoothed.append(formant)
        
        return FormantEstimate(formants=smoothed, is_valid=True)
    
    def get_statistics(self) -> dict:
        """Get formant statistics."""
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
        """Reset tracker state."""
        self._smoothed_formants = [Formant(0, 0, 0) for _ in range(self.max_formants)]
        self._formant_history = [[] for _ in range(self.max_formants)]
