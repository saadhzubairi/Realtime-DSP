"""
LPC (Linear Predictive Coding) module.
Provides LPC analysis, spectral envelope estimation, and synthesis filtering.
"""

import numpy as np
from scipy import signal
from scipy.linalg import solve_toeplitz
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class LPCResult:
    """Result of LPC analysis."""
    coefficients: np.ndarray    # LPC coefficients [1, -a1, -a2, ..., -ap]
    error: float                # Prediction error
    gain: float                 # Filter gain
    

def autocorrelation(x: np.ndarray, order: int) -> np.ndarray:
    """
    Compute autocorrelation coefficients.
    
    Args:
        x: Input signal
        order: Number of lags to compute
        
    Returns:
        Autocorrelation values [r(0), r(1), ..., r(order)]
    """
    n = len(x)
    r = np.zeros(order + 1)
    
    for k in range(order + 1):
        r[k] = np.sum(x[:n-k] * x[k:])
    
    return r


def levinson_durbin(r: np.ndarray, order: int) -> Tuple[np.ndarray, float]:
    """
    Levinson-Durbin recursion for solving LPC coefficients.
    
    Args:
        r: Autocorrelation coefficients
        order: LPC order
        
    Returns:
        Tuple of (LPC coefficients, prediction error)
    """
    # Initialize
    a = np.zeros(order + 1)
    a[0] = 1.0
    
    e = r[0]
    
    for i in range(1, order + 1):
        # Compute reflection coefficient
        lambda_i = 0.0
        for j in range(1, i):
            lambda_i += a[j] * r[i - j]
        lambda_i = (r[i] - lambda_i) / e
        
        # Update coefficients
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] - lambda_i * a[i - j]
        a_new[i] = lambda_i
        
        a = a_new
        e = e * (1 - lambda_i ** 2)
    
    # Return coefficients as [1, -a1, -a2, ...]
    a[1:] = -a[1:]
    
    return a, e


def compute_lpc(
    frame: np.ndarray,
    order: int,
    pre_emphasis: float = 0.97
) -> LPCResult:
    """
    Compute LPC coefficients for a frame.
    
    Args:
        frame: Input frame
        order: LPC order
        pre_emphasis: Pre-emphasis coefficient (0 to disable)
        
    Returns:
        LPCResult with coefficients, error, and gain
    """
    # Apply pre-emphasis
    if pre_emphasis > 0:
        emphasized = np.zeros_like(frame)
        emphasized[0] = frame[0]
        emphasized[1:] = frame[1:] - pre_emphasis * frame[:-1]
    else:
        emphasized = frame
    
    # Compute autocorrelation
    r = autocorrelation(emphasized, order)
    
    # Handle zero signal
    if r[0] < 1e-10:
        return LPCResult(
            coefficients=np.zeros(order + 1),
            error=0.0,
            gain=0.0
        )
    
    # Solve using Levinson-Durbin
    coeffs, error = levinson_durbin(r, order)
    
    # Compute gain
    gain = np.sqrt(max(error, 0))
    
    return LPCResult(
        coefficients=coeffs,
        error=error,
        gain=gain
    )


def lpc_to_spectrum(
    lpc_coeffs: np.ndarray,
    fft_size: int,
    gain: float = 1.0
) -> np.ndarray:
    """
    Convert LPC coefficients to spectral envelope magnitude.
    
    Args:
        lpc_coeffs: LPC coefficients [1, -a1, -a2, ...]
        fft_size: FFT size for spectrum computation
        gain: LPC gain (for amplitude scaling)
        
    Returns:
        Spectral envelope magnitude (rfft bins)
    """
    # Compute frequency response of LPC filter
    # H(z) = gain / A(z) where A(z) is the LPC polynomial
    
    n_bins = fft_size // 2 + 1
    
    # Evaluate A(z) on the unit circle
    # A(e^{jw}) = sum_{k=0}^{p} a[k] * e^{-jwk}
    
    freqs = np.linspace(0, np.pi, n_bins)
    A = np.zeros(n_bins, dtype=complex)
    
    for k, a_k in enumerate(lpc_coeffs):
        A += a_k * np.exp(-1j * k * freqs)
    
    # Spectral envelope is |H(w)| = gain / |A(w)|
    A_mag = np.abs(A)
    A_mag[A_mag < 1e-10] = 1e-10  # Prevent division by zero
    
    envelope = gain / A_mag
    
    return envelope.astype(np.float32)


def spectrum_to_lpc(
    envelope: np.ndarray,
    order: int
) -> np.ndarray:
    """
    Convert spectral envelope to LPC coefficients.
    
    Uses the autocorrelation method via IFFT.
    
    Args:
        envelope: Spectral envelope magnitude
        order: Desired LPC order
        
    Returns:
        LPC coefficients
    """
    # Convert to full symmetric spectrum
    n_bins = len(envelope)
    fft_size = (n_bins - 1) * 2
    
    # Compute log power spectrum
    log_power = 2 * np.log(np.maximum(envelope, 1e-10))
    
    # Full symmetric spectrum for IFFT
    full_spectrum = np.zeros(fft_size)
    full_spectrum[:n_bins] = log_power
    full_spectrum[n_bins:] = log_power[-2:0:-1]
    
    # IFFT to get cepstrum-like autocorrelation
    autocorr = np.fft.ifft(np.exp(full_spectrum)).real
    
    # Use Levinson-Durbin
    r = autocorr[:order + 1]
    coeffs, _ = levinson_durbin(r, order)
    
    return coeffs


def lpc_filter(
    excitation: np.ndarray,
    lpc_coeffs: np.ndarray,
    gain: float = 1.0,
    state: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply LPC synthesis filter to excitation signal.
    
    y[n] = gain * x[n] + sum_{k=1}^{p} a[k] * y[n-k]
    
    Args:
        excitation: Input excitation signal
        lpc_coeffs: LPC coefficients [1, -a1, -a2, ...]
        gain: Filter gain
        state: Initial filter state (for continuity)
        
    Returns:
        Tuple of (filtered output, final state)
    """
    order = len(lpc_coeffs) - 1
    
    if state is None:
        state = np.zeros(order, dtype=np.float32)
    
    # IIR filter: H(z) = gain / A(z)
    # Numerator is just gain, denominator is LPC polynomial
    b = np.array([gain], dtype=np.float32)
    a = lpc_coeffs.astype(np.float32)
    
    # Use scipy's lfilter with initial conditions
    zi = signal.lfilter_zi(b, a) * state[0] if len(state) > 0 and state[0] != 0 else np.zeros(max(len(a), len(b)) - 1)
    
    output, zf = signal.lfilter(b, a, excitation, zi=zi)
    
    # Update state
    new_state = np.zeros(order, dtype=np.float32)
    new_state[:min(order, len(zf))] = zf[:min(order, len(zf))]
    
    return output.astype(np.float32), new_state


def lpc_inverse_filter(
    signal_in: np.ndarray,
    lpc_coeffs: np.ndarray,
    state: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply LPC inverse (analysis) filter to get residual.
    
    e[n] = x[n] - sum_{k=1}^{p} a[k] * x[n-k]
    
    Args:
        signal_in: Input signal
        lpc_coeffs: LPC coefficients
        state: Initial filter state
        
    Returns:
        Tuple of (residual/excitation, final state)
    """
    order = len(lpc_coeffs) - 1
    
    if state is None:
        state = np.zeros(order, dtype=np.float32)
    
    # FIR filter with LPC coefficients
    b = lpc_coeffs.astype(np.float32)
    a = np.array([1.0], dtype=np.float32)
    
    zi = signal.lfilter_zi(b, a) * state[0] if len(state) > 0 and state[0] != 0 else np.zeros(len(b) - 1)
    
    residual, zf = signal.lfilter(b, a, signal_in, zi=zi)
    
    new_state = np.zeros(order, dtype=np.float32)
    new_state[:min(order, len(zf))] = zf[:min(order, len(zf))]
    
    return residual.astype(np.float32), new_state


def smooth_envelope(
    envelope: np.ndarray,
    smoothing_bins: int = 3
) -> np.ndarray:
    """
    Smooth a spectral envelope.
    
    Args:
        envelope: Input envelope
        smoothing_bins: Number of bins for moving average
        
    Returns:
        Smoothed envelope
    """
    if smoothing_bins <= 1:
        return envelope
    
    kernel = np.ones(smoothing_bins) / smoothing_bins
    smoothed = np.convolve(envelope, kernel, mode='same')
    
    return smoothed.astype(np.float32)


def interpolate_envelope(
    envelope: np.ndarray,
    target_size: int
) -> np.ndarray:
    """
    Interpolate envelope to a different size.
    
    Args:
        envelope: Input envelope
        target_size: Target number of bins
        
    Returns:
        Interpolated envelope
    """
    x_old = np.linspace(0, 1, len(envelope))
    x_new = np.linspace(0, 1, target_size)
    
    return np.interp(x_new, x_old, envelope).astype(np.float32)
