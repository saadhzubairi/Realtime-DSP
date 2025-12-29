"""LPC (Linear Predictive Coding) module."""

import numpy as np
from scipy import signal
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class LPCResult:
    """Result of LPC analysis."""
    coefficients: np.ndarray
    error: float
    gain: float


def autocorrelation(x: np.ndarray, order: int) -> np.ndarray:
    """Compute autocorrelation coefficients using FFT."""
    n = len(x)
    
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    
    x_padded = np.zeros(fft_size)
    x_padded[:n] = x
    X = np.fft.rfft(x_padded)
    r_full = np.fft.irfft(X * np.conj(X))
    
    return r_full[:order + 1].astype(np.float64)


def levinson_durbin(r: np.ndarray, order: int) -> Tuple[np.ndarray, float]:
    """Levinson-Durbin recursion for solving LPC coefficients."""
    a = np.zeros(order + 1)
    a[0] = 1.0
    
    e = r[0]
    
    for i in range(1, order + 1):
        lambda_i = 0.0
        for j in range(1, i):
            lambda_i += a[j] * r[i - j]
        lambda_i = (r[i] - lambda_i) / e
        
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] - lambda_i * a[i - j]
        a_new[i] = lambda_i
        
        a = a_new
        e = e * (1 - lambda_i ** 2)
    
    a[1:] = -a[1:]
    return a, e


def compute_lpc(frame: np.ndarray, order: int, pre_emphasis: float = 0.97) -> LPCResult:
    """Compute LPC coefficients for a frame."""
    if pre_emphasis > 0:
        emphasized = np.zeros_like(frame)
        emphasized[0] = frame[0]
        emphasized[1:] = frame[1:] - pre_emphasis * frame[:-1]
    else:
        emphasized = frame
    
    r = autocorrelation(emphasized, order)
    
    if r[0] < 1e-10:
        return LPCResult(
            coefficients=np.zeros(order + 1),
            error=0.0,
            gain=0.0
        )
    
    coeffs, error = levinson_durbin(r, order)
    gain = np.sqrt(max(error, 0))
    
    return LPCResult(
        coefficients=coeffs,
        error=error,
        gain=gain
    )


def lpc_to_spectrum(lpc_coeffs: np.ndarray, fft_size: int, gain: float = 1.0) -> np.ndarray:
    """Convert LPC coefficients to spectral envelope magnitude."""
    n_bins = fft_size // 2 + 1
    
    freqs = np.linspace(0, np.pi, n_bins)
    A = np.zeros(n_bins, dtype=complex)
    
    for k, a_k in enumerate(lpc_coeffs):
        A += a_k * np.exp(-1j * k * freqs)
    
    A_mag = np.abs(A)
    A_mag[A_mag < 1e-10] = 1e-10
    
    envelope = gain / A_mag
    return envelope.astype(np.float32)


def lpc_filter(
    excitation: np.ndarray,
    lpc_coeffs: np.ndarray,
    gain: float = 1.0,
    state: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply LPC synthesis filter to excitation signal."""
    order = len(lpc_coeffs) - 1
    
    if state is None:
        state = np.zeros(order, dtype=np.float32)
    
    b = np.array([gain], dtype=np.float32)
    a = lpc_coeffs.astype(np.float32)
    
    zi = signal.lfilter_zi(b, a) * state[0] if len(state) > 0 and state[0] != 0 else np.zeros(max(len(a), len(b)) - 1)
    
    output, zf = signal.lfilter(b, a, excitation, zi=zi)
    
    new_state = np.zeros(order, dtype=np.float32)
    new_state[:min(order, len(zf))] = zf[:min(order, len(zf))]
    
    return output.astype(np.float32), new_state


def lpc_inverse_filter(
    signal_in: np.ndarray,
    lpc_coeffs: np.ndarray,
    state: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply LPC inverse filter to get residual."""
    order = len(lpc_coeffs) - 1
    
    if state is None:
        state = np.zeros(order, dtype=np.float32)
    
    b = lpc_coeffs.astype(np.float32)
    a = np.array([1.0], dtype=np.float32)
    
    zi = signal.lfilter_zi(b, a) * state[0] if len(state) > 0 and state[0] != 0 else np.zeros(len(b) - 1)
    
    residual, zf = signal.lfilter(b, a, signal_in, zi=zi)
    
    new_state = np.zeros(order, dtype=np.float32)
    new_state[:min(order, len(zf))] = zf[:min(order, len(zf))]
    
    return residual.astype(np.float32), new_state
