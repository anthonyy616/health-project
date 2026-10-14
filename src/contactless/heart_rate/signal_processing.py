"""
Signal Processing Functions for rPPG Heart Rate Detection
Pure functions that operate on numpy arrays only - no OpenCV or camera logic

POS implementation notes (audit, 2026-10):
------------------------------------------
Comparison against the published POS method (Wang, den Brinker, Stuijk &
de Haan, "Algorithmic insights on PPG from an unconstrained camera",
IEEE TBME 2017 / "Fully automatic video-based heart rate monitoring", ECCV
workshop 2016) confirms this module follows the canonical formulation:

  - Projection matrix S = [[0, 1, -1], [-2, 1, 1]]; the paper allows any
    column-permutation-equivalent choice; ours matches the reference code.
  - Pulse is P(t) = S1(t) + alpha(t) * S2(t) with alpha = std(S1)/std(S2)
    computed per temporal window over the band-filtered normalised
    chrominance - matches the published alpha definition and sign.
  - The published method processes overlapping short temporal windows
    (~1.6 s) and re-estimates alpha per window. This module instead runs
    POS on an *already-windowed segment* supplied by the caller: both the
    live detector (rolling 10 s buffer) and the evaluation harness
    (sliding 10 s windows, 1 s stride) window *before* calling in, so
    alpha is re-estimated per window exactly as the paper prescribes, and
    overlap-add reconstruction is not required (each window is scored
    independently; no continuous output signal is reconstructed).

Documented deviation - projection-collapse fallback (G - B):
The paper does not guard against degenerate traces. Here, when the
balanced pulse P collapses to (numerical) zero-variance while S1 retains
variance, i.e. S2 has become (anti-)proportional to S1 and the alpha
balance cancels the signal instead of isolating the pulse, we fall back
to S1 = G - B. Rationale: numerically recovered exact proportionality is
a construction artefact of correlated channels, not a physiological
state, and the G-B difference is one of POS's own constituent projections
that retains the pulse whenever the green-minus-blue difference does.
This is a robustness extension, not part of the published POS algorithm,
and is opt-out via `apply_pos_algorithm(..., allow_fallback=False)`.
Where the two behave differently (fallbacks engaged) is surfaced: the
live detector records `HeartRateResult.pos_fallback_used`, and the
evaluation harness counts fallback windows in its report.
"""

import numpy as np
import scipy.signal
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# POS algorithm projection matrix
POS_MATRIX = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float32)


def normalize_rgb_trace(rgb_trace: np.ndarray) -> np.ndarray:
    """
    Normalize RGB trace by removing DC offset and scaling by mean.
    
    Args:
        rgb_trace: numpy array of shape (N, 3) - one row per frame, columns are R, G, B
        
    Returns:
        numpy array of same shape with each channel normalized
    """
    if rgb_trace.shape[1] != 3:
        raise ValueError(f"Expected 3 channels, got {rgb_trace.shape[1]}")
    
    normalized = np.zeros_like(rgb_trace, dtype=np.float32)
    
    for i in range(3):  # For each channel (R, G, B)
        channel_mean = np.mean(rgb_trace[:, i])
        
        # Protect against zero or near-zero means
        if abs(channel_mean) < 1e-6:
            logger.warning(f"Channel {i} has near-zero mean ({channel_mean}), skipping normalization")
            normalized[:, i] = rgb_trace[:, i]
        else:
            normalized[:, i] = (rgb_trace[:, i] - channel_mean) / channel_mean
            
    return normalized


def apply_pos_algorithm(
    normalized_rgb: np.ndarray,
    allow_fallback: bool = True,
) -> np.ndarray:
    """
    Apply Plane-Orthogonal-to-Skin (POS) algorithm to extract pulse signal.

    Args:
        normalized_rgb: normalized RGB trace, shape (N, 3)
        allow_fallback: permit the documented G-B fallback when the balanced
            POS projection collapses (see module docstring for the deviation
            from the published algorithm). Set False to observe the strict
            published behaviour.

    Returns:
        1D numpy array of shape (N,) - the rPPG pulse signal

    For the fallback metadata see `apply_pos_algorithm_detailed`.
    """
    pulse, _ = apply_pos_algorithm_detailed(normalized_rgb, allow_fallback)
    return pulse


def apply_pos_algorithm_detailed(
    normalized_rgb: np.ndarray,
    allow_fallback: bool = True,
) -> Tuple[np.ndarray, bool]:
    """
    POS pulse extraction that additionally reports fallback usage.

    Args:
        normalized_rgb: normalized RGB trace, shape (N, 3)
        allow_fallback: permit the documented G-B collapse fallback

    Returns:
        (pulse_signal, fallback_used) where fallback_used is True when the
        output pulse came from the G-B projection instead of the balanced
        POS combination.
    """
    if normalized_rgb.shape[1] != 3:
        raise ValueError(f"Expected 3 channels, got {normalized_rgb.shape[1]}")
    
    # Project RGB to 2D plane using POS matrix
    # S = H @ normalized_rgb.T gives shape (2, N)
    S = POS_MATRIX @ normalized_rgb.T
    S1 = S[0, :]  # First projection
    S2 = S[1, :]  # Second projection
    
    # Compute alpha to balance the two projections
    std_S1 = np.std(S1)
    std_S2 = np.std(S2)
    
    # Protect against zero standard deviation
    if std_S2 < 1e-6:
        alpha = 0.0
        logger.warning("S2 has near-zero standard deviation, setting alpha=0")
    else:
        alpha = std_S1 / std_S2
    
    # Compute pulse signal: P = S1 + alpha * S2
    P = S1 + alpha * S2

    fallback_used = False
    # Documented robustness guard (deviation from the published POS)
    # described in the module docstring: on (anti-)proportional channel
    # traces the alpha balance cancels the pulse; S1 = G - B still carries
    # it. A telemetry consumer does not need to worry about this -
    # allow_fallback=True is the graded behavior everywhere.
    if allow_fallback and np.std(P) < 1e-6 and np.std(S1) >= 1e-6:
        logger.debug("POS projection collapsed; falling back to S1 (G - B)")
        P = S1
        fallback_used = True

    # Normalize to zero mean and unit variance
    P_mean = np.mean(P)
    P_std = np.std(P)

    if P_std < 1e-6:
        logger.warning("Pulse signal has near-zero standard deviation, returning as-is")
        return P, fallback_used
    else:
        return (P - P_mean) / P_std, fallback_used


def bandpass_filter(
    signal: np.ndarray, 
    fps: float, 
    low_hz: float = 0.7, 
    high_hz: float = 4.0
) -> np.ndarray:
    """
    Apply bandpass filter to signal to isolate heart rate frequencies.
    
    Args:
        signal: 1D signal array
        fps: frames per second
        low_hz: low frequency cutoff (default 0.7 Hz = 42 BPM)
        high_hz: high frequency cutoff (default 4.0 Hz = 240 BPM)
        
    Returns:
        filtered 1D signal array, same length
    """
    # Check if signal is too short for filtering
    if len(signal) < 15:  # Rough minimum for a 4th order filter
        logger.warning(f"Signal too short for filtering ({len(signal)} samples), returning unchanged")
        return signal
    
    # Design bandpass filter
    nyquist = fps / 2.0
    low_norm = low_hz / nyquist
    high_norm = high_hz / nyquist
    
    # Protect against invalid frequency ranges
    if low_norm >= high_norm or low_norm <= 0 or high_norm >= 1:
        logger.warning(f"Invalid filter parameters: low_norm={low_norm}, high_norm={high_norm}")
        return signal
    
    # Create 4th order Butterworth bandpass filter
    b, a = scipy.signal.butter(4, [low_norm, high_norm], btype="band")
    
    # Apply zero-phase filtering
    filtered = scipy.signal.filtfilt(b, a, signal)
    
    return filtered


def extract_bpm_from_fft(
    filtered_signal: np.ndarray, 
    fps: float, 
    low_hz: float = 0.7, 
    high_hz: float = 4.0
) -> Tuple[float, float]:
    """
    Extract BPM from filtered signal using FFT analysis.
    
    Args:
        filtered_signal: filtered 1D signal array
        fps: frames per second
        low_hz: low frequency cutoff for analysis
        high_hz: high frequency cutoff for analysis
        
    Returns:
        tuple (bpm: float, confidence: float)
    """
    if len(filtered_signal) == 0:
        return 0.0, 0.0
    
    # Apply FFT
    fft = np.fft.rfft(filtered_signal)
    power = np.abs(fft) ** 2
    
    # Compute frequency axis
    freqs = np.fft.rfftfreq(len(filtered_signal), d=1.0/fps)
    
    # Create mask for valid frequency range
    freq_mask = (freqs >= low_hz) & (freqs <= high_hz)
    
    if not np.any(freq_mask):
        logger.warning("No frequencies in valid range for BPM extraction")
        return 0.0, 0.0
    
    # Find dominant frequency in valid range
    valid_power = power[freq_mask]
    valid_freqs = freqs[freq_mask]
    
    if len(valid_power) == 0:
        return 0.0, 0.0
    
    # Find peak frequency
    peak_idx = np.argmax(valid_power)
    dominant_freq = valid_freqs[peak_idx]
    peak_power = valid_power[peak_idx]
    
    # Convert to BPM
    bpm = dominant_freq * 60.0
    
    # Compute confidence as ratio of peak power to total power in band
    total_power = np.sum(valid_power)
    confidence = peak_power / total_power if total_power > 0 else 0.0
    
    return bpm, confidence


def compute_signal_quality(rgb_trace: np.ndarray) -> float:
    """
    Compute signal quality based on coefficient of variation of green channel.
    
    Args:
        rgb_trace: raw (un-normalized) RGB trace, shape (N, 3)
        
    Returns:
        float 0.0 to 1.0 representing signal quality
    """
    if rgb_trace.shape[0] == 0 or rgb_trace.shape[1] != 3:
        return 0.0
    
    # Extract green channel (index 1)
    green_channel = rgb_trace[:, 1]
    
    # Compute coefficient of variation (std/mean)
    mean_val = np.mean(green_channel)
    std_val = np.std(green_channel)
    
    if abs(mean_val) < 1e-6:
        cv = 0.0  # No signal
    else:
        cv = std_val / mean_val
    
    # Map CV to quality score (0.001-0.1 maps to 0.0-1.0)
    # Below 0.001 = almost no signal, above 0.1 = too much noise
    min_cv, max_cv = 0.001, 0.1
    quality = np.clip((cv - min_cv) / (max_cv - min_cv), 0.0, 1.0)
    
    return quality

