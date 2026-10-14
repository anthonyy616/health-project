"""
Deterministic synthetic RGB trace generator for rPPG method verification.

Used by the standalone verification scripts in this directory, the pytest
suite (tests/test_rppg_method_verification.py) and the research notebooks.

IMPORTANT integrity note:
-------------------------
Synthetic traces verify *algorithm behaviour* (can POS/CHROM/GREEN recover a
known embedded frequency under controlled disturbances). They are NOT
evidence of accuracy on human subjects and must never be reported as such.

Physiological plausibility of the construction:
- Positive skin-like RGB baselines (forehead mean values on the order of
  R=140, G=100, B=90 in 8-bit space), never zero-centred, so the DC
  normalisation used by every method actually engages.
- Small pulsatile variations relative to the baseline (sub-2% amplitude:
  real mean-ROI pulse amplitudes are small relative to skin reflectance),
  with channel-specific pulsatile strength (pulse is strongest in G,
  present in R, weaker but non-zero in B) rather than perfectly
  proportional channels.
- Optional motion-like disturbance: a slow common-mode component (head /
  illumination induced baseline shifts) plus a specular-reflection-like
  term with channel-dependent gains and phases, which is what motion
  artifacts actually look like in a mean-ROI trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

# Skin-like forehead baselines (8-bit scale). Positive by construction.
DEFAULT_BASELINE = (140.0, 100.0, 90.0)

# Relative pulsatile strength per channel (R, G, B): the pulse is dominant
# in G, weaker in R and B. Non-proportional on purpose: perfectly
# proportional channels are a known degenerate construction for POS
# (projection cancellation) and NOT physiologically meaningful.
DEFAULT_PULSE_STRENGTH = (0.4, 1.0, 0.6)

RESPIRATION_HZ = 0.3


@dataclass
class SyntheticTraceCase:
    """One named synthetic-data verification scenario."""
    name: str
    hr_bpm: float
    fps: float = 25.0
    duration_s: float = 30.0
    baseline: Tuple[float, float, float] = DEFAULT_BASELINE
    pulse_strength: Tuple[float, float, float] = DEFAULT_PULSE_STRENGTH
    pulse_amplitude: float = 0.012   # fraction of baseline (G channel, full scale)
    noise_std: float = 0.002         # fraction of baseline, per channel
    motion: bool = False             # add motion-like disturbances
    motion_amplitude: float = 0.02   # fraction of baseline
    seed: int = 42


def generate_synthetic_rgb(case: SyntheticTraceCase) -> np.ndarray:
    """Generate a deterministic (case.seed) synthetic mean-ROI RGB trace.

    Returns:
        np.ndarray of shape (N, 3), float64, strictly positive values,
        with a known sinusoidal pulse at case.hr_bpm embedded with
        channel-dependent strength.
    """
    rng = np.random.RandomState(case.seed)

    fps = case.fps
    n = int(case.fps * case.duration_s)
    t = np.arange(n, dtype=np.float64) / fps

    pulse_hz = case.hr_bpm / 60.0
    u = np.sin(2.0 * np.pi * pulse_hz * t)

    baseline = np.asarray(case.baseline, dtype=np.float64)
    strength = np.asarray(case.pulse_strength, dtype=np.float64)

    # Baseline (illumination-adjustable) starts as the skin RGB values.
    trace = np.tile(baseline, (n, 1))

    # Pulsatile component: channel-dependent amplitude relative to baseline.
    pulse_wave = np.sin(2.0 * np.pi * pulse_hz * t)
    pulsatile = case.pulse_amplitude * strength[None, :] * pulse_wave[:, None]
    trace += baseline * pulsatile

    # Respiration-like slow common-mode wave (mirror chest motion /
    # illumination coupling), small amplitude, common to all channels.
    resp = 0.006 * np.sin(2.0 * np.pi * RESPIRATION_HZ * t)
    trace += baseline * resp[:, None]

    if case.motion:
        # Common-mode baseline shift (head position / exposure drift):
        # affects every channel identically with a slow 0.5 Hz component.
        common_mode = case.motion_amplitude * 0.5 * np.sin(2.0 * np.pi * 0.5 * t)
        trace += baseline * common_mode[:, None]
        # Specular-reflection-like term: channel-dependent gains and phases.
        gains = (1.0, 0.75, 0.6)
        phases = (0.0, 0.9, 1.7)
        for c, (g_c, ph) in enumerate(zip(gains, phases)):
            spec = case.motion_amplitude * g_c * np.sin(2.0 * np.pi * 0.5 * t + ph)
            trace[:, c] += baseline[c] * spec

    # Per-channel Gaussian sensor/quantisation noise (deterministic seed).
    trace += baseline * rng.normal(0.0, case.noise_std, size=(n, 3))

    # Numerical safety: skin pixel values are strictly positive.
    trace = np.clip(trace, 1.0, 255.0)
    return trace


def fft_bin_tolerance_bpm(fps: float, window_size: int) -> float:
    """Maximum BPM error from FFT frequency quantisation for one window.

    The dominant-FFT-bin BPM estimate can be off by at most half a frequency
    bin. Any BPM error larger than this indicates the method failed to lock
    onto the embedded pulse, and any tolerance tighter than this risks
    false failures for trivially quantisable reasons.

    The verification tests use a small margin above half a bin because
    spectral leakage and noise can shift where the spectral peak lands.
    """
    resolution_hz = fps / window_size
    return resolution_hz * 60.0 / 2.0
