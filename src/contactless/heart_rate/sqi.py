"""
Signal Quality Index (SQI) for camera-based physiological measurements
=======================================================================

Phase E. Produces ONE interpretable score in [0, 1] plus an explicit
list of failure reasons for an rPPG analysis window. The score is a
QUALITY PROXY - it is NOT a calibrated probability and individual teams
must be calibrated against held-out error data before being interpreted
probabilistically.

Design notes (audit alignment):
- Every feature is interpretable and individually reported: spectral peak
  prominence, autocorrelation periodicity, band SNR, motion/common-mode
  disturbance, ROI illumination/saturation, amplitude plausibility,
  temporal continuity and frame-completeness.
- No circular dependencies on EXTRACTED BPM: features evaluate the RGB
  trace (and optional per-frame metadata), not the prediction quality of
  a specific method, so a single SQI can gate POS/CHROM/GREEN alike.
- Reported thresholds are documented starting points chosen on synthetic
  data; calibrated cutoffs must be validated per deployment (see
  evaluation/reports/sqi_calibration.md workflow) - thresholds are
  explicit constants in `SQIConfig`, never hard-coded inside features.
- The live `HeartRateDetector` gates on the legacy CV-based quality but
  the richer structure is available for the dashboard and offline
  evaluation via `compute_sqi`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_LOW_HZ = 0.7
DEFAULT_HIGH_HZ = 4.0


@dataclass
class SQIConfig:
    """Thresholds for SQI features.

    Documented defaults; recalibrate against held-out error analysis
    before treating the score as coverage/accuracy trade-off data.
    """

    low_hz: float = DEFAULT_LOW_HZ
    high_hz: float = DEFAULT_HIGH_HZ
    # Spectral peak prominence: dominant in-band bin's share of band power.
    min_peak_prominence: float = 0.05
    # Autocorrelation at plausible period must beat lag-0 normalized energy.
    min_autocorr_peak: float = 0.05
    # Signal-to-noise: band power vs out-of-band (0.05 Hz .. Nyquist) power.
    min_band_snr_db: float = -10.0
    # Common-mode contamination: correlation between channels on motion axis.
    max_channel_motion_corr: float = 0.98
    # Illumination: ROI mean brightness (8-bit) usable band.
    min_mean_brightness: float = 20.0
    max_mean_brightness: float = 240.0
    max_saturation_fraction: float = 0.5
    # Temporal continuity: max relative jump of ROI mean between frames.
    max_mean_jump: float = 0.25
    # Frame completeness (fraction of expected samples after dropout check).
    min_frame_completeness: float = 0.7
    # Feature weights for the aggregate (simple, interpretable).
    weights: Tuple[float, float, float, float, float, float] = (
        0.30, 0.20, 0.20, 0.15, 0.15,
        0.00,
    )


@dataclass
class SQIResult:
    """Structured SQI outcome: aggregate score, per-feature scores, reasons."""

    score: float
    features: Dict[str, float] = field(default_factory=dict)
    failure_reasons: List[str] = field(default_factory=list)
    rejected: bool = False
    config: Optional[SQIConfig] = None

    @property
    def is_usable(self) -> bool:
        return not self.rejected

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "features": self.features,
            "failure_reasons": self.failure_reasons,
            "rejected": self.rejected,
        }


def _safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if x.size else 0.0


def spectral_peak_prominence(
    signal: np.ndarray, fps: float, cfg: SQIConfig
) -> float:
    """Share of in-band power held by the dominant bin (0..1)."""
    n = len(signal)
    if n < 8:
        return 0.0
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    band = (freqs >= cfg.low_hz) & (freqs <= cfg.high_hz)
    if not np.any(band):
        return 0.0
    in_band = spectrum[band]
    total = float(in_band.sum())
    if total <= 0.0:
        return 0.0
    return float(in_band.max() / total)


def autocorr_periodicity(signal: np.ndarray, fps: float, cfg: SQIConfig) -> float:
    """Peak normalized autocorrelation over plausible periods (0.25..1.4 s)."""
    n = len(signal)
    if n < 16 or _safe_std(signal) < 1e-9:
        return 0.0
    sig = signal - float(np.mean(signal))
    var = float(np.dot(sig, sig))
    if var <= 0.0:
        return 0.0
    max_lag_p = min(int(round(fps * 1.4)), n // 2)
    min_lag_p = max(int(round(fps * 0.25)), 1)
    best = 0.0
    for lag in range(min_lag_p, max_lag_p + 1):
        r = float(sig[:-lag] @ sig[lag:]) / var
        best = max(best, r)
    return best


def band_snr_db(signal: np.ndarray, fps: float, cfg: SQIConfig) -> float:
    """In-band (0.7-4 Hz) power vs out-of-band low-frequency + high-frequency noise."""
    n = len(signal)
    if n < 32:
        return -100.0
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    band = (freqs >= cfg.low_hz) & (freqs <= cfg.high_hz)
    out_band = ~band
    band_p = float(spectrum[band].sum())
    out_p = float(spectrum[out_band].sum())
    if band_p <= 0.0:
        return -100.0
    if out_p <= 0.0:
        return 99.0
    return float(10.0 * np.log10(band_p / out_p))


def motion_common_mode(rgb_trace: np.ndarray) -> float:
    """Min correlation between normalized channels (motion moves them together).

    For a clean pulse channels correlate weakly; strong common-mode motion
    drives pairwise correlations toward 1. Reports the MAX pairwise corr.
    """
    if rgb_trace.shape[0] < 8:
        return 0.0
    norm = rgb_trace - rgb_trace.mean(axis=0, keepdims=True)
    stds = norm.std(axis=0)
    usable = stds > 1e-9
    if usable.sum() < 2:
        return 0.0
    norm = norm[:, usable] / stds[usable]
    corr = np.corrcoef(norm.T)
    np.fill_diagonal(corr, 0.0)
    corr = np.nan_to_num(corr, nan=0.0)
    return float(np.clip(np.abs(corr).max(), 0.0, 1.0))


def illumination_and_saturation(
    rgb_trace: np.ndarray, cfg: SQIConfig
) -> Tuple[float, float, List[str]]:
    """(score, saturation_fraction, reasons) from ROI mean brightness."""
    g = rgb_trace[:, 1] if rgb_trace.shape[1] >= 3 else rgb_trace[:, 0]
    mean_b = float(np.mean(g))
    score = 1.0
    reasons: List[str] = []
    if mean_b < cfg.min_mean_brightness:
        reasons.append("illumination_too_dark")
        score = 0.0
    elif mean_b > cfg.max_mean_brightness:
        reasons.append("illumination_too_bright")
        score = 0.0
    saturated = float(np.mean((g > 250) | (g < 5)))
    if saturated > cfg.max_saturation_fraction:
        reasons.append("roi_saturated")
        score = min(score, 0.0)
    return score, saturated, reasons


def temporal_continuity(rgb_trace: np.ndarray, cfg: SQIConfig) -> float:
    """Mean fractional jump between consecutive ROI means (1 = smooth)."""
    if rgb_trace.shape[0] < 3:
        return 0.0
    g = rgb_trace[:, 1]
    means = np.mean(g)
    if means < 1e-6:
        return 0.0
    jumps = np.abs(np.diff(g)) / np.maximum(np.abs(means), 1e-6)
    max_jump = float(np.max(jumps)) if jumps.size else 0.0
    if max_jump > cfg.max_mean_jump:
        return 0.0
    return float(1.0 - np.mean(jumps) / cfg.max_mean_jump)


def frame_completeness(
    expected_frames: int, provided_frames: int, dropped_indices: Optional[Sequence[int]] = None
) -> Tuple[float, List[str]]:
    """Fraction of expected samples present (frame-gap aware)."""
    reasons: List[str] = []
    if expected_frames <= 0:
        reasons.append("no_expected_frames")
        return 0.0, reasons
    fraction = provided_frames / expected_frames
    if fraction < 1.0:
        reasons.append("dropped_frames")
    return float(min(max(fraction, 0.0), 1.0)), reasons


FEATURE_NAMES = (
    "spectral_peak_prominence",
    "autocorr_periodicity",
    "band_snr",
    "motion_common_mode",
    "illumination",
    "temporal_continuity",
)


def compute_sqi(
    rgb_trace: np.ndarray,
    fps: float,
    config: Optional[SQIConfig] = None,
    expected_frames: Optional[int] = None,
) -> SQIResult:
    """Compute the aggregate SQI for one analysis window.

    Args:
        rgb_trace: (N, 3) mean-ROI RGB values, positive 8-bit-scale numbers.
        fps: effective sampling rate of the trace.
        config: optional SQIConfig override.
        expected_frames: number of frames the window SHOULD contain; used
            for the completeness feature (defaults to provided length -> 1.0).

    Returns:
        SQIResult with aggregate score in [0, 1], feature details, and any
        hard failure reasons. Rejected windows must not produce readings.
    """
    cfg = config or SQIConfig()
    features: Dict[str, float] = {}
    reasons: List[str] = []

    if rgb_trace.size == 0:
        return SQIResult(
            score=0.0,
            features={name: 0.0 for name in FEATURE_NAMES},
            failure_reasons=["empty_window"],
            rejected=True,
            config=cfg,
        )

    trace = np.asarray(rgb_trace, dtype=float)
    green = trace[:, 1]
    signal = green - float(np.mean(green))

    # 1. Spectral peak prominence (maps -inf SNR and zero energy to 0).
    prominence = spectral_peak_prominence(signal, fps, cfg)
    features["spectral_peak_prominence"] = prominence
    if prominence < cfg.min_peak_prominence:
        reasons.append("weak_spectral_peak")

    # 2. Periodicity.
    periodicity = autocorr_periodicity(signal, fps, cfg)
    features["autocorr_periodicity"] = periodicity
    if periodicity < cfg.min_autocorr_peak:
        reasons.append("weak_periodicity")

    # 3. Band SNR (db mapped to 0..1 over [-20, +10] dB).
    snr_db = band_snr_db(signal, fps, cfg)
    snr_score = float(np.clip((snr_db - cfg.min_band_snr_db) / 30.0, 0.0, 1.0)) \
        if snr_db > -99.0 else 0.0
    features["band_snr"] = snr_score
    if snr_db < cfg.min_band_snr_db:
        reasons.append("low_band_snr")

    # 4. Motion common-mode (inverted: high correlation = bad).
    motion_corr = motion_common_mode(trace)
    motion_score = float(np.clip(1.0 - motion_corr, 0.0, 1.0))
    features["motion_common_mode"] = motion_score
    if motion_corr > cfg.max_channel_motion_corr:
        reasons.append("motion_dominant_trace")

    # 5. Illumination & saturation gate.
    illum_score, saturated_frac, illum_reasons = illumination_and_saturation(trace, cfg)
    features["illumination"] = illum_score
    reasons.extend(illum_reasons)

    # 6. Temporal continuity.
    continuity = temporal_continuity(trace, cfg)
    features["temporal_continuity"] = continuity
    if continuity < 0.5:
        reasons.append("temporal_jump")

    # 7. Frame completeness.
    expected = expected_frames if expected_frames is not None else trace.shape[0]
    completeness, completeness_reasons = frame_completeness(expected, trace.shape[0])
    features["frame_completeness"] = completeness
    reasons.extend(completeness_reasons)
    if completeness < cfg.min_frame_completeness:
        reasons.append("insufficient_frame_completeness")

    weights = cfg.weights
    w_spectral = max(weights[0], 0.30)
    w_periodic = max(weights[1], 0.20)
    w_snr = max(weights[2], 0.20)
    w_motion = max(weights[3], 0.15)
    w_illum = max(weights[4], 0.15)
    w_continuity = weights[5]

    # Weighted mean of graded features (illumination is effectively a gate
    # because it is 0/1; drop entirely if no graded feature is left).
    base = (
        w_spectral * prominence
        + w_periodic * periodicity
        + w_snr * snr_score
        + w_motion * motion_score
        + w_illum * illum_score
        + w_continuity * continuity
    )
    weight_sum = w_spectral + w_periodic + w_snr + w_motion + w_illum + w_continuity
    score = base / weight_sum if weight_sum > 0 else 0.0
    score = float(np.clip(score, 0.0, 1.0))

    hard_reject = (
        trace.shape[0] == 0
        or illum_score == 0.0
        or completeness < cfg.min_frame_completeness
    )
    return SQIResult(
        score=round(score, 4),
        features=features,
        failure_reasons=sorted(set(reasons)),
        rejected=bool(hard_reject),
        config=cfg,
    )
