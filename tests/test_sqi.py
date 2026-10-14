"""
Tests for the rPPG Signal Quality Index (Phase E).

Covered here in code: thresholds live in SQIConfig, the aggregate score is
interpretable (not assumed calibrated), rejection semantics are explicit,
and behavior degrades loudly rather than generating plausible-looking
scores from unusable windows. Calibration against held-out error data is
an external validated-evaluation requirement.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contactless.heart_rate.sqi import (
    SQIConfig,
    SQIResult,
    compute_sqi,
    spectral_peak_prominence,
    autocorr_periodicity,
    band_snr_db,
    motion_common_mode,
)


def _spec_load():
    # Reuse the deterministic synthetic trace generator from the reports
    # helpers (not a package: loaded by path).
    spec = importlib.util.spec_from_file_location(
        "synthetic_rppg", PROJECT_ROOT / "evaluation" / "reports" / "synthetic_rppg.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


synthetic_rppg = _spec_load()
SyntheticCase = synthetic_rppg.SyntheticTraceCase
generate = synthetic_rppg.generate_synthetic_rgb


class TestSqiGoodWindows:
    def test_clean_pulse_scores_high(self):
        case = SyntheticCase("sqi-clean", hr_bpm=70.0, seed=101)
        trace = generate(case)
        result = compute_sqi(trace, case.fps)
        assert isinstance(result, SQIResult)
        assert result.score > 0.5, f"clean pulse scored {result.score}"
        assert not result.rejected
        assert result.failure_reasons == [] or not result.rejected

    def test_noisy_pulse_scores_lower_than_clean(self):
        clean = compute_sqi(generate(SyntheticCase("c", 70.0, seed=1)), 25.0)
        noisy = compute_sqi(
            generate(SyntheticCase("n", 70.0, noise_std=0.008, seed=2)), 25.0
        )
        assert noisy.score < clean.score

    def test_score_bounded_and_reasons_populated(self):
        trace = generate(SyntheticCase("b", 70.0, motion=True, seed=3))
        result = compute_sqi(trace, 25.0)
        assert 0.0 <= result.score <= 1.0
        assert set(result.features) >= {
            "spectral_peak_prominence", "autocorr_periodicity", "band_snr",
            "motion_common_mode", "illumination", "frame_completeness",
        }


class TestSqiRejection:
    def test_empty_window_rejected(self):
        result = compute_sqi(np.zeros((0, 3)), 25.0)
        assert result.rejected
        assert "empty_window" in result.failure_reasons

    def test_dark_window_rejected(self):
        trace = generate(SyntheticCase("dark", 70.0, seed=4))
        dark = trace * 0.05  # near-black ROI
        result = compute_sqi(dark, 25.0)
        assert result.rejected
        assert "illumination_too_dark" in result.failure_reasons

    def test_dropped_frames_rejected_below_completeness(self):
        trace = generate(SyntheticCase("drop", 70.0, seed=5, duration_s=10.0))
        keep = trace[::3]  # heavily dropped frames
        result = compute_sqi(keep, 25.0, expected_frames=len(trace))
        assert result.rejected
        assert "insufficient_frame_completeness" in result.failure_reasons


class TestSqiFeatures:
    def test_spectral_prominence_locks_pulse_bin(self):
        cfg = SQIConfig()
        case = SyntheticCase("spec", hr_bpm=75.0, seed=6)
        trace = generate(case)
        green = trace[:, 1]
        signal = green - green.mean()
        prominence = spectral_peak_prominence(signal, case.fps, cfg)
        assert prominence > 0.3

    def test_periodicity_positive_for_pulse(self):
        cfg = SQIConfig()
        trace = generate(SyntheticCase("per", hr_bpm=80.0, seed=7))
        green = trace[:, 1]
        value = autocorr_periodicity(green - green.mean(), 25.0, cfg)
        assert value > 0.2

    def test_motion_common_mode_detects_correlated_motion(self):
        t = np.arange(250) / 25.0
        motion = np.sin(2 * np.pi * 0.5 * t)
        base = np.array([140.0, 100.0, 90.0])
        trace = base[None, :] * (1 + 0.05 * motion[:, None])  # pure common-mode
        corr = motion_common_mode(trace)
        assert corr > 0.9

    def test_band_snr_high_for_pulse(self):
        cfg = SQIConfig()
        trace = generate(SyntheticCase("snr", hr_bpm=90.0, seed=8))
        green = trace[:, 1]
        snr = band_snr_db(green - green.mean(), 25.0, cfg)
        assert snr > cfg.min_band_snr_db


class TestSqiConfig:
    def test_configurable_thresholds_change_rejection(self):
        trace = generate(SyntheticCase("cfg", hr_bpm=60.0, noise_std=0.005, seed=9))
        strict = compute_sqi(trace, 25.0, config=SQIConfig(min_peak_prominence=0.9))
        lenient = compute_sqi(trace, 25.0, config=SQIConfig(min_peak_prominence=0.01))
        assert strict.score <= lenient.score

    def test_defaults_are_documented_constants(self):
        cfg = SQIConfig()
        assert cfg.low_hz == 0.7 and cfg.high_hz == 4.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
