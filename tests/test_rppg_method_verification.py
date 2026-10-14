"""
Synthetic verification of the POS / CHROM / GREEN rPPG methods.

These tests verify ALGORITHM BEHAVIOUR on deterministic, physiologically
plausible synthetic traces (positive skin-like baselines, small pulsatile
variations, channel-dependent pulsatile strength, motion-like common-mode
and specular disturbances). They are NOT evidence of accuracy on human
subjects and must never be reported as real-world validation.

The diagnostic cases match the standalone scripts
evaluation/reports/check_pos_chrom_green.py and check_chrom_green.py so
running those scripts reproduces exactly what is asserted here.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contactless.heart_rate.signal_processing import (
    normalize_rgb_trace,
    apply_pos_algorithm,
    apply_pos_algorithm_detailed,
    bandpass_filter,
    extract_bpm_from_fft,
    compute_signal_quality,
)

# evaluation/reports is not a package (evaluation.reports resolves to
# evaluation/reports.py), so load the shared helper module by path,
# mirroring how the standalone scripts import it.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "synthetic_rppg", PROJECT_ROOT / "evaluation" / "reports" / "synthetic_rppg.py"
)
synthetic_rppg = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = synthetic_rppg  # register before exec: dataclass resolution needs it
_spec.loader.exec_module(synthetic_rppg)

SyntheticTraceCase = synthetic_rppg.SyntheticTraceCase
generate_synthetic_rgb = synthetic_rppg.generate_synthetic_rgb

FPS = 25.0
WINDOW_SECONDS = 10.0
BPM_TOLERANCE = 4.0  # > half FFT bin (3 BPM at 25 fps / 10 s window)


def _first_window_bpm(case: SyntheticTraceCase, method: str):
    """Run one method on the first analysis window and return (bpm, conf)."""
    trace = generate_synthetic_rgb(case)
    window = int(case.fps * WINDOW_SECONDS)
    segment = trace[:window]

    if method == "pos":
        normalized = normalize_rgb_trace(segment.astype(np.float32))
        pulse, fallback_used = apply_pos_algorithm_detailed(normalized)
        assert np.all(np.isfinite(pulse))
        filtered = bandpass_filter(pulse, case.fps)
        bpm, conf = extract_bpm_from_fft(filtered, case.fps)
        return bpm, conf, fallback_used

    from scipy.signal import butter, filtfilt

    means = np.mean(segment, axis=0)
    norm = np.zeros_like(segment, dtype=np.float64)
    for c in range(3):
        assert abs(means[c]) > 1.0, "synthetic baselines must be non-zero (DC norm engaged)"
        norm[:, c] = (segment[:, c] - means[c]) / means[c]

    if method == "green":
        green = norm[:, 1]
        nyq = case.fps / 2.0
        b, a = butter(4, [0.7 / nyq, 4.0 / nyq], btype="band")
        filtered = filtfilt(b, a, green)
    else:  # chrom
        xs = np.array([[1.0, -1.0, 0.0]])
        xd = np.array([[0.5, 0.5, -1.0]])
        s = (norm @ xs.T)[:, 0]
        d = (norm @ xd.T)[:, 0]
        std_d = np.std(d)
        alpha = np.std(s) / std_d if std_d > 1e-6 else 0.0
        pulse = s - alpha * d
        std_p = np.std(pulse)
        if std_p > 1e-6:
            pulse = (pulse - np.mean(pulse)) / std_p
        nyq = case.fps / 2.0
        b, a = butter(4, [0.7 / nyq, 4.0 / nyq], btype="band")
        filtered = filtfilt(b, a, pulse)

    fft_mag = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), d=1.0 / case.fps)
    mask = (freqs >= 0.7) & (freqs <= 4.0)
    valid_power = fft_mag[mask]
    valid_freqs = freqs[mask]
    idx = int(np.argmax(valid_power))
    total = np.sum(valid_power)
    return valid_freqs[idx] * 60.0, valid_power[idx] / total, False


CASES = [
    SyntheticTraceCase("resting-60bpm", hr_bpm=60.0, seed=1),
    SyntheticTraceCase("slow-50bpm", hr_bpm=50.0, seed=2),
    SyntheticTraceCase("elevated-90bpm", hr_bpm=90.0, seed=3),
    SyntheticTraceCase("fast-130bpm", hr_bpm=130.0, seed=4),
    SyntheticTraceCase("noisy-60bpm", hr_bpm=60.0, noise_std=0.006, seed=5),
    SyntheticTraceCase("motion-80bpm", hr_bpm=80.0, motion=True, seed=6),
    SyntheticTraceCase("lowlight-60bpm",
                       hr_bpm=60.0, pulse_amplitude=0.006, noise_std=0.004, seed=7),
]

POS_CONF_FLOOR = 0.25          # calibrated sanity floor across SNR regimes
GREEN_CONF_FLOOR = 0.15        # single-channel method
CHROM_CONF_FLOOR = 0.05        # chrominance projections spread band energy


class TestSyntheticTraceConstruction:
    """The generator must produce the documented, usable inputs."""

    def test_trace_shape_and_positivity(self):
        case = SyntheticTraceCase("construct", hr_bpm=60.0, seed=11)
        trace = generate_synthetic_rgb(case)
        assert trace.shape == (int(25 * 30), 3)
        assert np.all(np.isfinite(trace))
        assert np.all(trace > 0)
        assert np.all(trace <= 255)

    def test_deterministic(self):
        case = SyntheticTraceCase("determinism", hr_bpm=60.0, seed=3)
        t1 = generate_synthetic_rgb(case)
        t2 = generate_synthetic_rgb(case)
        np.testing.assert_array_equal(t1, t2)

    def test_pulse_frequency_present_in_spectrum(self):
        case = SyntheticTraceCase("spectral", hr_bpm=60.0, seed=12)
        trace = generate_synthetic_rgb(case)
        green = np.asarray(trace, dtype=np.float64)[:, 1]
        green_norm = (green - green.mean()) / green.mean()
        spectrum = np.abs(np.fft.rfft(green_norm)) ** 2
        freqs = np.fft.rfftfreq(len(green), d=1.0 / case.fps)
        target_bin = int(np.argmin(np.abs(freqs - 1.0)))
        noise_floor = np.median(spectrum)
        assert spectrum[target_bin] > 10.0 * noise_floor, (
            "embedded pulse must be spectrally dominant in its bin"
        )


class TestPosMethod:
    """POS must lock onto the embedded pulse across regimes."""

    @pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
    def test_pos_recovers_pulse_frequency(self, case):
        bpm, conf, fallback_used = _first_window_bpm(case, "pos")
        assert abs(bpm - case.hr_bpm) <= BPM_TOLERANCE, (
            f"{case.name}: POS BPM {bpm:.2f} vs {case.hr_bpm:.2f}"
        )
        assert conf > POS_CONF_FLOOR
        assert isinstance(fallback_used, bool)

    def test_pos_output_is_normalized(self):
        case = SyntheticTraceCase("normcheck", hr_bpm=72.0, seed=21)
        trace = generate_synthetic_rgb(case)
        window = int(case.fps * WINDOW_SECONDS)
        pulse = apply_pos_algorithm(
            normalize_rgb_trace(trace[:window].astype(np.float32))
        )
        assert np.all(np.isfinite(pulse))
        assert abs(np.mean(pulse)) < 1e-6
        assert abs(np.std(pulse) - 1.0) < 1e-3

    def test_pos_fallback_engages_on_degenerate_projection_trace(self):
        """Channels whose normalized pulses make S2 anti-proportional to S1
        cancel the balanced POS pulse while S1 retains variance; the
        documented G-B fallback must engage, stay finite and still recover
        the pulse frequency.

        Construction: normalized channel pulses p=(r,g,b)=(1,1,0.5)
        => S1 = g - b = 0.5u, S2 = -2r + g + b = -0.5u, alpha=1 => P = 0
        while std(S1) > 0.
        """
        fps = 25.0
        n = int(FPS * WINDOW_SECONDS)
        t = np.arange(n) / fps
        u = np.sin(2.0 * np.pi * 1.0 * t)
        strengths = np.array([1.0, 1.0, 0.5])
        baselines = np.array([140.0, 100.0, 90.0])
        trace = baselines * (1.0 + 0.01 * strengths[None, :] * u[:, None])
        pulse, fallback_used = apply_pos_algorithm_detailed(
            normalize_rgb_trace(trace.astype(np.float32)), allow_fallback=True
        )
        assert fallback_used is True
        assert np.all(np.isfinite(pulse))
        filtered = bandpass_filter(pulse, fps)
        bpm, _ = extract_bpm_from_fft(filtered, fps)
        assert abs(bpm - 60.0) <= BPM_TOLERANCE

    def test_pos_strict_mode_does_not_fallback(self):
        """allow_fallback=False surfaces the degeneracy instead of hiding it."""
        n = int(FPS * WINDOW_SECONDS)
        t = np.arange(n) / FPS
        u = np.sin(2.0 * np.pi * 1.0 * t)
        strengths = np.array([1.0, 1.0, 0.5])
        baselines = np.array([140.0, 100.0, 90.0])
        trace = baselines * (1.0 + 0.01 * strengths[None, :] * u[:, None])
        pulse, fallback_used = apply_pos_algorithm_detailed(
            normalize_rgb_trace(trace.astype(np.float32)), allow_fallback=False
        )
        assert fallback_used is False
        # The balanced combination collapses to (numerical) zero variance.
        assert float(np.ptp(pulse)) < 1e-4


class TestChromAndGreen:
    @pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
    def test_green_recovers_pulse_frequency(self, case):
        bpm, conf, _ = _first_window_bpm(case, "green")
        assert abs(bpm - case.hr_bpm) <= BPM_TOLERANCE, (
            f"{case.name}: GREEN BPM {bpm:.2f} vs {case.hr_bpm:.2f}"
        )
        assert conf > GREEN_CONF_FLOOR

    @pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
    def test_chrom_recovers_pulse_frequency(self, case):
        bpm, conf, _ = _first_window_bpm(case, "chrom")
        assert abs(bpm - case.hr_bpm) <= BPM_TOLERANCE, (
            f"{case.name}: CHROM BPM {bpm:.2f} vs {case.hr_bpm:.2f}"
        )
        assert conf > CHROM_CONF_FLOOR


class TestMatchedMethodComparison:
    """All three methods must agree on the same inputs and settings."""

    def test_methods_agree_on_clean_trace(self):
        case = SyntheticTraceCase("matched-clean", hr_bpm=75.0, seed=31)
        bpms = {}
        for method in ("pos", "green", "chrom"):
            bpm, conf, _ = _first_window_bpm(case, method)
            assert conf > 0.05
            bpms[method] = bpm
        values = np.array(list(bpms.values()))
        assert np.ptp(values) <= BPM_TOLERANCE + 2.0, f"method spread too large: {bpms}"
        assert all(abs(v - case.hr_bpm) <= BPM_TOLERANCE for v in values)


class TestDegenerateAndEdgeInputs:
    """Guards: invalid inputs are raised or handled, never silently pass."""

    def test_zero_mean_channel_warns_and_survives(self):
        trace = np.tile(np.sin(np.linspace(0, 10, 200)), (3, 1)).T
        normalized = normalize_rgb_trace(trace)
        # near-zero-mean channel is passed through; output still finite
        assert normalized.shape == trace.shape
        assert np.all(np.isfinite(normalized))

    def test_output_pulse_finite_positive_baseline(self):
        case = SyntheticTraceCase("finite", hr_bpm=60.0, motion=True,
                                  noise_std=0.008, seed=41)
        trace = generate_synthetic_rgb(case)
        window = int(case.fps * WINDOW_SECONDS)
        normalized = normalize_rgb_trace(trace[:window].astype(np.float32))
        pulse, _ = apply_pos_algorithm_detailed(normalized)
        assert np.all(np.isfinite(pulse))
        filtered = bandpass_filter(pulse, case.fps)
        bpm, _ = extract_bpm_from_fft(filtered, case.fps)
        assert bpm == 0.0 or (0.7 * 60 <= bpm <= 4.0 * 60)

    def test_signal_quality_finite(self):
        case = SyntheticTraceCase("sqi", hr_bpm=60.0, seed=43)
        trace = generate_synthetic_rgb(case)
        window = int(case.fps * WINDOW_SECONDS)
        q = compute_signal_quality(trace[:window])
        assert np.isfinite(q)
        assert 0.0 <= q <= 1.0

    def test_pipeline_detector_reports_fallback_flag(self):
        """HeartRateResult carries pos_fallback_used (default False)."""
        from src.contactless.heart_rate.rppg import HeartRateResult

        result = HeartRateResult(
            bpm=72.0, confidence=0.5, signal_quality=0.5,
            buffer_fill=1.0, inference_time_ms=1.0, raw_rgb_trace=None,
        )
        assert result.pos_fallback_used is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
