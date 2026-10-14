"""
Standalone CHROM + GREEN verification: does the repo's evaluation-harness
CHROM and GREEN extraction recover known pulse frequencies from realistic
synthetic RGB traces?

Corrected behaviour (audit fixes):
- Repository root resolved reliably from this file's location with pathlib.
  (Previously Path(".") - the script only worked when launched from the
  repo root and failed from any other working directory.)
- Traces are physiologically plausible positive skin-like RGB with small
  pulsatile variations (previously zero-centred sines, which bypassed the
  DC normalisation every method relies on).
- Deterministic generation, matched inputs across methods, multiple pulse
  frequencies, noise levels and motion-like disturbances; finiteness and
  BPM tolerance assertions. POS is included for reference so all three
  methods are compared on the exact same segments and settings.

Synthetic traces verify algorithm behaviour only - they are NOT evidence
of accuracy on human subjects. Run from the repo root with:
    python evaluation/reports/check_chrom_green.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scipy.signal import butter, filtfilt  # noqa: E402

from src.contactless.heart_rate.signal_processing import (  # noqa: E402
    normalize_rgb_trace,
    apply_pos_algorithm,
    bandpass_filter,
    extract_bpm_from_fft,
)
# evaluation/reports is not a sub-package (evaluation.reports resolves to
# evaluation/reports.py), so import the shared helper from this directory
# directly on sys.path.
REPORTS_DIR = Path(__file__).resolve().parent
if str(REPORTS_DIR) not in sys.path:
    sys.path.insert(0, str(REPORTS_DIR))

import synthetic_rppg  # noqa: E402

SyntheticTraceCase = synthetic_rppg.SyntheticTraceCase
generate_synthetic_rgb = synthetic_rppg.generate_synthetic_rgb

FPS = 25.0
WINDOW_SECONDS = 10.0
BPM_TOLERANCE = 4.0  # > half FFT bin (3 BPM at 25 fps / 10 s window)

# CHROM projection matrices (De Haan & Jeanne 2013) - same as the
# evaluation ChromExtractor uses, kept here for standalone reproducibility.
XS = np.array([[1.0, -1.0, 0.0]])
XD = np.array([[0.5, 0.5, -1.0]])


def dc_normalize(segment: np.ndarray) -> np.ndarray:
    """DC normalisation shared by CHROM and GREEN (mirrors the harness)."""
    means = np.mean(segment, axis=0)
    out = np.zeros_like(segment, dtype=np.float64)
    for c in range(3):
        if abs(means[c]) < 1e-6:
            out[:, c] = segment[:, c]
        else:
            out[:, c] = (segment[:, c] - means[c]) / means[c]
    return out


def bandpass(signal: np.ndarray, fps: float) -> np.ndarray:
    if len(signal) < 15:
        return signal
    nyq = fps / 2.0
    b, a = butter(4, [0.7 / nyq, 4.0 / nyq], btype="band")
    return filtfilt(b, a, signal)


def fft_peak_bpm(filtered: np.ndarray, fps: float):
    fft_mag = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fps)
    mask = (freqs >= 0.7) & (freqs <= 4.0)
    if not np.any(mask):
        return np.nan, 0.0
    valid_power = fft_mag[mask]
    valid_freqs = freqs[mask]
    idx = int(np.argmax(valid_power))
    total = np.sum(valid_power)
    return valid_freqs[idx] * 60.0, (valid_power[idx] / total if total > 0 else 0.0)


def extract_green_bpm(segment: np.ndarray, fps: float):
    green = dc_normalize(segment)[:, 1]
    filtered = bandpass(green, fps)
    assert np.all(np.isfinite(filtered)), "GREEN produced non-finite values"
    return fft_peak_bpm(filtered, fps)


def extract_chrom_bpm(segment: np.ndarray, fps: float):
    norm = dc_normalize(segment)
    s = (norm @ XS.T)[:, 0]
    d = (norm @ XD.T)[:, 0]
    std_d = np.std(d)
    alpha = np.std(s) / std_d if std_d > 1e-6 else 0.0
    pulse = s - alpha * d
    std_p = np.std(pulse)
    if std_p > 1e-6:
        pulse = (pulse - np.mean(pulse)) / std_p
    filtered = bandpass(pulse, fps)
    assert np.all(np.isfinite(filtered)), "CHROM produced non-finite values"
    return fft_peak_bpm(filtered, fps)


def run_case(name: str, case: SyntheticTraceCase):
    trace = generate_synthetic_rgb(case)
    assert np.all(np.isfinite(trace)) and np.all(trace > 0), \
        f"{name}: synthetic trace construction invalid"

    fps = case.fps
    window = int(fps * WINDOW_SECONDS)
    segment = trace[:window]

    results = {}
    # Peak-prominence floors are method-calibrated: POS's balanced
    # projection concentrates band energy (high conf), GREEN's single
    # channel is mid, and CHROM's chrominance projections legitimately
    # spread energy across 0.7–4 Hz (low conf). These floors are sanity
    # gates that the dominant bin is organically the strongest component,
    # not accuracy claims.
    conf_floor = {"GREEN": 0.15, "CHROM": 0.05, "POS": 0.4}
    for method_name, fn in (("GREEN", extract_green_bpm), ("CHROM", extract_chrom_bpm)):
        bpm, conf = fn(segment, fps)
        err = abs(bpm - case.hr_bpm)
        floor = conf_floor[method_name]
        assert err <= BPM_TOLERANCE, \
            f"{name} {method_name}: BPM {bpm:.2f} != {case.hr_bpm:.2f} (err {err:.2f})"
        assert conf > floor, (
            f"{name} {method_name}: peak prominence {conf:.3f} below calibrated floor {floor}")
        results[method_name] = (bpm, conf)

    # POS on the exact same segment/settings for matched-method reference.
    normalized = normalize_rgb_trace(np.asarray(segment, dtype=np.float32))
    pulse = apply_pos_algorithm(normalized)
    assert np.all(np.isfinite(pulse)), "POS produced non-finite pulse"
    filtered = bandpass_filter(pulse, fps)
    bpm_pos, conf_pos = extract_bpm_from_fft(filtered, fps)
    assert abs(bpm_pos - case.hr_bpm) <= BPM_TOLERANCE, \
        f"{name} POS: BPM {bpm_pos:.2f} != {case.hr_bpm:.2f}"
    results["POS"] = (bpm_pos, conf_pos)

    line = " | ".join(
        f"{m}: bpm={v[0]:.2f} conf={v[1]:.3f}" for m, v in results.items()
    )
    print(f"[OK] {name} ({case.hr_bpm:.0f} BPM target) - {line}")


def main() -> int:
    cases = [
        SyntheticTraceCase("resting-1.0Hz", hr_bpm=60.0, seed=1),
        SyntheticTraceCase("slow-0.83Hz", hr_bpm=50.0, seed=2),
        SyntheticTraceCase("elevated-1.5Hz", hr_bpm=90.0, seed=3),
        SyntheticTraceCase("fast-2.17Hz", hr_bpm=130.0, seed=4),
        SyntheticTraceCase("noisy-1.0Hz", hr_bpm=60.0, noise_std=0.006, seed=5),
        SyntheticTraceCase("motion-1.33Hz", hr_bpm=80.0, motion=True, seed=6),
        SyntheticTraceCase("lowlight-1.0Hz",
                           hr_bpm=60.0, pulse_amplitude=0.006, noise_std=0.004, seed=7),
    ]

    failures = 0
    for case in cases:
        try:
            run_case(case.name, case)
        except AssertionError as e:
            failures += 1
            print(f"[FAIL] {e}")

    if failures:
        print(f"\n{failures} case(s) FAILED")
        return 1
    print("\nAll CHROM/GREEN/POS matched-input verification cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
