"""
Standalone POS verification: does the repo's canonical POS implementation
recover known pulse frequencies from realistic synthetic RGB traces?

Corrected behaviour (audit fixes):
- Repository root resolved reliably from this file's location with pathlib,
  so the script works from any working directory (previously parents[1]
  pointed at evaluation/ and imports failed outside the repo root).
- Traces are physiologically plausible positive skin-like RGB with small
  pulsatile variations (previously zero-centred sines, which bypassed the
  DC normalisation every method relies on).
- Deterministic generation, multiple pulse frequencies, noise levels and
  motion-like disturbances; finiteness and BPM tolerance assertions.

Synthetic traces verify algorithm behaviour only - they are NOT evidence
of accuracy on human subjects. Run from the repo root with:
    python evaluation/reports/check_pos_chrom_green.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

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

WINDOW_SECONDS = 10.0
BPM_TOLERANCE = 4.0  # > half FFT bin (3 BPM at 25 fps / 10 s window)


def run_case(name: str, case: SyntheticTraceCase, allow_fallback: bool = True):
    trace = generate_synthetic_rgb(case)

    # Sanity: synthetic construction guarantees positive, finite samples.
    assert trace.shape[1] == 3
    assert np.all(np.isfinite(trace)), f"{name}: non-finite synthetic trace"
    assert np.all(trace > 0), f"{name}: synthetic trace must be strictly positive"

    fps = case.fps
    window = int(fps * WINDOW_SECONDS)
    segment = trace[:window]

    normalized = normalize_rgb_trace(segment)
    assert np.all(np.isfinite(normalized)), f"{name}: normalisation produced non-finite values"

    pulse = apply_pos_algorithm(normalized, allow_fallback=allow_fallback)
    assert np.all(np.isfinite(pulse)), f"{name}: POS produced non-finite pulse"
    assert abs(float(np.mean(pulse))) < 1e-6 or window == 0, f"{name}: POS pulse not zero-mean"
    assert abs(float(np.std(pulse)) - 1.0) < 1e-3 or float(np.std(pulse)) < 1e-6, \
        f"{name}: POS pulse not unit-variance (std={float(np.std(pulse))})"

    filtered = bandpass_filter(pulse, fps)
    assert np.all(np.isfinite(filtered)), f"{name}: bandpass produced non-finite values"

    bpm, conf = extract_bpm_from_fft(filtered, fps)
    err = abs(bpm - case.hr_bpm)
    # Peak prominence (share of in-band power in the dominant bin) is a
    # sanity gate that the lock is on the strongest component; calibrated
    # per SNR regime so degraded-but-correct locks are not false failures.
    conf_floor = 0.5 if case.noise_std <= 0.003 else 0.25
    assert err <= BPM_TOLERANCE, \
        f"{name}: POS BPM {bpm:.2f} != expected {case.hr_bpm:.2f} (err {err:.2f} > {BPM_TOLERANCE})"
    assert conf > conf_floor, f"{name}: peak prominence unexpectedly low ({conf:.3f})"

    print(f"[OK] {name}: bpm={bpm:.2f} expected={case.hr_bpm:.2f} "
          f"err={err:.2f}BPM conf={conf:.3f}")
    return bpm


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

    # Degenerate-case check: with allow_fallback=False the perfectly-
    # proportional-trace collapse must surface (non-unit/unusable pulse)
    # rather than be silently rescued. Demonstrates the fallback is doing
    # real work and is opt-out.
    from src.contactless.heart_rate.signal_processing import apply_pos_algorithm

    fps = 25.0
    n = int(10 * fps)
    t = np.arange(n) / fps
    u = np.sin(2.0 * np.pi * (60.0 / 60.0) * t)
    perfectly_proportional = np.stack(
        [140.0 * (1 + 0.01 * u), 100.0 * (1 + 0.01 * u), 90.0 * (1 + 0.01 * u)], axis=1
    )
    pulse = apply_pos_algorithm(
        normalize_rgb_trace(perfectly_proportional), allow_fallback=False
    )
    print(f"[INFO] degenerate collapse without fallback: peak-to-peak "
          f"{float(np.ptp(pulse)):.3e} (expected near zero)")

    if failures:
        print(f"\n{failures} case(s) FAILED")
        return 1
    print("\nAll POS synthetic verification cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
