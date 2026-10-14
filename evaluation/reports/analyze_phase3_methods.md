# Phase 3 HR-method audit notes

**Date:** 2026-10-14  
**Branch being worked on:** `feat/vitals-improvements`  
**Purpose:** record what was actually checked during the POS / CHROM / GREEN / timing audit,
what was changed, and what still needs follow-up before SQI / lighting / motion eval infra.

## 1. POS robustness guard (committed)

**File:** `src/contactless/heart_rate/signal_processing.py`  
**Change:** `apply_pos_algorithm()` now falls back to `S1` (= `G - B`) when the combined POS
pulse `P = S1 + alpha * S2` collapses to near-zero variance while `S1` still has signal.

Concretely, after computing `P`:

- if `std(P)` is essentially zero **and** `std(S1)` is not, log a debug line and set
  `P = S1` before the usual zero-mean / unit-variance normalization;
- otherwise keep the existing behavior.

This is a backward-compatible robustness improvement, not a rewrite of the POS geometry.
It only changes the output on traces where the original POS path was already producing a
near-dead pulse.

## 2. Verified failure mode for POS (before the guard)

I checked POS with multiple synthetic traces using the existing canonical normalization and
FFT extraction:

- pure green-variation pulse → POS recovers the right BPM;
- balanced RFU-style trace (red, green, blue all modulated with the same 1 Hz component,
  magnitudes roughly: red 0.02, green 0.04, blue 0.01) → POS collapses to a near-zero pulse
  before the guard;
- skin-like trace (green-dominant with smaller red/blue contributions) → POS recovers;
- gray pulse (all three channels modulated equally) → POS collapses before the guard.

CHROM and GREEN did not collapse on the same balanced inputs; they returned sensible BPM / low
confidence where expected. So the issue was specific to the POS combination under balanced /
highly correlated RGB, not to the normalization or FFT stages in general.

## 3. CHROM and GREEN in the evaluation harness

The canonical rPPG primitives live in `src/contactless/heart_rate/signal_processing.py`:

- `normalize_rgb_trace()`
- `apply_pos_algorithm()`
- `apply_chrom_algorithm()`
- `apply_green_algorithm()`
- `bandpass_filter()`
- `extract_bpm_from_fft()`
- `compute_signal_quality()`

The evaluation wrappers in `evaluation/evaluate_hr.py` already operate from the same canonical
signals through `PosExtractor`, and I aligned the docstrings there so POS, CHROM and GREEN are
compared with the same ROI, same RGB trace, same window/stride, and same physiological range.
I did **not** rewrite CHROM or GREEN themselves for this phase; I just made the comparison setup
explicit so the matched-method verification is reproducible.

## 4. Ground-truth and timestamp handling observations

The evaluation loader side already has subject-level structures and time handling worth
preserving; I did not change it in this phase. For future matched-method eval the key points are:

- use the same ground-truth heart-rate series for POS, CHROM and GREEN on the same subject;
- same analysis windows, stride, fps assumption, and physiological band;
- when available, keep actual frame timestamps rather than assuming perfectly uniform sampling;
- when playback drops frames or resets, HR series should not silently treat gaps as continuous
  samples;
- “no face / no ROI / stale measurement” should remain distinct failure modes, not get reported
  as a normal low-BPM reading.

Those are Phase 3/Phase 4 items to formalize in the measurement contract rather than a change
made here.

## 5. SQI direction

The canonical module already has `compute_signal_quality(rgb_trace)` based on green-channel
coefficient of variation. That is the natural place to extend rather than inventing a parallel
metric somewhere else. For Phase 3 the practical next step is to:

- keep `compute_signal_quality()` as the baseline SQI root;
- feed SQI into extraction decisions and report confidence separately from BPM;
- distinguish “low signal quality” from “low confidence because of a weak pulse” so low-quality
  data does not masquerade as a stable low heart rate.

## 6. What I did not change in this phase

- No changes to CHROM / GREEN algorithm internals.
- No changes to respiration, pupil, face-detection, overlay, or pipeline public APIs.
- No changes to evaluation datasets / loaders beyond documentation in this note.
- No synthetic benchmark claims, no fitted metrics, no accuracy assertions.

## 7. Follow-ups before closing Phase 3

- Decide whether to record the POS collapse behavior as a documented limitation in addition to
  the robustness guard, or just let the guard and this note carry it.
- Add a short verification script or notebook cell that repeats the same matched comparison
  (same synthetic RGB trace, same window/stride, POS vs CHROM vs GREEN) so it can be re-run
  after future signal-processing changes.
- Once SQI direction is agreed, hook it into both the live pipeline reporting and the evaluation
  harness so the same quality concept is used in both places.
