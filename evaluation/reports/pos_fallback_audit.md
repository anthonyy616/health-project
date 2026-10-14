# POS Fallback Audit — projection-collapse handling in `signal_processing.py`

**Date:** 2026-10-15 · **Branch:** `feat/vitals-improvements` · **Status:** Implemented and unit-verified (synthetic data)

## 1. What was audited

`src/contactless/heart_rate/signal_processing.py::apply_pos_algorithm` hardens the
canonical POS implementation with a fallback: when the balanced POS pulse
`P = S1 + α·S2` collapses to (numerical) zero variance while `S1` retains variance, the
output falls back to `S1 = G − B`.

## 2. Comparison against the published POS method

Checked against Wang, den Brinker, Stuijk & de Haan (IEEE TBME 2017 / ECCV-W 2016):

| Aspect | Published POS | This repo | Verdict |
|---|---|---|---|
| Projection matrix | `C_p`-based plane, e.g. `[[0,1,-1], [-2,1,1]]` | identical (`POS_MATRIX`) | ✅ matches |
| Pulse combination | `P = S1 + α·S2` | identical | ✅ matches |
| α definition | `std(S1)/std(S2)`, re-estimated per temporal window | identical, computed per supplied analysis window | ✅ matches |
| Temporal windowing | overlapping short windows (~1.6 s) | caller windows *first*: live detector uses a rolling buffer; evaluation harness uses sliding 10 s/1 s windows — α is re-estimated per window as prescribed | ✅ equivalent-by-construction |
| Overlap-add | only needed if windows are fused into one continuous signal | windows are scored independently (FFT per window); no continuous output is reconstructed | ✅ not applicable |
| Degenerate-trace guard | none | G−B fallback (see below) | ⚠️ deliberate documented deviation |

**Conclusion:** the numerical core of POS matches the published algorithm. No overlap-add defect exists because no window fusion is performed.

## 3. When does the balanced POS pulse truly collapse?

Algebra: `P = 0` requires `S2 ≈ −(1/α)·S1`, i.e. `S2` anti-proportional to `S1` with
ratio `-α` (=−std(S1)/std(S2)). In normalized channel-pulse coordinates `p = (r, g, b)`:

- `S1 = g − b`, `S2 = −2r + g + b`

Exact cancellation is a **rank-deficiency of the input construction**, not a
physiological state: real skin PPG has channel-dependent pulsatile strength
(strongest in G, weaker in B) plus channel- and phase-distinct motion artifacts,
so exact proportionality does not occur on real subjects. What *can* occur is
approximate proportionality during strong common-mode motion, in which case the
α-balanced combination amplifies the residual noise of the non-cancelling part.

## 4. Is falling back to S1 (G − B) scientifically appropriate?

**Yes, conditionally — and it is now disclosed and opt-out.**

- Numerical degeneracy (exact/near-exact proportionality, e.g. synthetic
  perfectly-proportional channels): the collapse is an input-construction
  artefact. `S1 = G − B` is one of POS's own constituent projections and
  retains pulsatility whenever the green-minus-blue difference does; returning
  a near-zero signal would be strictly worse (no measurement at all).
- Genuinely poor physiological signal (very weak pulse, saturated/blank ROI):
  the collapse is real evidence the ROI signal is unusable. Returning G−B
  noise as a "reading" would be dishonest — this is why the live detector
  independently applies the `compute_signal_quality` gate **and** now reports
  `HeartRateResult.pos_fallback_used`, so downstream consumers (SQI, fusion,
  dashboard) know when the fallback engaged and the evaluation harness counts
  fallback windows in its report (`pos_fallback_windows`).

Behaviour decision: **keep the fallback, gated and disclosed** — silent
substitution is what was audited against, not substitution per se. Tests cover
both modes; `apply_pos_algorithm(..., allow_fallback=False)` restores strict
published behaviour.

## 5. Regression coverage added

`tests/test_rppg_method_verification.py`:

- `test_pos_fallback_engages_on_degenerate_projection_trace` — collapse case
  where `S2 = −S1` precisely; asserts fallback engages, output finite, and the
  recovered BPM matches the embedded frequency within FFT half-bin.
- `test_pos_strict_mode_does_not_fallback` — the same input with
  `allow_fallback=False` surfaces the zero-variance collapse instead of
  rescuing it.
- Standard POS method tests on 7 physiologically plausible synthetic regimes
  (rest/low/elevated/fast HR, noise, motion, low light) assert the fallback
  does **not** engage on realistic traces.

## 6. Integrity limits

Synthetic traces verify algorithm behaviour only. No claim is made here about
classification accuracy on human subjects; that requires the UBFC-rPPG or
equivalent real ground-truth evaluation (dataset not available locally — see
`docs/datasets.md`).
