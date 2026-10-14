# Model Artifact Status — Age Detection

Last verified: 2026-10-15 on branch `feat/vitals-improvements`.

This doc tracks which files under `models/weights/age_detection/` are actually loadable, which are placeholders, and what the current inference path depends on.

The authoritative, generated inventory is `models/weights/age_detection/manifest.json`
(produced by `scripts/verify_model_artifacts.py`; includes SHA-256 hashes, sizes and
per-file status). Regenerate it after any artifact change:

```
python scripts/verify_model_artifacts.py
```

## Recovery from Git LFS pointer stubs (2026-10-15)

Several listed files previously appeared as 132-byte ASCII text — unresolved
Git LFS pointer stubs, not corrupt weights. On a clean checkout:

```
git lfs install          # once per machine
git lfs pull             # downloads all tracked LFS objects for the checkout
# or selectively:
git lfs pull --include="models/weights/age_detection/*"
```

Verification of identity down to the pointer OID (example: `age_model_fp32.onnx`
pointer declared `sha256:b00a31dfff...`, size 4,384,824) matches the downloaded
binary size, and the files now load and run (manifest status VALID).

## Verified artifact status (includes the two age_model_20260103_* legacy backups)

| File | Status | Notes |
|---|---|---|
| `best_model.pt` | **VALID (mobilenet_v3)** | Production inference weights, ~4.5 MB |
| `best_checkpoint.pt` | **VALID** | Checkpoint format (`model_state_dict` inside) |
| `checkpoint_epoch_10.pt` / `_20.pt` | **VALID** | Checkpoint format |
| `age_model_20260103_132845.pt` | **VALID** | Legacy backup, loads into MobileNetV3Age |
| `age_model_20260103_000816.pt` | INVALID | Matches no known architecture — verify provenance before deleting |
| `age_model_fp32.onnx` | **VALID** | Export verified: outputs match PyTorch within 1e-3 on multiple random inputs (`tests/test_age_onnx.py`) |
| `age_model_int8.onnx` | ⚠️ **QUARANTINED** | See below |

## INT8 quarantine finding (validated 2026-10-15)

Dynamic INT8 quantization (`onnxruntime.quantization.quantize_dynamic`, weight
types QUInt8/QInt8, with and without per-channel) **systematically distorts this
MobileNet regressor**:

- Measured worst FP32→INT8 divergence ~**30 age units** on uniform-noise
  reference inputs (earlier export) and ~30 units on the regenerated export as
  well — reproduced from the validated FP32 file, so it is a property of the
  quantization method on this architecture, not of a stale artifact.
- On naturalistic low-frequency inputs the INT8 output collapses toward a
  near-constant (~75) while FP32 varies 21–53; correlation ~0.1–0.4.
- Endpoint is therefore: **do not use `age_model_int8.onnx` for any prediction**
  until an accurate export exists — static quantization with real calibration
  data, or QAT. The expectation is encoded as a documented xfail in
  `tests/test_age_onnx.py::test_onnx_int8_accuracy_matches_fp32`, which goes
  green automatically when an accurate quantized artifact is produced.
- The FP32 ONNX export pipeline **is** verified (PyTorch parity within 1e-3,
  finite outputs, batch-symbolic input contract).

## Inference-path safety (audit P0)

- `AgeEstimator` no longer predicts from random weights when the checkpoint is
  missing: construction succeeds (for preflight/availability display) but
  `estimate()` raises `RuntimeError` refusing random-weight ages. Callers must
  surface an explicit "age unavailable" state.
- Tests covering the unavailable-weights refusal:
  `tests/test_age_model.py::TestEstimator::test_estimator_refuses_random_weight_inference`.
- Age confidence remains a heuristic (not calibrated); ages stay clamped to
  [0, 100] and remain a demographic estimate, not a vital sign.

## What this means in practice

- Age inference uses `best_model.pt` with the MobileNetV3 class — the
  architecture documentation previously claiming EfficientNet did not match
  the on-disk weights; production (`src/main.py`, `VitalsPipeline`) already
  uses `model_type="mobilenet"`.
- Any code path that tries to load invalid/quarantined files as real weights
  must fail explicitly rather than silently load a placeholder.
- If a file is missing (e.g. bare clone without LFS), the correct behavior is
  to `git lfs pull`, or skip the dependent test with an explicit reason — never
  to fake weights.

## Intended workflow

- Training produces checkpoints / final weights via `training/age_detection/train.py`
  (Colab-capable rerun path: `docs/COLAB-TRAINING.md`).
- Export produces ONNX from the trained model via `scripts/optimization/export_age_onnx.py`.
- Quantization via `scripts/optimization/quantize_age_model.py` — current output is
  quarantined; see the INT8 section before relying on it.
- The inference wrapper and evaluation harness select artifacts by provenance
  (manifest hashes), not by filename alone.

Until that pipeline has produced real, manifest-verified artifacts for a given
file, assume the file is unavailable.
