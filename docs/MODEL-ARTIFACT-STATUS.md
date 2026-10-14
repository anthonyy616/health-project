# Model Artifact Status — Age Detection

Last verified: 2026-10-14 on branch `feat/vitals-improvements` from commit `53d275c`.

This doc tracks which files under `models/weights/age_detection/` are actually loadable, which are placeholders, and what the current inference path depends on.

## Currently valid artifact

- `models/weights/age_detection/best_model.pt`
  - Valid PyTorch state dict
  - Architecture family: MobileNetV3-style age model (`src/contactless/age_estimation/estimator.py` uses `model_type="mobilenet"` for this file)
  - This is the only artifact currently usable for age inference in the codebase

## Invalid / placeholder artifacts

The following files are **not** usable as model weights in their current form:

- `models/weights/age_detection/age_model_20260103_000816.pt`
- `models/weights/age_detection/age_model_20260103_132845.pt`
- `models/weights/age_detection/best_checkpoint.pt`
- `models/weights/age_detection/checkpoint_epoch_10.pt`
- `models/weights/age_detection/checkpoint_epoch_20.pt`
- `models/weights/age_detection/age_model_fp32.onnx`
- `models/weights/age_detection/age_model_int8.onnx`

Current status:

- Small files (around 132-133 bytes) that are Git LFS pointer stubs or otherwise not real weights
- `torch.load(..., weights_only=True)` rejects them
- `onnx.checker` / ONNXRuntime cannot load the `.onnx` files as protobuf models

These should be treated as missing artifacts, not as loadable models.

## What this means in practice

- Age inference should only use `best_model.pt` with the matching model class.
- Any code path that tries to load the other `.pt`/`.onnx` files as real weights must fail explicitly rather than silently load a placeholder.
- ONNX export / INT8 quantization evaluation cannot be verified until real exported artifacts exist. Until then, those paths are blocked, not "passing."

## Recovering or replacing artifacts

To restore the other artifacts, the actual sources must be regenerated from authorized training/export scripts, for example:

- `scripts/optimization/export_age_onnx.py`
- `scripts/optimization/quantize_age_model.py`
- `training/age_detection/train.py`

Do not attempt to fake these files or hand-write plausible-looking weights. If a file is missing, the correct behavior is to skip the dependent test or evaluation step with an explicit reason.

## Intended workflow

- Training produces checkpoints / final weights via `training/age_detection/train.py`
- Export produces ONNX from the trained model via the optimization scripts
- Quantization produces the INT8 variant if needed
- The inference wrapper and evaluation harness should select artifacts by provenance, not by filename alone

Until that pipeline has produced real artifacts for a given file, assume the file is unavailable.
