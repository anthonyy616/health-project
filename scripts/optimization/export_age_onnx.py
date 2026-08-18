"""
Export Age Model to ONNX (FP32)
================================
One-shot script: exports the accepted age model to ONNX FP32 and verifies
the export is numerically faithful to the original PyTorch model.

Similar in spirit to scripts/preprocessing/preprocess_utkface.py - a
run-once script, not a reusable module.

Usage:
    python scripts/optimization/export_age_onnx.py
    python scripts/optimization/export_age_onnx.py --weights <path> --output <path>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contactless.age_estimation.estimator import AgeEstimator
from models.age_detection.efficientnet_age import EfficientNetAge
from models.age_detection.mobilenet_age import MobileNetV3Age


# ---------------------------------------------------------------------------
# IMPORTANT - model type deviation from the stage plan (verified 2026-08-18)
# ---------------------------------------------------------------------------
# The stage plan says "Load EfficientNetAge, load best_model.pt weights"
# (EfficientNet was documented as the accepted best model). The actual
# weights on this machine are MobileNetV3: every checkpoint in
# models/weights/age_detection/ loads into MobileNetV3Age and FAILS to load
# into EfficientNetAge, and the production path in src/main.py
# (run_age_estimation) already uses AgeEstimator(model_type="mobilenet")
# with best_model.pt. No EfficientNet weights exist anywhere on disk.
#
# Therefore the DEFAULT here is "mobilenet" to export what production
# actually runs. The "efficientnet" choice is kept available via --model-type
# for when/if EfficientNet weights are restored.
# ---------------------------------------------------------------------------

# Default paths (gitignored, present locally)
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "models/weights/age_detection/best_model.pt"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "models/weights/age_detection/age_model_fp32.onnx"

# Named constants - no magic numbers
ONNX_OPSET = 13  # Broadly supported; onnx 1.19 / onnxruntime 1.19 both accept it
EXPORT_CHECK_ATOL = 1e-3  # Max allowed |PyTorch - ONNX| output difference on dummy input
MODEL_TYPES = ("efficientnet", "mobilenet")


def build_model(model_type: str, weights_path: Path) -> torch.nn.Module:
    """Build the chosen architecture (pretrained=False - weights come from file)
    and load the trained state dict. Fails loudly on architecture mismatch."""
    if model_type == "efficientnet":
        model = EfficientNetAge(pretrained=False)
    elif model_type == "mobilenet":
        model = MobileNetV3Age(pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {model_type} (choose from {MODEL_TYPES})")

    print(f"Loading weights from {weights_path} (model_type={model_type})...")
    state_dict = torch.load(weights_path, map_location="cpu")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"ERROR: {weights_path.name} does not match model_type={model_type}.")
        print("       Check which architecture the weights belong to (--model-type).")
        print(f"       Details: {str(e)[:200]}")
        sys.exit(1)

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Export age model to ONNX FP32")
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS_PATH),
        help=f"Path to trained weights (default: {DEFAULT_WEIGHTS_PATH.name})"
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output ONNX path (default: {DEFAULT_OUTPUT_PATH.name})"
    )
    parser.add_argument(
        "--model-type",
        default="mobilenet",
        choices=MODEL_TYPES,
        help="Architecture of the weights file (default: mobilenet - see module docstring)"
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    output_path = Path(args.output)

    if not weights_path.exists():
        print(f"ERROR: weights file not found: {weights_path}")
        print("       This is a blocker - export cannot proceed without trained weights.")
        sys.exit(1)

    model = build_model(args.model_type, weights_path)

    # Dummy input shape confirmed against AgeEstimator.INPUT_SIZE (224, 224)
    # rather than hardcoded independently.
    dummy_input = torch.randn(1, 3, *AgeEstimator.INPUT_SIZE)

    print(f"Exporting {args.model_type} model to ONNX FP32: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["age"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "age": {0: "batch_size"}
        },
        opset_version=ONNX_OPSET
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved ONNX model ({size_mb:.2f} MB)")

    # ------------------------------------------------------------------
    # Verification (NOT optional): run the same dummy input through both
    # the PyTorch model and the exported ONNX session and compare outputs.
    # A silently-broken export would corrupt every downstream number, so
    # the script exits non-zero on mismatch.
    # ------------------------------------------------------------------
    print("Verifying export: PyTorch vs ONNX output on dummy input...")
    import onnxruntime

    session = onnxruntime.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    with torch.no_grad():
        pt_output = model(dummy_input).numpy()
    onnx_output = session.run(None, {input_name: dummy_input.numpy()})[0]

    max_diff = float(np.abs(pt_output - onnx_output).max())
    if not np.allclose(pt_output, onnx_output, atol=EXPORT_CHECK_ATOL):
        print(f"FAIL: ONNX export is NOT numerically faithful (max diff {max_diff:.6f} "
              f"> tolerance {EXPORT_CHECK_ATOL}). Do NOT use this model file.")
        sys.exit(1)

    print(f"PASS: ONNX output matches PyTorch within tolerance "
          f"(max diff {max_diff:.2e} <= {EXPORT_CHECK_ATOL})")


if __name__ == "__main__":
    main()
