"""
Quantize Age Model to INT8 (Dynamic Quantization)
==================================================
One-shot script: applies onnxruntime dynamic INT8 quantization to the
FP32 ONNX export produced by export_age_onnx.py.

Dynamic (not static/QAT) quantization is used per the Stage 4 plan: static
quantization would require calibration-data infrastructure that this stage
does not build. Dynamic quantization quantizes the model weights (QUInt8)
at load time, which roughly quarters the model file size.

Usage:
    python scripts/optimization/quantize_age_model.py
    python scripts/optimization/quantize_age_model.py --input <fp32.onnx> --output <int8.onnx>
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from onnxruntime.quantization import quantize_dynamic, QuantType

# Default paths (gitignored, present locally after export_age_onnx.py runs)
DEFAULT_INPUT_PATH = PROJECT_ROOT / "models/weights/age_detection/age_model_fp32.onnx"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "models/weights/age_detection/age_model_int8.onnx"

# Named constant - quantization weight type (standard for CNN backbones with
# onnxruntime dynamic quantization)
QUANT_WEIGHT_TYPE = QuantType.QUInt8


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX age model to INT8 (dynamic)")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help=f"Input FP32 ONNX model (default: {DEFAULT_INPUT_PATH.name})"
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output INT8 ONNX model (default: {DEFAULT_OUTPUT_PATH.name})"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: input FP32 ONNX model not found: {input_path}")
        print("       Run scripts/optimization/export_age_onnx.py first.")
        sys.exit(1)

    print(f"Quantizing {input_path.name} -> {output_path.name} (weight_type={QUANT_WEIGHT_TYPE})...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(input_path),
        str(output_path),
        weight_type=QUANT_WEIGHT_TYPE
    )

    fp32_size = file_size_mb(input_path)
    int8_size = file_size_mb(output_path)
    print("-" * 50)
    print(f"  FP32 ONNX: {fp32_size:8.2f} MB  ({input_path.name})")
    print(f"  INT8 ONNX: {int8_size:8.2f} MB  ({output_path.name})")
    print(f"  Reduction: {100.0 * (1.0 - int8_size / fp32_size):.1f}% "
          f"(~4x smaller expected from weight quantization)")
    print("-" * 50)


if __name__ == "__main__":
    main()
