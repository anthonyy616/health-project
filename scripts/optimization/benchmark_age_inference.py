"""
Benchmark Age Model Inference: PyTorch FP32 vs ONNX FP32 vs ONNX INT8
=====================================================================
Answers "was quantization worth it?" by measuring:

1. Latency - single-image (batch size 1, matching real usage) inference
   over a warmup + timed iteration loop, reported as mean and p95 in ms.
2. Accuracy - MAE over the full existing test split, using the same
   formula as training/age_detection/train.py's evaluate_test()
   (mean absolute error over all samples, raw un-clamped predictions).

The accuracy section requires the processed UTKFace test split
(data/processed/utkface/test_manifest.json). If it is missing, accuracy
is skipped with an explicit note - the script still reports latency and
saves the partial JSON, because a missing evaluation dataset is a
reported blocker, not a reason to fabricate numbers.

Usage:
    python scripts/optimization/benchmark_age_inference.py
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.age_detection.efficientnet_age import EfficientNetAge
from models.age_detection.mobilenet_age import MobileNetV3Age
from src.contactless.age_estimation.estimator import AgeEstimator
from src.contactless.age_estimation.processed_dataset import ProcessedUTKFaceDataset

# Default paths
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "models/weights/age_detection/best_model.pt"
DEFAULT_FP32_ONNX = PROJECT_ROOT / "models/weights/age_detection/age_model_fp32.onnx"
DEFAULT_INT8_ONNX = PROJECT_ROOT / "models/weights/age_detection/age_model_int8.onnx"
TEST_MANIFEST = PROJECT_ROOT / "data/processed/utkface/test_manifest.json"
LOG_DIR = PROJECT_ROOT / "logs/optimization"

# Named constants - benchmark parameters
BENCH_WARMUP_ITERS = 50    # discarded iterations (session/model init, caches)
BENCH_TIMED_ITERS = 200    # timed iterations for the latency numbers
ACCURACY_BATCH_SIZE = 32
LATENCY_SEED = 0           # deterministic dummy input across all backends

# See the note on the "<1%" ambiguity below - intentionally NOT gating exit
# status on a threshold Anthony has not picked.


def build_pytorch_model(weights_path: Path, model_type: str) -> torch.nn.Module:
    """Build the raw PyTorch model exactly like production does (via the
    architecture classes + best_model.pt), for a faithful FP32 baseline."""
    if model_type == "efficientnet":
        model = EfficientNetAge(pretrained=False)
    elif model_type == "mobilenet":
        model = MobileNetV3Age(pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    state_dict = torch.load(weights_path, map_location="cpu")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"ERROR: {weights_path.name} does not match model_type={model_type} "
              f"({str(e)[:150]}). Use --model-type matching the weights file.")
        sys.exit(1)
    model.eval()
    return model


def create_onnx_session(onnx_path: Path):
    import onnxruntime
    return onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def run_latency_benchmark(run_fn, warmup_iters: int, timed_iters: int):
    """Time a single-image callable. Returns (mean_ms, p95_ms)."""
    for _ in range(warmup_iters):
        run_fn()

    timings = []
    for _ in range(timed_iters):
        start = time.perf_counter()
        run_fn()
        timings.append((time.perf_counter() - start) * 1000)

    return float(np.mean(timings)), float(np.percentile(timings, 95))


def compute_mae_from_arrays(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    MAE over ALL samples - mirrors evaluate_test() in
    training/age_detection/train.py (mean absolute error of the raw,
    un-clamped predictions). Reused formula, not reimplemented differently.
    """
    return float(np.mean(np.abs(predictions - targets)))


def evaluate_test_mae_pytorch(model: torch.nn.Module, loader: DataLoader) -> float:
    preds, targets = [], []
    with torch.no_grad():
        for images, ages in loader:
            preds.append(model(images).numpy().flatten())
            targets.append(ages.numpy().flatten())
    return compute_mae_from_arrays(np.concatenate(preds), np.concatenate(targets))


def evaluate_test_mae_onnx(session, loader: DataLoader) -> float:
    input_name = session.get_inputs()[0].name
    preds, targets = [], []
    for images, ages in loader:
        preds.append(session.run(None, {input_name: images.numpy()})[0].flatten())
        targets.append(ages.numpy().flatten())
    return compute_mae_from_arrays(np.concatenate(preds), np.concatenate(targets))


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0


def main():
    parser = argparse.ArgumentParser(description="Benchmark age inference: PyTorch vs ONNX FP32 vs ONNX INT8")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS_PATH), help="PyTorch weights (.pt)")
    parser.add_argument("--fp32-onnx", default=str(DEFAULT_FP32_ONNX), help="Exported FP32 ONNX model")
    parser.add_argument("--int8-onnx", default=str(DEFAULT_INT8_ONNX), help="Quantized INT8 ONNX model")
    parser.add_argument("--warmup", type=int, default=BENCH_WARMUP_ITERS, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=BENCH_TIMED_ITERS, help="Timed iterations")
    parser.add_argument("--model-type", default="mobilenet", choices=("efficientnet", "mobilenet"),
                        help="Architecture of the .pt weights (default mobilenet - see export script)")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    fp32_path = Path(args.fp32_onnx)
    int8_path = Path(args.int8_onnx)

    for label, p in [("weights", weights_path), ("FP32 ONNX", fp32_path), ("INT8 ONNX", int8_path)]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)

    # ---- Load all three backends -----------------------------------------
    print("Loading backends...")
    model = build_pytorch_model(weights_path, args.model_type)
    fp32_session = create_onnx_session(fp32_path)
    int8_session = create_onnx_session(int8_path)

    onnx_input_name = fp32_session.get_inputs()[0].name

    # ---- Shared single-image input (deterministic, same for all backends) --
    rng = np.random.RandomState(LATENCY_SEED)
    dummy_image = rng.uniform(0.0, 1.0, (1, 3, *AgeEstimator.INPUT_SIZE)).astype(np.float32)
    dummy_tensor = torch.from_numpy(dummy_image)

    # ---- Latency benchmark ------------------------------------------------
    print(f"\nLatency benchmark: batch size 1, {args.warmup} warmup + {args.iters} timed iterations")
    print("-" * 65)

    def run_pytorch():
        with torch.no_grad():
            model(dummy_tensor)

    def run_fp32():
        fp32_session.run(None, {onnx_input_name: dummy_image})

    def run_int8():
        int8_session.run(None, {onnx_input_name: dummy_image})

    pt_mean, pt_p95 = run_latency_benchmark(run_pytorch, args.warmup, args.iters)
    fp32_mean, fp32_p95 = run_latency_benchmark(run_fp32, args.warmup, args.iters)
    int8_mean, int8_p95 = run_latency_benchmark(run_int8, args.warmup, args.iters)

    latency = {
        "pytorch_fp32": {"mean_ms": pt_mean, "p95_ms": pt_p95},
        "onnx_fp32": {"mean_ms": fp32_mean, "p95_ms": fp32_p95},
        "onnx_int8": {"mean_ms": int8_mean, "p95_ms": int8_p95},
    }

    # ---- Accuracy benchmark (only if the test split exists) ---------------
    accuracy = None
    if TEST_MANIFEST.exists():
        print(f"\nAccuracy benchmark: full test split ({TEST_MANIFEST.parent})")
        print("-" * 65)
        test_dataset = ProcessedUTKFaceDataset(split="test")
        test_loader = DataLoader(test_dataset, batch_size=ACCURACY_BATCH_SIZE, shuffle=False)

        pt_mae = evaluate_test_mae_pytorch(model, test_loader)
        fp32_mae = evaluate_test_mae_onnx(fp32_session, test_loader)
        int8_mae = evaluate_test_mae_onnx(int8_session, test_loader)

        accuracy = {
            "test_samples": len(test_dataset),
            "pytorch_fp32_mae": pt_mae,
            "onnx_fp32_mae": fp32_mae,
            "onnx_int8_mae": int8_mae,
            "int8_vs_pytorch_delta_years": int8_mae - pt_mae,
            "int8_vs_pytorch_delta_pct": 100.0 * (int8_mae - pt_mae) / pt_mae if pt_mae else None,
        }
    else:
        print(f"\nSKIPPED accuracy benchmark: test split not found at {TEST_MANIFEST}")
        print("      data/processed/utkface/ is empty on this machine - reported as a blocker in")
        print("      docs/execution_path.txt. Re-run after the test split is regenerated.")
        print("-" * 65)

    # ---- Summary table -----------------------------------------------------
    print("\n" + "=" * 68)
    print("AGE MODEL INFERENCE BENCHMARK")
    print("=" * 68)
    print(f"{'Backend':<16} | {'Mean Lat (ms)':>13} | {'p95 Lat (ms)':>12} | {'Test MAE (yrs)':>14}")
    print("-" * 68)
    print(f"{'PyTorch FP32':<16} | {pt_mean:>13.2f} | {pt_p95:>12.2f} | "
          f"{(str(round(accuracy['pytorch_fp32_mae'], 2)) if accuracy else 'n/a (data missing)'):>14}")
    print(f"{'ONNX FP32':<16} | {fp32_mean:>13.2f} | {fp32_p95:>12.2f} | "
          f"{(str(round(accuracy['onnx_fp32_mae'], 2)) if accuracy else 'n/a (data missing)'):>14}")
    print(f"{'ONNX INT8':<16} | {int8_mean:>13.2f} | {int8_p95:>12.2f} | "
          f"{(str(round(accuracy['onnx_int8_mae'], 2)) if accuracy else 'n/a (data missing)'):>14}")
    print("=" * 68)

    # ---- On the "<1% accuracy degradation" target from the original plan --
    print("\nNOTE on the original plan's '<1% accuracy degradation' wording: MAE is measured")
    print("in YEARS, not a percentage, so '<1%' is ambiguous. No threshold is enforced here -")
    print("the numbers above are reported plainly for Anthony to make the accept/reject call.")
    if accuracy:
        print(f"      INT8 vs PyTorch baseline: {accuracy['int8_vs_pytorch_delta_years']:+.2f} years "
              f"({accuracy['int8_vs_pytorch_delta_pct']:+.2f}% of baseline MAE).")

    # ---- Save results -------------------------------------------------------
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "model_type": args.model_type,
        "weights": str(weights_path),
        "fp32_onnx": str(fp32_path),
        "int8_onnx": str(int8_path),
        "file_sizes_mb": {
            "pytorch_weights": file_size_mb(weights_path),
            "onnx_fp32": file_size_mb(fp32_path),
            "onnx_int8": file_size_mb(int8_path),
        },
        "benchmark_config": {
            "warmup_iters": args.warmup,
            "timed_iters": args.iters,
            "batch_size_latency": 1,
            "accuracy_batch_size": ACCURACY_BATCH_SIZE,
        },
        "latency": latency,
        "accuracy": accuracy,
        "note_on_1pct_ambiguity": (
            "The original plan's '<1% accuracy degradation' is ambiguous because MAE is "
            "measured in years, not a percentage. Reported raw MAE values instead; no "
            "threshold enforced - accept/reject is Anthony's call."
        ),
    }

    results_path = LOG_DIR / f"onnx_benchmark_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
