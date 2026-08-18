#Tests for Stage 4: ONNX Export + INT8 Quantization (Age Model)
#
# DEPENDENCY WARNING: unlike the synthetic rPPG/respiration/pupil suites,
# correctness here is defined as "matches the existing PyTorch model", so
# these tests require the exported ONNX artifacts to exist:
#   models/weights/age_detection/age_model_fp32.onnx
#   models/weights/age_detection/age_model_int8.onnx
# Run scripts/optimization/export_age_onnx.py and quantize_age_model.py
# first. Tests SKIP GRACEFULLY (print a note and pass as no-ops) when the
# artifacts - or the processed test data for the accuracy test - are
# missing, so this file never fails with a confusing error on a clean
# checkout.

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Ensure ✓ checkmarks print on Windows consoles with cp1252 default encoding
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from torch.utils.data import DataLoader

from models.age_detection.efficientnet_age import EfficientNetAge
from models.age_detection.mobilenet_age import MobileNetV3Age
from src.contactless.age_estimation.estimator import AgeEstimator

# Artifact paths (produced by the export/quantize scripts)
ONNX_FP32_PATH = project_root / "models/weights/age_detection/age_model_fp32.onnx"
ONNX_INT8_PATH = project_root / "models/weights/age_detection/age_model_int8.onnx"
MODEL_WEIGHTS_PATH = project_root / "models/weights/age_detection/best_model.pt"
TEST_MANIFEST_PATH = project_root / "data/processed/utkface/test_manifest.json"

# Named constants
EXPORT_MATCH_ATOL = 1e-3            # same tolerance as export_age_onnx.py verification
NUM_MATCH_INPUTS = 5                # random inputs for the export-match test
SPEED_SMOKE_WARMUP = 3              # warmup iterations for the speed smoke test
SPEED_SMOKE_ITERS = 20              # timed iterations for the speed smoke test
ACCURACY_SUBSET_SIZE = 200          # samples for the accuracy sanity test (not the full split)
ACCURACY_BATCH_SIZE = 16
# Generous sanity bound: INT8 MAE must not be wildly worse than the FP32
# baseline. Pick the larger of 2x the FP32 MAE or +3 years absolute. This
# catches a badly broken quantization, NOT a precise accuracy target - that
# judgment belongs to Anthony reviewing the benchmark script output.
ACCURACY_MAX_MULTIPLIER = 2.0
ACCURACY_MAX_ABS_DELTA_YEARS = 3.0
MATCH_TEST_SEED = 1234


def _require(artifact_paths, what):
    """Return True if all artifacts exist; otherwise print a skip note and
    return False so the caller can no-op instead of failing confusingly."""
    missing = [str(p) for p in artifact_paths if not p.exists()]
    if missing:
        print(f"SKIPPED: {what} - required artifact(s) not found: {missing}")
        print("         Run scripts/optimization/export_age_onnx.py and "
              "quantize_age_model.py first.")
        return False
    return True


def load_pytorch_model():
    """Load best_model.pt into whichever architecture it actually matches
    (mobilenet on this machine - see export_age_onnx.py module docstring)."""
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")
    for cls, name in ((EfficientNetAge, "efficientnet"), (MobileNetV3Age, "mobilenet")):
        model = cls(pretrained=False)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            continue
        model.eval()
        return model, name
    raise RuntimeError(
        "best_model.pt matches neither EfficientNetAge nor MobileNetV3Age"
    )


def _onnx_inputs(session):
    """Return (input_name, numpy input) for a batch-1 random image."""
    name = session.get_inputs()[0].name
    rng = np.random.RandomState(MATCH_TEST_SEED)
    x = rng.uniform(0.0, 1.0, (1, 3, *AgeEstimator.INPUT_SIZE)).astype(np.float32)
    return name, x


def test_onnx_export_matches_pytorch():
    """ONNX FP32 output must match the PyTorch model on MULTIPLE random
    inputs - a single lucky match is not sufficient evidence."""
    if not _require([ONNX_FP32_PATH, MODEL_WEIGHTS_PATH], "export-match test"):
        return

    import onnxruntime

    model, model_name = load_pytorch_model()
    session = onnxruntime.InferenceSession(str(ONNX_FP32_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    rng = np.random.RandomState(MATCH_TEST_SEED)
    worst_diff = 0.0
    for i in range(NUM_MATCH_INPUTS):
        x_np = rng.uniform(0.0, 1.0, (1, 3, *AgeEstimator.INPUT_SIZE)).astype(np.float32)
        with torch.no_grad():
            pt_out = model(torch.from_numpy(x_np)).numpy()
        onnx_out = session.run(None, {input_name: x_np})[0]
        diff = float(np.abs(pt_out - onnx_out).max())
        worst_diff = max(worst_diff, diff)
        assert np.allclose(pt_out, onnx_out, atol=EXPORT_MATCH_ATOL), \
            f"Input {i}: ONNX FP32 differs from PyTorch (max diff {diff:.6f} > {EXPORT_MATCH_ATOL})"

    assert worst_diff <= EXPORT_MATCH_ATOL
    print(f"✓ test_onnx_export_matches_pytorch passed "
          f"({NUM_MATCH_INPUTS} inputs, {model_name}, worst diff {worst_diff:.2e})")


def test_onnx_int8_output_shape():
    """Quantized model keeps the I/O contract: batch-1 input -> (1,) output."""
    if not _require([ONNX_INT8_PATH], "INT8 shape test"):
        return

    import onnxruntime

    session = onnxruntime.InferenceSession(str(ONNX_INT8_PATH), providers=["CPUExecutionProvider"])
    input_name, x = _onnx_inputs(session)

    output = session.run(None, {input_name: x})[0]

    assert output.shape == (1,), \
        f"INT8 model output shape {output.shape} != expected (1,) - I/O contract corrupted"
    assert np.isfinite(output).all(), "INT8 output contains non-finite values"

    print("✓ test_onnx_int8_output_shape passed (output shape (1,), finite)")


def test_onnx_int8_faster_than_fp32():
    """Smoke test: INT8 mean latency must beat ONNX FP32. A failure here is
    meaningful information (quantization overhead exceeding benefit on this
    CPU), so the message says exactly that."""
    if not _require([ONNX_INT8_PATH, ONNX_FP32_PATH], "INT8 speed test"):
        return

    import onnxruntime
    import time

    fp32_session = onnxruntime.InferenceSession(str(ONNX_FP32_PATH), providers=["CPUExecutionProvider"])
    int8_session = onnxruntime.InferenceSession(str(ONNX_INT8_PATH), providers=["CPUExecutionProvider"])
    fp32_name, x = _onnx_inputs(fp32_session)
    int8_name = int8_session.get_inputs()[0].name

    def time_session(session, input_name):
        for _ in range(SPEED_SMOKE_WARMUP):
            session.run(None, {input_name: x})
        times = []
        for _ in range(SPEED_SMOKE_ITERS):
            start = time.perf_counter()
            session.run(None, {input_name: x})
            times.append((time.perf_counter() - start) * 1000)
        return float(np.mean(times))

    fp32_mean = time_session(fp32_session, fp32_name)
    int8_mean = time_session(int8_session, int8_name)

    assert int8_mean < fp32_mean, (
        f"INT8 mean latency {int8_mean:.2f}ms is NOT lower than FP32 {fp32_mean:.2f}ms - "
        f"quantization overhead appears to exceed the benefit on this CPU. "
        f"Report this to Anthony; it is a meaningful result, not a test bug."
    )
    print(f"✓ test_onnx_int8_faster_than_fp32 passed "
          f"(FP32 {fp32_mean:.2f}ms -> INT8 {int8_mean:.2f}ms)")


def test_onnx_int8_accuracy_reasonable():
    """Sanity bound on a 200-sample test subset: INT8 MAE must not be wildly
    worse than PyTorch FP32 MAE. Catches a badly broken quantization, not a
    precise target (see ACCURACY_MAX_* constants)."""
    if not _require([ONNX_INT8_PATH, MODEL_WEIGHTS_PATH], "INT8 accuracy test"):
        return
    if not TEST_MANIFEST_PATH.exists():
        print(f"SKIPPED: INT8 accuracy test - test split manifest not found at "
              f"{TEST_MANIFEST_PATH} (data/processed/utkface/ is empty on this machine).")
        print("         Reported as a blocker in docs/execution_path.txt; re-run once "
              "the test split is regenerated.")
        return

    from src.contactless.age_estimation.processed_dataset import ProcessedUTKFaceDataset

    import onnxruntime

    model, _ = load_pytorch_model()
    session = onnxruntime.InferenceSession(str(ONNX_INT8_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    test_dataset = ProcessedUTKFaceDataset(split="test")
    loader = DataLoader(test_dataset, batch_size=ACCURACY_BATCH_SIZE, shuffle=False)

    pt_preds, int8_preds, targets = [], [], []
    with torch.no_grad():
        for images, ages in loader:
            pt_preds.append(model(images).numpy().flatten())
            int8_preds.append(session.run(None, {input_name: images.numpy()})[0].flatten())
            targets.append(ages.numpy().flatten())
            if len(np.concatenate(targets)) >= ACCURACY_SUBSET_SIZE:
                break

    pt_preds = np.concatenate(pt_preds)[:ACCURACY_SUBSET_SIZE]
    int8_preds = np.concatenate(int8_preds)[:ACCURACY_SUBSET_SIZE]
    targets = np.concatenate(targets)[:ACCURACY_SUBSET_SIZE]

    fp32_mae = float(np.mean(np.abs(pt_preds - targets)))
    int8_mae = float(np.mean(np.abs(int8_preds - targets)))

    bound = max(ACCURACY_MAX_MULTIPLIER * fp32_mae, fp32_mae + ACCURACY_MAX_ABS_DELTA_YEARS)
    assert int8_mae <= bound, (
        f"INT8 MAE {int8_mae:.2f} yrs is wildly worse than FP32 MAE {fp32_mae:.2f} yrs "
        f"(bound {bound:.2f}) - quantization appears badly broken."
    )
    print(f"✓ test_onnx_int8_accuracy_reasonable passed "
          f"(FP32 MAE {fp32_mae:.2f} vs INT8 MAE {int8_mae:.2f} on {len(targets)} samples)")


def run_all_tests():
    """Run all tests"""
    print("Running age ONNX tests...")
    print("-" * 30)

    test_onnx_export_matches_pytorch()
    test_onnx_int8_output_shape()
    test_onnx_int8_faster_than_fp32()
    test_onnx_int8_accuracy_reasonable()

    print("-" * 30)
    print("All tests passed! ✅ (any SKIPPED lines above are expected when "
          "ONNX artifacts / test data are missing)")


if __name__ == "__main__":
    run_all_tests()
