"""
Model Artifact Verification and Manifest Builder
================================================
Verifies the age model artifacts actually load and produces a versioned
manifest (SHA-256 hashes, sizes, architecture, load status) that other
tooling and reports reference.

Distinguishes three artifact classes explicitly:
  - VALID: loads and produces finite outputs
  - LFS_UNRESOLVED: git-lfs pointer stub (recovered via `git lfs pull`)
  - INVALID: data present but unusable (corrupt protobuf / wrong arch)

The manifest QUARANTINES artifacts that load but are numerically untrusted
(e.g. the dynamically-quantized INT8 model - see the xfail in
tests/test_age_onnx.py for the measured divergence).

Usage:
    python scripts/verify_model_artifacts.py            # verify + write manifest
    python scripts/verify_model_artifacts.py --json-only
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WEIGHTS_DIR = PROJECT_ROOT / "models/weights/age_detection"
MANIFEST_PATH = WEIGHTS_DIR / "manifest.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("verify_artifacts")


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def lfs_pointer_status(path: Path) -> bool:
    """True when the file is an unresolved git-lfs pointer stub."""
    try:
        head = path.read_bytes()[:200]
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def verify_pytorch_state(path: Path) -> Dict[str, Any]:
    """Load the checkpoint and report architecture compatibility."""
    import torch
    from models.age_detection.efficientnet_age import EfficientNetAge
    from models.age_detection.mobilenet_age import MobileNetV3Age

    out: Dict[str, Any] = {
        "status": "INVALID",
        "architecture": None,
        "error": None,
        "state_dict_keys": 0,
    }
    try:
        state_dict = torch.load(path, map_location="cpu")
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            # checkpoint format (e.g. best_checkpoint.pt)
            state_dict = state_dict["model_state_dict"]
        out["state_dict_keys"] = len(state_dict)
        for cls, name in ((MobileNetV3Age, "mobilenet_v3"), (EfficientNetAge, "efficientnet")):
            try:
                model = cls(pretrained=False)
                model.load_state_dict(state_dict)
                out["architecture"] = name
                out["status"] = "VALID"
                break
            except RuntimeError:
                continue
        if out["status"] != "VALID":
            out["error"] = "checkpoint matches no known architecture"
    except Exception as e:  # noqa: BLE001 - report, don't crash the manifest build
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def verify_onnx(path: Path) -> Dict[str, Any]:
    import onnx
    out: Dict[str, Any] = {"status": "INVALID", "error": None, "io": None}
    try:
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        session_inputs = [i.name for i in model.graph.input]
        session_outputs = [o.name for o in model.graph.output]
        import onnxruntime
        sess = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        rng = np.random.RandomState(0)
        shape = list(sess.get_inputs()[0].shape or [1, 3, 224, 224])
        # Symbolic batch dims (e.g. ['batch_size', 3, 224, 224]) are normal
        # for exported graph models; substitute concrete 1/C/H/W values.
        if len(shape) == 4:
            dims = [v if isinstance(v, int) else d for v, d in zip(shape, (1, 3, 224, 224))]
            x = rng.uniform(0.0, 1.0, tuple(dims)).astype(np.float32)
            y = sess.run(None, {sess.get_inputs()[0].name: x})[0]
            out["io"] = {"inputs": session_inputs, "outputs": session_outputs}
            if np.isfinite(y).all():
                out["status"] = "VALID"
            else:
                out["error"] = "non-finite outputs on reference input"
        else:
            out["status"] = "INVALID"
            out["error"] = f"unexpected input shape contract: {shape}"
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


QUARANTINED = {
    # Dynamic INT8 quantization distorts this regressor - measured in
    # tests/test_age_onnx.py (xfail with evidence) and
    # evaluation/reports analysis. Not for production predictions.
    "age_model_int8.onnx": (
        "QUARANTINED",
        "Dynamic INT8 quantization diverges ~30 age units from FP32 "
        "(corr ~0.1); needs static-with-calibration or QAT export.",
    ),
}


def build_manifest(json_only: bool) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for path in sorted(WEIGHTS_DIR.glob("*")):
        if path.name in ("manifest.json",) or path.is_dir():
            continue
        entry: Dict[str, Any] = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        if lfs_pointer_status(path):
            entry["status"] = "LFS_UNRESOLVED"
            entry["recovery"] = "git lfs pull --include='models/weights/age_detection/*'"
        elif path.suffix == ".pt":
            entry.update(verify_pytorch_state(path))
        elif path.suffix == ".onnx":
            entry.update(verify_onnx(path))
        else:
            entry["status"] = "SKIPPED"
        if path.name in QUARANTINED:
            status, reason = QUARANTINED[path.name]
            entry["status"] = status
            entry["quarantine_reason"] = reason
        entries.append(entry)
        logger.info(f"{path.name}: {entry['status']} ({entry['bytes']} bytes)")

    manifest = {
        "generated_by": "scripts/verify_model_artifacts.py",
        "note": {
            "VALID": "loads and produces finite outputs",
            "QUARANTINED": "loads but numerically untrusted - not for predictions",
            "LFS_UNRESOLVED": "git-lfs pointer stub, run `git lfs pull`",
            "INVALID": "present but unusable (corrupt/arch mismatch)",
            "SKIPPED": "not a weight artifact",
        },
        "entry_count": len(entries),
        "artifacts": entries,
    }
    if not json_only:
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest written: {MANIFEST_PATH}")
    return entries


def main():
    parser = argparse.ArgumentParser(description="Verify model artifacts and build manifest")
    parser.add_argument("--json-only", action="store_true", help="print JSON instead of writing manifest")
    args = parser.parse_args()
    entries = build_manifest(json_only=args.json_only)
    unresolved = [e["file"] for e in entries if e["status"] == "LFS_UNRESOLVED"]
    if unresolved:
        logger.warning(f"LFS pointer stubs remain: {unresolved} - run `git lfs pull`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
