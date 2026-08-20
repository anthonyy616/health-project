"""
Respiration Rate Evaluation Harness
=====================================

Evaluates the optical-flow respiratory rate detector against ground-truth
data (respiratory belt, manual counting, etc.).

Usage:
    python evaluation/evaluate_respiration.py --data-dir data/respiration_gt/
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from evaluation.metrics import (
    compute_mae,
    compute_rmse,
    compute_pearson_r,
    compute_r2,
    compute_bland_altman,
    compute_percentage_within_threshold,
    compute_failure_rate_simple,
    HREvaluationResult,
)
from evaluation.plots import (
    plot_predicted_vs_ground_truth,
    plot_bland_altman,
    plot_error_histogram,
)

logger = logging.getLogger(__name__)


def evaluate_respiration_on_recording(
    video_frames: np.ndarray,
    fps: int,
    gt_bpm: np.ndarray,
    use_face_detector: bool = True,
) -> Dict:
    """
    Evaluate respiration detector on a single recording.

    Args:
        video_frames: (N, H, W, 3) BGR frames
        fps: Video frame rate
        gt_bpm: Ground truth respiratory rate values (per-second)
        use_face_detector: Whether to use FaceDetector for chest ROI derivation

    Returns:
        Dictionary with predicted BPMs, ground truth, and latencies
    """
    from src.contactless.respiration.detect import RespirationDetector
    from src.contactless.face_detection.detect import FaceDetector

    detector = RespirationDetector(fps=fps)
    face_detector = FaceDetector() if use_face_detector else None

    predicted_bpm = []
    latencies = []

    for frame in video_frames:
        # Get face result
        if face_detector is not None:
            face_result = face_detector.detect(frame)
        else:
            # Minimal FaceResult-like object with just a bbox
            from src.contactless.face_detection.detect import FaceResult
            h, w = frame.shape[:2]
            # Rough face bbox (center 40% of frame)
            bbox = (int(w * 0.3), int(h * 0.1), int(w * 0.4), int(h * 0.5))
            face_result = FaceResult(
                detected=True,
                bbox=bbox,
                landmarks=None,
                face_roi=None,
                forehead_roi=None,
            )

        start = time.perf_counter()
        result = detector.add_frame(frame, face_result)
        elapsed_ms = (time.perf_counter() - start) * 1000

        predicted_bpm.append(result.bpm)
        latencies.append(elapsed_ms)

    return {
        "predicted_bpm": np.array([p if p is not None else np.nan for p in predicted_bpm]),
        "ground_truth_bpm": gt_bpm,
        "latencies_ms": np.array(latencies),
    }


def evaluate_respiration_on_dataset(
    data_dir: str = "data/respiration_gt",
    output_dir: str = "results/respiration_evaluation",
    use_face_detector: bool = True,
) -> Dict:
    """
    Evaluate respiration on a dataset directory.

    Expected structure:
        data/respiration_gt/
        ├── manifest.json
        ├── recording1/
        │   ├── video.avi
        │   └── ground_truth.json   {"bpm_values": [12.5, 13.0, ...]}
        └── recording2/
            └── ...

    Returns:
        Evaluation result dictionary
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_path = data_path / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"No manifest.json found at {manifest_path}")
        return {}

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    all_predicted = []
    all_ground_truth = []
    all_latencies = []

    for recording in manifest.get("recordings", []):
        rec_id = recording["id"]
        video_path = data_path / recording["video"]
        gt_path = data_path / recording["ground_truth"]

        if not video_path.exists():
            logger.warning(f"Skipping {rec_id}: video not found")
            continue

        # Load video
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Skipping {rec_id}: cannot open video")
            continue

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            continue

        video_frames = np.array(frames)

        # Load ground truth
        with open(gt_path, "r") as f:
            gt_data = json.load(f)
        gt_bpm = np.array(gt_data["bpm_values"])

        # Evaluate
        result = evaluate_respiration_on_recording(
            video_frames, fps, gt_bpm, use_face_detector=use_face_detector
        )

        # Align (predicted may be longer due to buffer fill)
        n = min(len(result["predicted_bpm"]), len(gt_bpm))
        valid_pred = result["predicted_bpm"][:n]
        valid_gt = result["ground_truth_bpm"][:n]

        # Filter NaN
        valid_mask = ~np.isnan(valid_pred)
        all_predicted.extend(valid_pred[valid_mask].tolist())
        all_ground_truth.extend(valid_gt[valid_mask].tolist())
        all_latencies.extend(result["latencies_ms"][:n][valid_mask].tolist())

    if len(all_predicted) == 0:
        logger.warning("No valid predictions across all recordings")
        return {}

    predicted = np.array(all_predicted)
    ground_truth = np.array(all_ground_truth)
    latencies = np.array(all_latencies)

    # Compute metrics
    mae = compute_mae(predicted, ground_truth)
    rmse = compute_rmse(predicted, ground_truth)
    pearson_r = compute_pearson_r(predicted, ground_truth)
    r2 = compute_r2(predicted, ground_truth)
    ba = compute_bland_altman(predicted, ground_truth)
    pct_2bpm = compute_percentage_within_threshold(predicted, ground_truth, 2.0)
    pct_3bpm = compute_percentage_within_threshold(predicted, ground_truth, 3.0)
    failure_rate = compute_failure_rate_simple(predicted, ground_truth, valid_range=(4.0, 40.0))

    # Report
    report = {
        "mae": mae,
        "rmse": rmse,
        "pearson_r": pearson_r,
        "r2": r2,
        "bland_altman_bias": ba.mean_bias,
        "bland_altman_loa_lower": ba.lower_loa,
        "bland_altman_loa_upper": ba.upper_loa,
        "percentage_within_2bpm": pct_2bpm,
        "percentage_within_3bpm": pct_3bpm,
        "failure_rate": failure_rate,
        "mean_latency_ms": float(np.mean(latencies)),
        "n_valid": len(predicted),
    }

    logger.info(f"\n{'='*50}")
    logger.info("Respiration Rate Evaluation")
    logger.info(f"{'='*50}")
    logger.info(f"  N valid:     {report['n_valid']}")
    logger.info(f"  MAE:         {report['mae']:.2f} breaths/min")
    logger.info(f"  RMSE:        {report['rmse']:.2f} breaths/min")
    logger.info(f"  Pearson r:   {report['pearson_r']:.3f}")
    logger.info(f"  R²:          {report['r2']:.3f}")
    logger.info(f"  Bias:        {report['bland_altman_bias']:.2f} breaths/min")
    logger.info(f"  LoA:         [{report['bland_altman_loa_lower']:.2f}, "
                 f"{report['bland_altman_loa_upper']:.2f}]")
    logger.info(f"  ±2 breaths:  {report['percentage_within_2bpm']:.1%}")
    logger.info(f"  ±3 breaths:  {report['percentage_within_3bpm']:.1%}")
    logger.info(f"  Failure:     {report['failure_rate']:.1%}")
    logger.info(f"  Mean latency:{report['mean_latency_ms']:.1f} ms")

    # Generate plots
    plot_predicted_vs_ground_truth(
        predicted, ground_truth,
        title="Respiration: Predicted vs Ground Truth",
        xlabel="Ground Truth (breaths/min)",
        ylabel="Predicted (breaths/min)",
        save_path=str(output_path / "predicted_vs_gt.png"),
        method_name="respiration",
    )

    plot_bland_altman(
        predicted, ground_truth,
        title="Respiration: Bland-Altman",
        save_path=str(output_path / "bland_altman.png"),
        method_name="respiration",
    )

    plot_error_histogram(
        predicted, ground_truth,
        title="Respiration: Error Distribution",
        save_path=str(output_path / "error_hist.png"),
        method_name="respiration",
    )

    # Save report
    report_path = output_path / "respiration_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport saved to: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Respiration Rate Evaluation Harness")
    parser.add_argument("--data-dir", default="data/respiration_gt")
    parser.add_argument("--output-dir", default="results/respiration_evaluation")
    parser.add_argument("--no-face-detection", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    evaluate_respiration_on_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_face_detector=not args.no_face_detection,
    )


if __name__ == "__main__":
    main()
