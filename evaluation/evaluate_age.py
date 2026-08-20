"""
Age Estimation Evaluation Harness
===================================

Evaluates the age estimation model against the UTKFace test split with
overall metrics and per-age-group breakdown.

Usage:
    python evaluation/evaluate_age.py --data-dir data/processed/utkface
    python evaluation/evaluate_age.py --model-type mobilenet --model-path models/weights/age_detection/best_model.pt
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
    compute_latency_stats,
)
from evaluation.plots import (
    plot_predicted_vs_ground_truth,
    plot_bland_altman,
    plot_error_histogram,
    plot_per_subject_mae,
)

logger = logging.getLogger(__name__)

# Age group definitions
AGE_GROUPS = {
    "0-12": (0, 12),
    "13-17": (13, 17),
    "18-29": (18, 29),
    "30-44": (30, 44),
    "45-59": (45, 59),
    "60+": (60, 120),
}


def evaluate_age_on_test_split(
    data_dir: str = "data/processed/utkface",
    model_type: str = "mobilenet",
    model_path: Optional[str] = None,
    output_dir: str = "results/age_evaluation",
    batch_size: int = 32,
) -> Dict:
    """
    Evaluate age estimation on the UTKFace test split.

    Args:
        data_dir: Path to processed UTKFace directory (with train/val/test splits)
        model_type: "mobilenet", "efficientnet", or "lightnet"
        model_path: Path to model weights (None = use default)
        output_dir: Where to save results
        batch_size: Batch size for inference

    Returns:
        Evaluation results dictionary
    """
    import torch
    from torch.utils.data import DataLoader
    import cv2

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    try:
        from src.contactless.age_estimation.estimator import AgeEstimator
        estimator = AgeEstimator(model_type=model_type, model_path=model_path)
    except Exception as e:
        logger.error(f"Failed to load AgeEstimator: {e}")
        return {}

    # Load test dataset
    try:
        from src.contactless.age_estimation.processed_dataset import ProcessedUTKFaceDataset
        test_dataset = ProcessedUTKFaceDataset(split="test")
    except Exception as e:
        logger.error(f"Failed to load test dataset: {e}")
        return {}

    if len(test_dataset) == 0:
        logger.error("Test dataset is empty")
        return {}

    logger.info(f"Evaluating age model ({model_type}) on {len(test_dataset)} test samples")

    # Run inference
    all_predictions = []
    all_ground_truth = []
    all_latencies = []

    for idx in range(len(test_dataset)):
        try:
            image_tensor, age = test_dataset[idx]

            # Convert tensor to numpy BGR for estimator
            # image_tensor is (C, H, W) float [0,1]
            img_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            start = time.perf_counter()
            result = estimator.estimate(img_bgr)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if result is not None:
                all_predictions.append(result.age)
                all_ground_truth.append(age)
                all_latencies.append(elapsed_ms)
        except Exception as e:
            logger.debug(f"Failed on sample {idx}: {e}")
            continue

    if len(all_predictions) == 0:
        logger.error("No successful predictions")
        return {}

    predicted = np.array(all_predictions)
    ground_truth = np.array(all_ground_truth)
    latencies = np.array(all_latencies)

    # Overall metrics
    mae = compute_mae(predicted, ground_truth)
    rmse = compute_rmse(predicted, ground_truth)
    pearson_r = compute_pearson_r(predicted, ground_truth)
    r2 = compute_r2(predicted, ground_truth)
    pct_3yr = compute_percentage_within_threshold(predicted, ground_truth, 3.0)
    pct_5yr = compute_percentage_within_threshold(predicted, ground_truth, 5.0)
    pct_10yr = compute_percentage_within_threshold(predicted, ground_truth, 10.0)
    latency_stats = compute_latency_stats(latencies)

    report = {
        "model_type": model_type,
        "n_samples": len(predicted),
        "overall": {
            "mae": mae,
            "rmse": rmse,
            "pearson_r": pearson_r,
            "r2": r2,
            "percentage_within_3yr": pct_3yr,
            "percentage_within_5yr": pct_5yr,
            "percentage_within_10yr": pct_10yr,
        },
        "latency": latency_stats,
        "per_age_group": {},
    }

    # Per-age-group breakdown
    for group_name, (min_age, max_age) in AGE_GROUPS.items():
        mask = (ground_truth >= min_age) & (ground_truth <= max_age)
        n_group = int(np.sum(mask))
        if n_group < 2:
            report["per_age_group"][group_name] = {"n": n_group, "mae": None}
            continue

        group_pred = predicted[mask]
        group_gt = ground_truth[mask]

        report["per_age_group"][group_name] = {
            "n": n_group,
            "mae": compute_mae(group_pred, group_gt),
            "rmse": compute_rmse(group_pred, group_gt),
            "pct_within_5yr": compute_percentage_within_threshold(group_pred, group_gt, 5.0),
        }

    # Log results
    logger.info(f"\n{'='*50}")
    logger.info(f"Age Estimation Evaluation ({model_type})")
    logger.info(f"{'='*50}")
    logger.info(f"  N samples:   {report['n_samples']}")
    logger.info(f"  MAE:         {mae:.2f} years")
    logger.info(f"  RMSE:        {rmse:.2f} years")
    logger.info(f"  Pearson r:   {pearson_r:.3f}")
    logger.info(f"  R²:          {r2:.3f}")
    logger.info(f"  ±3 years:    {pct_3yr:.1%}")
    logger.info(f"  ±5 years:    {pct_5yr:.1%}")
    logger.info(f"  ±10 years:   {pct_10yr:.1%}")
    logger.info(f"  Latency:     {latency_stats['mean']:.1f} ms (P95: {latency_stats['p95']:.1f} ms)")

    logger.info("\nPer-age-group breakdown:")
    for group_name, group_data in report["per_age_group"].items():
        if group_data["mae"] is not None:
            logger.info(f"  {group_name:>8s}: n={group_data['n']:>5d}  "
                        f"MAE={group_data['mae']:.2f}  "
                        f"±5yr={group_data.get('pct_within_5yr', 0):.1%}")
        else:
            logger.info(f"  {group_name:>8s}: n={group_data['n']:>5d}  (insufficient samples)")

    # Generate plots
    plot_predicted_vs_ground_truth(
        predicted, ground_truth,
        title=f"Age Estimation ({model_type}): Predicted vs True Age",
        xlabel="True Age (years)",
        ylabel="Predicted Age (years)",
        save_path=str(output_path / "predicted_vs_gt.png"),
        method_name=f"age_{model_type}",
    )

    plot_bland_altman(
        predicted, ground_truth,
        title=f"Age Estimation ({model_type}): Bland-Altman",
        xlabel="Mean Age (years)",
        ylabel="Difference (years)",
        save_path=str(output_path / "bland_altman.png"),
        method_name=f"age_{model_type}",
    )

    plot_error_histogram(
        predicted, ground_truth,
        title=f"Age Estimation ({model_type}): Error Distribution",
        xlabel="Error (years)",
        save_path=str(output_path / "error_hist.png"),
        method_name=f"age_{model_type}",
    )

    # Save report
    report_path = output_path / "age_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\nReport saved to: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Age Estimation Evaluation Harness")
    parser.add_argument("--data-dir", default="data/processed/utkface")
    parser.add_argument("--model-type", default="mobilenet",
                        choices=["mobilenet", "efficientnet", "lightnet"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", default="results/age_evaluation")
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    evaluate_age_on_test_split(
        data_dir=args.data_dir,
        model_type=args.model_type,
        model_path=args.model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
