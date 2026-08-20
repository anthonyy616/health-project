"""
Evaluation Framework for Contactless Vital Signs Monitoring
============================================================

Provides tools for rigorous quantitative evaluation of:
- Heart rate estimation (rPPG methods)
- Respiratory rate estimation
- Pupil diameter estimation
- Age estimation
- Full pipeline performance

Usage:
    from evaluation.metrics import compute_mae, compute_rmse, compute_bland_altman
    from evaluation.plots import plot_predicted_vs_ground_truth, plot_bland_altman
    from evaluation.datasets import UBFCRPPGLoader
    from evaluation.reports import generate_report, save_report
"""

from evaluation.metrics import (
    compute_mae,
    compute_rmse,
    compute_pearson_r,
    compute_r2,
    compute_bland_altman,
    compute_percentage_within_threshold,
    compute_failure_rate,
    compute_latency_stats,
    HREvaluationResult,
)
from evaluation.reports import generate_report, save_report, EvaluationReport

__all__ = [
    "compute_mae",
    "compute_rmse",
    "compute_pearson_r",
    "compute_r2",
    "compute_bland_altman",
    "compute_percentage_within_threshold",
    "compute_failure_rate",
    "compute_latency_stats",
    "HREvaluationResult",
    "generate_report",
    "save_report",
    "EvaluationReport",
]
