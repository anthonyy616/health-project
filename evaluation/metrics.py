"""
Core Evaluation Metrics
=======================

Implements standard biomedical measurement evaluation metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Pearson correlation coefficient
- R² (coefficient of determination)
- Bland-Altman analysis (bias + limits of agreement)
- Percentage within threshold (±5 BPM, ±10 BPM)
- Failure rate
- Latency statistics

References:
    - Bland & Altman (1986) "Statistical methods for assessing agreement
      between two methods of clinical measurement"
    - ISO 81060-2:2018 "Non-invasive sphygmomanometers — Part 2: Clinical
      investigation of automated measurement type"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class HREvaluationResult:
    """Container for heart rate evaluation results"""
    method: str
    mae: float  # BPM
    rmse: float  # BPM
    pearson_r: float
    r2: float
    bland_altman_bias: float  # BPM
    bland_altman_loa_lower: float  # BPM
    bland_altman_loa_upper: float  # BPM
    percentage_within_5bpm: float  # 0-1
    percentage_within_10bpm: float  # 0-1
    failure_rate: float  # 0-1
    n_valid: int
    n_total: int
    subject_results: List[Dict] = field(default_factory=list)


@dataclass
class BlandAltmanResult:
    """Container for Bland-Altman analysis results"""
    mean_bias: float
    std_difference: float
    lower_loa: float  # bias - 1.96 * std
    upper_loa: float  # bias + 1.96 * std
    mean_predicted: float
    differences: np.ndarray
    means: np.ndarray


def compute_mae(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.
    
    MAE = (1/N) * Σ |predicted_i - ground_truth_i|
    
    Args:
        predicted: Array of predicted values
        ground_truth: Array of ground truth values
    
    Returns:
        MAE value (same units as input)
    
    Raises:
        ValueError: If arrays have different lengths or are empty
    """
    predicted = np.asarray(predicted)
    ground_truth = np.asarray(ground_truth)
    
    if len(predicted) != len(ground_truth):
        raise ValueError(f"Array lengths differ: {len(predicted)} vs {len(ground_truth)}")
    
    if len(predicted) == 0:
        raise ValueError("Cannot compute MAE on empty arrays")
    
    return float(np.mean(np.abs(predicted - ground_truth)))


def compute_rmse(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.
    
    RMSE = sqrt((1/N) * Σ (predicted_i - ground_truth_i)²)
    
    Args:
        predicted: Array of predicted values
        ground_truth: Array of ground truth values
    
    Returns:
        RMSE value (same units as input)
    
    Raises:
        ValueError: If arrays have different lengths or are empty
    """
    predicted = np.asarray(predicted)
    ground_truth = np.asarray(ground_truth)
    
    if len(predicted) != len(ground_truth):
        raise ValueError(f"Array lengths differ: {len(predicted)} vs {len(ground_truth)}")
    
    if len(predicted) == 0:
        raise ValueError("Cannot compute RMSE on empty arrays")
    
    return float(np.sqrt(np.mean((predicted - ground_truth) ** 2)))


def compute_pearson_r(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient.
    
    r = cov(predicted, ground_truth) / (std(predicted) * std(ground_truth))
    
    Args:
        predicted: Array of predicted values
        ground_truth: Array of ground truth values
    
    Returns:
        Pearson r (-1 to 1)
    
    Raises:
        ValueError: If arrays have different lengths or are empty
    """
    predicted = np.asarray(predicted)
    ground_truth = np.asarray(ground_truth)
    
    if len(predicted) != len(ground_truth):
        raise ValueError(f"Array lengths differ: {len(predicted)} vs {len(ground_truth)}")
    
    if len(predicted) < 2:
        raise ValueError("Need at least 2 samples for correlation")
    
    # Handle constant arrays
    if np.std(predicted) == 0 or np.std(ground_truth) == 0:
        logger.warning("One or both arrays are constant; correlation is undefined")
        return 0.0
    
    return float(np.corrcoef(predicted, ground_truth)[0, 1])


def compute_r2(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute R² (coefficient of determination).
    
    R² = 1 - (SS_res / SS_tot)
    
    where SS_res = Σ (predicted_i - ground_truth_i)²
          SS_tot = Σ (ground_truth_i - mean(ground_truth))²
    
    Args:
        predicted: Array of predicted values
        ground_truth: Array of ground truth values
    
    Returns:
        R² value (0 to 1, higher is better)
    
    Raises:
        ValueError: If arrays have different lengths or are empty
    """
    predicted = np.asarray(predicted)
    ground_truth = np.asarray(ground_truth)
    
    if len(predicted) != len(ground_truth):
        raise ValueError(f"Array lengths differ: {len(predicted)} vs {len(ground_truth)}")
    
    if len(predicted) == 0:
        raise ValueError("Cannot compute R² on empty arrays")
    
    ss_res = np.sum((predicted - ground_truth) ** 2)
    ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
    
    if ss_tot == 0:
        logger.warning("Ground truth is constant; R² is undefined")
        return 0.0
    
    return float(1.0 - (ss_res / ss_tot))


def compute_bland_altman(predicted: np.ndarray, ground_truth: np.ndarray) -> BlandAltmanResult:
    """
    Compute Bland-Altman analysis.
    
    The Bland-Altman method evaluates agreement between two measurement methods
    by plotting the difference against the mean and computing:
    - Mean bias (average difference)
    - Limits of agreement (mean ± 1.96 * std of differences)
    
    Args:
        predicted: Array of predicted values (new method)
        ground_truth: Array of ground truth values (reference method)
    
    Returns:
        BlandAltmanResult with bias, LoA, and raw data for plotting
    
    Raises:
        ValueError: If arrays have different lengths or are empty
    """
    predicted = np.asarray(predicted)
    ground_truth = np.asarray(ground_truth)
    
    if len(predicted) != len(ground_truth):
        raise ValueError(f"Array lengths differ: {len(predicted)} vs {len(ground_truth)}")
    
    if len(predicted) < 2:
        raise ValueError("Need at least 2 samples for Bland-Altman analysis")
    
    # Differences and means
    differences = predicted - ground_truth
    means = (predicted + ground_truth) / 2.0
    
    # Statistical measures
    mean_bias = float(np.mean(differences))
    std_difference = float(np.std(differences, ddof=1))  # Sample std
    
    # 95% limits of agreement
    lower_loa = mean_bias - 1.96 * std_difference
    upper_loa = mean_bias + 1.96 * std_difference
    
    return BlandAltmanResult(
        mean_bias=mean_bias,
        std_difference=std_difference,
        lower_loa=lower_loa,
        upper_loa=upper_loa,
        mean_predicted=float(np.mean(means)),
        differences=differences,
        means=means,
    )


def compute_percentage_within_threshold(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float
) -> float:
    """
    Compute percentage of predictions within a threshold of ground truth.
    
    Common thresholds: ±5 BPM, ±10 BPM for heart rate
    
    Args:
        predicted: Array of predicted values
        ground_truth: Array of ground truth values
        threshold: Maximum allowed absolute difference
    
    Returns:
        Percentage within threshold (0-1)
    
    Raises:
        ValueError: If arrays have different lengths or are empty
    """
    predicted = np.asarray(predicted)
    ground_truth = np.asarray(ground_truth)
    
    if len(predicted) != len(ground_truth):
        raise ValueError(f"Array lengths differ: {len(predicted)} vs {len(ground_truth)}")
    
    if len(predicted) == 0:
        raise ValueError("Cannot compute percentage on empty arrays")
    
    within_threshold = np.abs(predicted - ground_truth) <= threshold
    return float(np.mean(within_threshold))


def compute_failure_rate(
    predictions: List[Optional[float]],
    ground_truths: List[float],
    valid_range: Tuple[float, float] = (40.0, 200.0)
) -> float:
    """
    Compute failure rate for heart rate estimation.
    
    A measurement is considered a failure if:
    - prediction is None
    - prediction is outside valid physiological range
    - prediction deviates from ground truth by more than 2x the threshold
    
    Args:
        predictions: List of predicted values (None = failed measurement)
        ground_truths: List of ground truth values
        valid_range: Tuple of (min_valid, max_valid) for physiological range
    
    Returns:
        Failure rate (0-1)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"List lengths differ: {len(predictions)} vs {len(ground_truths)}")
    
    if len(predictions) == 0:
        raise ValueError("Cannot compute failure rate on empty lists")
    
    n_failures = 0
    for pred, gt in zip(predictions, ground_truths):
        # Failure conditions
        if pred is None:
            n_failures += 1
        elif pred < valid_range[0] or pred > valid_range[1]:
            n_failures += 1
        elif abs(pred - gt) > 30.0:  # >30 BPM deviation is clearly wrong
            n_failures += 1
    
    return n_failures / len(predictions)


def compute_failure_rate_simple(
    predictions: np.ndarray,
    ground_truths: np.ndarray,
    valid_range: Tuple[float, float] = (40.0, 200.0)
) -> float:
    """
    Simplified failure rate for numpy arrays (no None values).
    
    Args:
        predictions: Array of predicted values
        ground_truths: Array of ground truth values
        valid_range: Tuple of (min_valid, max_valid)
    
    Returns:
        Failure rate (0-1)
    """
    predictions = np.asarray(predictions)
    ground_truths = np.asarray(ground_truths)
    
    if len(predictions) == 0:
        return 1.0
    
    # Check valid range
    outside_range = (predictions < valid_range[0]) | (predictions > valid_range[1])
    
    # Check large deviation
    large_deviation = np.abs(predictions - ground_truths) > 30.0
    
    failures = outside_range | large_deviation
    return float(np.mean(failures))


def compute_latency_stats(latencies_ms: np.ndarray) -> Dict[str, float]:
    """
    Compute latency statistics.
    
    Args:
        latencies_ms: Array of latency values in milliseconds
    
    Returns:
        Dictionary with mean, median, std, p95, p99, min, max
    """
    latencies_ms = np.asarray(latencies_ms)
    
    if len(latencies_ms) == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    
    return {
        "mean": float(np.mean(latencies_ms)),
        "median": float(np.median(latencies_ms)),
        "std": float(np.std(latencies_ms)),
        "p95": float(np.percentile(latencies_ms, 95)),
        "p99": float(np.percentile(latencies_ms, 99)),
        "min": float(np.min(latencies_ms)),
        "max": float(np.max(latencies_ms)),
    }


def evaluate_hr_method(
    predicted_bpm: np.ndarray,
    ground_truth_bpm: np.ndarray,
    method_name: str = "unknown"
) -> HREvaluationResult:
    """
    Comprehensive heart rate evaluation for a single method.
    
    Args:
        predicted_bpm: Array of predicted heart rates (BPM)
        ground_truth_bpm: Array of ground truth heart rates (BPM)
        method_name: Name of the rPPG method
    
    Returns:
        HREvaluationResult with all metrics
    """
    # Filter out None/NaN values
    valid_mask = ~(np.isnan(predicted_bpm) | np.isnan(ground_truth_bpm))
    predicted_valid = predicted_bpm[valid_mask]
    ground_truth_valid = ground_truth_bpm[valid_mask]
    
    n_total = len(predicted_bpm)
    n_valid = len(predicted_valid)
    failure_rate = 1.0 - (n_valid / n_total) if n_total > 0 else 1.0
    
    if n_valid < 2:
        logger.warning(f"Too few valid samples ({n_valid}) for evaluation")
        return HREvaluationResult(
            method=method_name,
            mae=float('inf'),
            rmse=float('inf'),
            pearson_r=0.0,
            r2=0.0,
            bland_altman_bias=0.0,
            bland_altman_loa_lower=0.0,
            bland_altman_loa_upper=0.0,
            percentage_within_5bpm=0.0,
            percentage_within_10bpm=0.0,
            failure_rate=failure_rate,
            n_valid=n_valid,
            n_total=n_total,
        )
    
    # Compute metrics
    mae = compute_mae(predicted_valid, ground_truth_valid)
    rmse = compute_rmse(predicted_valid, ground_truth_valid)
    pearson_r = compute_pearson_r(predicted_valid, ground_truth_valid)
    r2 = compute_r2(predicted_valid, ground_truth_valid)
    
    # Bland-Altman
    ba = compute_bland_altman(predicted_valid, ground_truth_valid)
    
    # Percentage within thresholds
    pct_5bpm = compute_percentage_within_threshold(predicted_valid, ground_truth_valid, 5.0)
    pct_10bpm = compute_percentage_within_threshold(predicted_valid, ground_truth_valid, 10.0)
    
    return HREvaluationResult(
        method=method_name,
        mae=mae,
        rmse=rmse,
        pearson_r=pearson_r,
        r2=r2,
        bland_altman_bias=ba.mean_bias,
        bland_altman_loa_lower=ba.lower_loa,
        bland_altman_loa_upper=ba.upper_loa,
        percentage_within_5bpm=pct_5bpm,
        percentage_within_10bpm=pct_10bpm,
        failure_rate=failure_rate,
        n_valid=n_valid,
        n_total=n_total,
    )
