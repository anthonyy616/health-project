"""
Tests for the Evaluation Framework
=====================================

Tests metrics computation, plot generation, and dataset loading.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from evaluation.metrics import (
    compute_mae,
    compute_rmse,
    compute_pearson_r,
    compute_r2,
    compute_bland_altman,
    compute_percentage_within_threshold,
    compute_failure_rate,
    compute_failure_rate_simple,
    compute_latency_stats,
    evaluate_hr_method,
)


# ---------------------------------------------------------------------------
# MAE tests
# ---------------------------------------------------------------------------

class TestMAE:
    def test_perfect_prediction(self):
        """Identical arrays should give MAE = 0."""
        arr = np.array([70.0, 75.0, 80.0])
        assert compute_mae(arr, arr) == 0.0

    def test_known_value(self):
        """MAE of [10, 20, 30] vs [12, 18, 35] = mean(2+2+5) = 3.0."""
        predicted = np.array([10.0, 20.0, 30.0])
        ground_truth = np.array([12.0, 18.0, 35.0])
        assert abs(compute_mae(predicted, ground_truth) - 3.0) < 1e-9

    def test_symmetry(self):
        """MAE(a, b) == MAE(b, a)."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert abs(compute_mae(a, b) - compute_mae(b, a)) < 1e-9

    def test_single_element(self):
        assert compute_mae(np.array([5.0]), np.array([8.0])) == 3.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            compute_mae(np.array([1.0]), np.array([1.0, 2.0]))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_mae(np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# RMSE tests
# ---------------------------------------------------------------------------

class TestRMSE:
    def test_perfect_prediction(self):
        arr = np.array([70.0, 75.0, 80.0])
        assert compute_rmse(arr, arr) == 0.0

    def test_known_value(self):
        """RMSE of [0, 0] vs [3, 4] = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355."""
        predicted = np.array([0.0, 0.0])
        ground_truth = np.array([3.0, 4.0])
        expected = np.sqrt(12.5)
        assert abs(compute_rmse(predicted, ground_truth) - expected) < 1e-6

    def test_rmse_geq_mae(self):
        """RMSE is always >= MAE (by Jensen's inequality)."""
        a = np.random.RandomState(42).randn(100) + 70
        b = np.random.RandomState(43).randn(100) + 70
        assert compute_rmse(a, b) >= compute_mae(a, b) - 1e-9


# ---------------------------------------------------------------------------
# Pearson r tests
# ---------------------------------------------------------------------------

class TestPearsonR:
    def test_perfect_positive(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        assert abs(compute_pearson_r(arr, arr) - 1.0) < 1e-9

    def test_perfect_negative(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([4.0, 3.0, 2.0, 1.0])
        assert abs(compute_pearson_r(a, b) - (-1.0)) < 1e-9

    def test_uncorrelated(self):
        """Orthogonal vectors should give r ≈ 0."""
        a = np.array([1.0, -1.0, 1.0, -1.0])
        b = np.array([1.0, 1.0, -1.0, -1.0])
        assert abs(compute_pearson_r(a, b)) < 1e-9

    def test_constant_returns_zero(self):
        """Constant array should return 0 (undefined correlation)."""
        a = np.array([5.0, 5.0, 5.0])
        b = np.array([1.0, 2.0, 3.0])
        assert compute_pearson_r(a, b) == 0.0

    def test_two_samples(self):
        assert abs(compute_pearson_r(np.array([1.0, 2.0]), np.array([3.0, 4.0])) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# R² tests
# ---------------------------------------------------------------------------

class TestR2:
    def test_perfect_prediction(self):
        arr = np.array([70.0, 75.0, 80.0])
        assert abs(compute_r2(arr, arr) - 1.0) < 1e-9

    def test_worse_than_mean(self):
        """Predicting the mean gives R² = 0."""
        gt = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.full_like(gt, np.mean(gt))
        assert abs(compute_r2(pred, gt)) < 1e-9

    def test_negative_r2(self):
        """Predicting inversely gives R² < 0."""
        gt = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([4.0, 3.0, 2.0, 1.0])
        assert compute_r2(pred, gt) < 0


# ---------------------------------------------------------------------------
# Bland-Altman tests
# ---------------------------------------------------------------------------

class TestBlandAltman:
    def test_zero_bias_identical(self):
        a = np.array([70.0, 75.0, 80.0])
        result = compute_bland_altman(a, a)
        assert abs(result.mean_bias) < 1e-9
        assert abs(result.std_difference) < 1e-9

    def test_known_bias(self):
        """If predicted = ground_truth + 5, bias should be 5."""
        gt = np.array([70.0, 75.0, 80.0])
        pred = gt + 5.0
        result = compute_bland_altman(pred, gt)
        assert abs(result.mean_bias - 5.0) < 1e-9

    def test_loa_symmetry(self):
        """LoA should be symmetric around the bias."""
        gt = np.array([70.0, 75.0, 80.0, 85.0, 90.0])
        pred = gt + np.array([0, 2, -1, 3, -2])
        result = compute_bland_altman(pred, gt)
        assert abs(result.lower_loa - (result.mean_bias - 1.96 * result.std_difference)) < 1e-9
        assert abs(result.upper_loa - (result.mean_bias + 1.96 * result.std_difference)) < 1e-9

    def test_returns_arrays_for_plotting(self):
        gt = np.array([70.0, 75.0, 80.0])
        pred = gt + 1.0
        result = compute_bland_altman(pred, gt)
        assert len(result.differences) == 3
        assert len(result.means) == 3


# ---------------------------------------------------------------------------
# Threshold tests
# ---------------------------------------------------------------------------

class TestPercentageWithinThreshold:
    def test_all_within(self):
        a = np.array([70.0, 71.0, 72.0])
        b = np.array([70.5, 71.5, 72.5])
        assert compute_percentage_within_threshold(a, b, 1.0) == 1.0

    def test_none_within(self):
        a = np.array([0.0, 0.0])
        b = np.array([100.0, 100.0])
        assert compute_percentage_within_threshold(a, b, 1.0) == 0.0

    def test_half_within(self):
        a = np.array([0.0, 0.0, 10.0, 10.0])
        b = np.array([0.5, 0.5, 20.0, 20.0])
        assert abs(compute_percentage_within_threshold(a, b, 1.0) - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Failure rate tests
# ---------------------------------------------------------------------------

class TestFailureRate:
    def test_all_valid(self):
        preds = [70.0, 75.0, 80.0]
        gts = [70.0, 75.0, 80.0]
        assert compute_failure_rate(preds, gts) == 0.0

    def test_none_pred_is_failure(self):
        preds = [None, 75.0, 80.0]
        gts = [70.0, 75.0, 80.0]
        assert abs(compute_failure_rate(preds, gts) - 1 / 3) < 1e-9

    def test_out_of_range_is_failure(self):
        preds = [10.0, 75.0, 300.0]  # 10 < 40 min, 300 > 200 max
        gts = [70.0, 75.0, 80.0]
        assert abs(compute_failure_rate(preds, gts) - 2 / 3) < 1e-9

    def test_simple_version(self):
        preds = np.array([10.0, 75.0, 300.0])
        gts = np.array([70.0, 75.0, 80.0])
        assert abs(compute_failure_rate_simple(preds, gts) - 2 / 3) < 1e-9


# ---------------------------------------------------------------------------
# Latency stats tests
# ---------------------------------------------------------------------------

class TestLatencyStats:
    def test_known_values(self):
        latencies = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        stats = compute_latency_stats(latencies)
        assert abs(stats["mean"] - 30.0) < 1e-9
        assert abs(stats["median"] - 30.0) < 1e-9
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0

    def test_empty(self):
        stats = compute_latency_stats(np.array([]))
        assert stats["mean"] == 0.0
        assert stats["p95"] == 0.0


# ---------------------------------------------------------------------------
# Full HR evaluation test
# ---------------------------------------------------------------------------

class TestEvaluateHRMethod:
    def test_perfect_predictions(self):
        gt = np.array([70.0, 75.0, 80.0, 85.0, 90.0])
        result = evaluate_hr_method(gt.copy(), gt, method_name="test")
        assert result.mae < 0.01
        assert result.rmse < 0.01
        assert abs(result.pearson_r - 1.0) < 0.01
        assert result.failure_rate == 0.0
        assert result.percentage_within_5bpm == 1.0
        assert result.percentage_within_10bpm == 1.0

    def test_with_nan_predictions(self):
        """NaN predictions should be treated as failures."""
        gt = np.array([70.0, 75.0, 80.0, 85.0, 90.0])
        pred = np.array([70.0, 75.0, np.nan, 85.0, 90.0])
        result = evaluate_hr_method(pred, gt, method_name="test")
        assert result.n_valid == 4
        assert result.n_total == 5
        assert abs(result.failure_rate - 0.2) < 1e-9

    def test_too_few_valid(self):
        """Less than 2 valid samples should return inf MAE."""
        gt = np.array([70.0, 75.0])
        pred = np.array([np.nan, np.nan])
        result = evaluate_hr_method(pred, gt, method_name="test")
        assert result.mae == float('inf')
