"""
Tests for Evaluation Plots
===========================

Tests all plot generation functions in evaluation/plots.py using
synthetic data and verifying file output.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from evaluation.plots import (
    plot_predicted_vs_ground_truth,
    plot_bland_altman,
    plot_error_histogram,
    plot_confidence_vs_error,
    plot_per_subject_mae,
    plot_latency_distribution,
    plot_method_comparison,
    plot_lighting_conditions,
    plot_distance_conditions,
)


@pytest.fixture
def sample_data():
    """Generate sample prediction/ground-truth arrays."""
    np.random.seed(42)
    gt = np.array([70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0])
    pred = gt + np.random.normal(0, 2, len(gt))
    return pred, gt


@pytest.fixture
def tmp_output(tmp_path):
    """Return a temporary output directory."""
    return str(tmp_path / "plots")


class TestPlotPredictedVsGroundTruth:
    def test_creates_file(self, sample_data, tmp_output):
        pred, gt = sample_data
        save_path = str(Path(tmp_output) / "pvt.png")
        result = plot_predicted_vs_ground_truth(pred, gt, save_path=save_path)
        assert Path(result).exists()

    def test_returns_path(self, sample_data, tmp_output):
        pred, gt = sample_data
        save_path = str(Path(tmp_output) / "pvt2.png")
        result = plot_predicted_vs_ground_truth(pred, gt, save_path=save_path)
        assert isinstance(result, str)

    def test_custom_labels(self, sample_data, tmp_output):
        pred, gt = sample_data
        save_path = str(Path(tmp_output) / "custom.png")
        result = plot_predicted_vs_ground_truth(
            pred, gt,
            title="Custom Title",
            xlabel="True Age",
            ylabel="Predicted Age",
            save_path=save_path,
        )
        assert Path(result).exists()

    def test_no_regression_line(self, sample_data, tmp_output):
        pred, gt = sample_data
        save_path = str(Path(tmp_output) / "no_reg.png")
        result = plot_predicted_vs_ground_truth(
            pred, gt, save_path=save_path, show_regression_line=False
        )
        assert Path(result).exists()

    def test_single_point(self, tmp_output):
        """Single-point plot should not crash."""
        pred = np.array([70.0])
        gt = np.array([70.0])
        save_path = str(Path(tmp_output) / "single.png")
        result = plot_predicted_vs_ground_truth(pred, gt, save_path=save_path)
        assert Path(result).exists()


class TestPlotBlandAltman:
    def test_creates_file(self, sample_data, tmp_output):
        pred, gt = sample_data
        save_path = str(Path(tmp_output) / "ba.png")
        result = plot_bland_altman(pred, gt, save_path=save_path)
        assert Path(result).exists()

    def test_zero_bias(self, tmp_output):
        """Identical arrays should produce zero bias line."""
        arr = np.array([70.0, 75.0, 80.0])
        save_path = str(Path(tmp_output) / "ba_zero.png")
        result = plot_bland_altman(arr, arr, save_path=save_path)
        assert Path(result).exists()


class TestPlotErrorHistogram:
    def test_creates_file(self, sample_data, tmp_output):
        pred, gt = sample_data
        save_path = str(Path(tmp_output) / "err.png")
        result = plot_error_histogram(pred, gt, save_path=save_path)
        assert Path(result).exists()

    def test_custom_bins(self, sample_data, tmp_output):
        pred, gt = sample_data
        save_path = str(Path(tmp_output) / "err_bins.png")
        result = plot_error_histogram(pred, gt, save_path=save_path, bins=10)
        assert Path(result).exists()


class TestPlotConfidenceVsError:
    def test_creates_file(self, tmp_output):
        conf = np.random.rand(50)
        errors = np.random.randn(50)
        save_path = str(Path(tmp_output) / "conf.png")
        result = plot_confidence_vs_error(conf, errors, save_path=save_path)
        assert Path(result).exists()


class TestPlotPerSubjectMAE:
    def test_creates_file(self, tmp_output):
        subjects = np.array(["s1", "s2", "s3", "s4", "s5"])
        maes = np.array([1.2, 3.5, 2.1, 8.0, 1.8])
        save_path = str(Path(tmp_output) / "psmae.png")
        result = plot_per_subject_mae(subjects, maes, save_path=save_path)
        assert Path(result).exists()


class TestPlotLatencyDistribution:
    def test_creates_file(self, tmp_output):
        latencies = np.random.exponential(15, 200)
        save_path = str(Path(tmp_output) / "lat.png")
        result = plot_latency_distribution(latencies, save_path=save_path)
        assert Path(result).exists()


class TestPlotMethodComparison:
    def test_creates_file(self, tmp_output):
        methods = ["pos", "chrom", "green"]
        metrics = {
            "MAE": [2.1, 3.5, 4.2],
            "RMSE": [3.0, 4.1, 5.0],
            "r": [0.95, 0.88, 0.82],
        }
        save_path = str(Path(tmp_output) / "comp.png")
        result = plot_method_comparison(methods, metrics, save_path=save_path)
        assert Path(result).exists()


class TestPlotLightingConditions:
    def test_creates_file(self, tmp_output):
        results = {"bright": {"mae": 2.0}, "dim": {"mae": 5.0}, "dark": {"mae": 8.0}}
        save_path = str(Path(tmp_output) / "light.png")
        result = plot_lighting_conditions(results, save_path=save_path)
        assert Path(result).exists()


class TestPlotDistanceConditions:
    def test_creates_file(self, tmp_output):
        results = {"0.5m": {"mae": 1.5}, "1.0m": {"mae": 2.5}, "2.0m": {"mae": 5.0}}
        save_path = str(Path(tmp_output) / "dist.png")
        result = plot_distance_conditions(results, save_path=save_path)
        assert Path(result).exists()
