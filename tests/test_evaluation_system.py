"""
System-Level Evaluation Tests
==============================

Tests pipeline latency, measurement timeout handling, low-quality signal
resilience, and the evaluation harnesses (evaluate_age, evaluate_pupil,
evaluate_pipeline).
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Pipeline latency tests
# ---------------------------------------------------------------------------

class TestPipelineLatency:
    """Verify VitalsPipeline meets latency constraints."""

    def test_single_frame_latency_under_threshold(self):
        """A single frame through the pipeline should complete within 200ms."""
        from src.contactless.pipeline import VitalsPipeline

        try:
            pipeline = VitalsPipeline(fps=30, use_threading=False)
        except Exception:
            pytest.skip("Pipeline components not available")

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        start = time.perf_counter()
        reading = pipeline.process_frame(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000

        pipeline.close()
        assert elapsed_ms < 200, f"Single frame took {elapsed_ms:.0f}ms (threshold: 200ms)"

    def test_pipeline_returns_vitals_reading(self):
        """Pipeline should always return a VitalsReading, even on bad input."""
        from src.contactless.pipeline import VitalsPipeline, VitalsReading

        try:
            pipeline = VitalsPipeline(fps=30, use_threading=False)
        except Exception:
            pytest.skip("Pipeline components not available")

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        reading = pipeline.process_frame(frame)
        pipeline.close()

        assert isinstance(reading, VitalsReading)
        assert hasattr(reading, "total_latency_ms")
        assert reading.total_latency_ms >= 0

    def test_pipeline_throughput_above_5fps(self):
        """Pipeline should sustain at least 5 FPS on synthetic frames."""
        from src.contactless.pipeline import VitalsPipeline

        try:
            pipeline = VitalsPipeline(fps=30, use_threading=False)
        except Exception:
            pytest.skip("Pipeline components not available")

        n_frames = 20
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(n_frames)
        ]

        start = time.perf_counter()
        for frame in frames:
            pipeline.process_frame(frame)
        total = time.perf_counter() - start

        pipeline.close()
        throughput = n_frames / total
        assert throughput >= 5.0, f"Throughput {throughput:.1f} FPS below 5 FPS threshold"


# ---------------------------------------------------------------------------
# Measurement timeout tests
# ---------------------------------------------------------------------------

class TestMeasurementTimeout:
    """Verify the pipeline handles timeouts gracefully."""

    def test_pipeline_does_not_crash_on_black_frame(self):
        """Black frames (no face) should return reading with face_detected=False."""
        from src.contactless.pipeline import VitalsPipeline

        try:
            pipeline = VitalsPipeline(fps=30, use_threading=False)
        except Exception:
            pytest.skip("Pipeline components not available")

        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        reading = pipeline.process_frame(black_frame)
        pipeline.close()

        assert reading.face_detected is False
        assert reading.heart_rate_bpm is None
        assert reading.module_errors == {} or len(reading.module_errors) == 0

    def test_pipeline_handles_noisy_frame(self):
        """Pure noise frames should not crash the pipeline."""
        from src.contactless.pipeline import VitalsPipeline

        try:
            pipeline = VitalsPipeline(fps=30, use_threading=False)
        except Exception:
            pytest.skip("Pipeline components not available")

        noise_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        reading = pipeline.process_frame(noise_frame)
        pipeline.close()

        # Should return a valid reading (face may or may not be detected)
        assert hasattr(reading, "face_detected")
        assert hasattr(reading, "total_latency_ms")

    def test_pipeline_handles_tiny_frame(self):
        """Very small frames should not crash the pipeline."""
        from src.contactless.pipeline import VitalsPipeline

        try:
            pipeline = VitalsPipeline(fps=30, use_threading=False)
        except Exception:
            pytest.skip("Pipeline components not available")

        tiny_frame = np.ones((10, 10, 3), dtype=np.uint8) * 128
        reading = pipeline.process_frame(tiny_frame)
        pipeline.close()

        assert hasattr(reading, "face_detected")


# ---------------------------------------------------------------------------
# Low-quality signal tests
# ---------------------------------------------------------------------------

class TestLowQualitySignal:
    """Verify detectors handle low-quality signals without crashing."""

    def test_hr_detector_on_constant_signal(self):
        """Heart rate detector on constant (DC) signal should not crash."""
        from src.contactless.heart_rate import HeartRateDetector

        try:
            detector = HeartRateDetector(fps=30)
        except Exception:
            pytest.skip("HeartRateDetector not available")

        # Simulate a face result with constant RGB values
        mock_face = MagicMock()
        mock_face.detected = True
        mock_face.face_roi = np.full((100, 100, 3), 128, dtype=np.uint8)
        mock_face.forehead_roi = np.full((50, 50, 3), 128, dtype=np.uint8)
        mock_face.bbox = (100, 50, 200, 200)

        # Feed many constant frames
        for _ in range(60):
            result = detector.add_frame(mock_face)

        # Should return a result (possibly None BPM) without crashing
        assert result is not None
        assert hasattr(result, "bpm")
        assert hasattr(result, "confidence")

    def test_respiration_detector_on_constant_signal(self):
        """Respiration detector on constant signal should not crash."""
        from src.contactless.respiration import RespirationDetector

        try:
            detector = RespirationDetector(fps=30)
        except Exception:
            pytest.skip("RespirationDetector not available")

        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        mock_face = MagicMock()
        mock_face.detected = True
        mock_face.face_roi = np.full((100, 100, 3), 128, dtype=np.uint8)
        mock_face.forehead_roi = np.full((50, 50, 3), 128, dtype=np.uint8)
        mock_face.bbox = (100, 50, 200, 200)

        for _ in range(60):
            result = detector.add_frame(frame, mock_face)

        assert result is not None
        assert hasattr(result, "bpm")


# ---------------------------------------------------------------------------
# Evaluation harness tests
# ---------------------------------------------------------------------------

class TestEvaluateAgeHarness:
    """Tests for evaluate_age.py harness logic (without real data)."""

    def test_age_groups_cover_common_ranges(self):
        """AGE_GROUPS dict should cover ages 0-120."""
        from evaluation.evaluate_age import AGE_GROUPS

        # Check coverage
        covered = set()
        for name, (lo, hi) in AGE_GROUPS.items():
            for age in range(lo, min(hi + 1, 121)):
                covered.add(age)

        # Should cover at least 0-100
        for age in range(0, 101):
            assert age in covered, f"Age {age} not covered by any group"

    def test_evaluate_age_returns_empty_on_missing_data(self, tmp_path):
        """Should return {} when data directory doesn't exist."""
        from evaluation.evaluate_age import evaluate_age_on_test_split

        result = evaluate_age_on_test_split(
            data_dir=str(tmp_path / "nonexistent"),
            output_dir=str(tmp_path / "output"),
        )
        assert result == {}


class TestEvaluatePupilHarness:
    """Tests for evaluate_pupil.py harness logic."""

    def test_synthetic_pupil_evaluation(self, tmp_path):
        """Synthetic mode should produce a valid report."""
        from evaluation.evaluate_pupil import evaluate_pupil_on_synthetic

        result = evaluate_pupil_on_synthetic(
            n_samples=100,
            noise_std=0.1,
            output_dir=str(tmp_path / "pupil_output"),
        )

        assert "overall" in result
        assert "mae" in result["overall"]
        assert "rmse" in result["overall"]
        assert "pearson_r" in result["overall"]
        assert result["overall"]["mae"] >= 0
        assert result["overall"]["rmse"] >= 0
        assert 0 <= result["overall"]["pearson_r"] <= 1

    def test_synthetic_pupil_creates_plots(self, tmp_path):
        """Synthetic mode should create plot files."""
        from evaluation.evaluate_pupil import evaluate_pupil_on_synthetic

        evaluate_pupil_on_synthetic(
            n_samples=50,
            output_dir=str(tmp_path / "pupil_plots"),
        )

        output_dir = tmp_path / "pupil_plots"
        assert (output_dir / "predicted_vs_gt.png").exists()
        assert (output_dir / "bland_altman.png").exists()
        assert (output_dir / "error_hist.png").exists()
        assert (output_dir / "pupil_evaluation_report.json").exists()


class TestEvaluatePipelineHarness:
    """Tests for evaluate_pipeline.py harness logic."""

    def test_benchmark_returns_report(self, tmp_path):
        """Benchmark mode should produce a valid report dict."""
        from evaluation.evaluate_pipeline import benchmark_pipeline

        try:
            from src.contactless.pipeline import VitalsPipeline
        except Exception:
            pytest.skip("Pipeline not available")

        result = benchmark_pipeline(
            n_frames=10,
            fps=30,
            output_dir=str(tmp_path / "pipe_output"),
        )

        assert "throughput_fps" in result
        assert "latency" in result
        assert "module_success_rate" in result
        assert result["throughput_fps"] > 0

    def test_stress_test_returns_report(self, tmp_path):
        """Stress test should produce a valid report dict."""
        from evaluation.evaluate_pipeline import stress_test_pipeline

        try:
            from src.contactless.pipeline import VitalsPipeline
        except Exception:
            pytest.skip("Pipeline not available")

        result = stress_test_pipeline(
            n_frames=20,
            error_injection_rate=0.2,
            output_dir=str(tmp_path / "stress_output"),
        )

        assert "crash_rate" in result
        assert "graceful_degradation_rate" in result
        assert 0 <= result["crash_rate"] <= 1
        assert 0 <= result["graceful_degradation_rate"] <= 1


# ---------------------------------------------------------------------------
# Reports module tests
# ---------------------------------------------------------------------------

class TestReports:
    """Tests for evaluation/reports.py."""

    def test_generate_report(self):
        from evaluation.reports import generate_report

        report = generate_report(
            hr_results={"pos": {"mae": 2.1, "rmse": 3.0}},
            dataset_info={"name": "ubfc-rppg", "n_subjects": 42},
        )
        assert report.timestamp is not None
        assert report.hr_results["pos"]["mae"] == 2.1

    def test_save_and_load_report(self, tmp_path):
        from evaluation.reports import generate_report, save_report

        report = generate_report(
            hr_results={"pos": {"mae": 2.1}},
            dataset_info={"name": "test"},
        )
        save_path = str(tmp_path / "report.json")
        save_report(report, save_path)

        with open(save_path) as f:
            loaded = json.load(f)

        assert loaded["hr_results"]["pos"]["mae"] == 2.1
        assert loaded["dataset_info"]["name"] == "test"

    def test_markdown_summary(self):
        from evaluation.reports import generate_report, generate_markdown_summary

        report = generate_report(
            hr_results={"pos": {"mae": 2.1, "rmse": 3.0, "pearson_r": 0.95,
                                "r2": 0.90, "percentage_within_5bpm": 0.85,
                                "percentage_within_10bpm": 0.95,
                                "failure_rate": 0.05, "n_valid": 100, "n_total": 105}},
            dataset_info={"name": "ubfc-rppg"},
        )
        md = generate_markdown_summary(report)
        assert "# Evaluation Report" in md
        assert "Heart Rate Evaluation" in md
        assert "pos" in md.lower()
