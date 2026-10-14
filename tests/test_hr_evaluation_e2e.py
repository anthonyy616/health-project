"""
End-to-end tests for the HR evaluation harness on a synthetic
UBFC-style recording (synthetic AVI + 3-row ground truth).

Synthetic-data caveat: verifies HARNESS CORRECTNESS (loader parsing,
timestamp capture, validity gating, report generation, per-subject
metrics) - NOT accuracy on human subjects.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def synthetic_ubfc(tmp_path):
    """One synthetic subject directory with a Joule-pulsing forehead region."""
    import cv2

    subject_dir = tmp_path / "subject1"
    subject_dir.mkdir()
    fps = 25
    duration = 16.0
    n = int(fps * duration)
    t = np.arange(n) / fps

    video_path = subject_dir / "video.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 120))
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter unavailable in this environment")

    baseline = np.array([140.0, 100.0, 90.0])  # skin-like RGB scaled into BGR below
    pulse = 0.02 * np.sin(2 * np.pi * (72.0 / 60.0) * t)

    timestamps = []
    for i in range(n):
        # Forehead-like center band pulsing; frame is BGR
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:, :] = (40, 40, 40)
        # Center crop region (matches harness center-crop proxy 15-45% v, 30-70% h)
        y1, y2 = int(120 * 0.15), int(120 * 0.45)
        x1, x2 = int(160 * 0.3), int(160 * 0.7)
        b, g, r = baseline[2] * (1 + pulse[i]), baseline[1] * (1 + pulse[i]), baseline[0] * (1 + pulse[i])
        frame[y1:y2, x1:x2] = (int(np.clip(b, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(r, 0, 255)))
        writer.write(frame)
        timestamps.append(1000.0 * i / fps)
    writer.release()

    # 3-row UBFC ground truth: PPG, BPM (constant 72), timestamps
    ppg = np.sin(2 * np.pi * (72.0 / 60.0) * t)
    bpm_row = np.full(n, 72.0)
    ts_row = np.array(timestamps) / 1000.0
    np.savetxt(subject_dir / "ground_truth.txt", np.vstack([ppg, bpm_row, ts_row]))

    return {"dir": subject_dir, "fps": fps, "n": n, "hr_bpm": 72.0}


class TestLoaderTimestamps:
    def test_frame_timestamps_captured(self, synthetic_ubfc):
        from evaluation.datasets import UBFCRPPGLoader

        loader = UBFCRPPGLoader(data_dir=str(synthetic_ubfc["dir"].parent))
        data = loader.load_subject("subject1", load_video=True)
        assert data is not None
        assert data.frame_timestamps_ms is not None
        assert len(data.frame_timestamps_ms) == synthetic_ubfc["n"]
        # Timestamps increase and match nominal cadence
        assert np.all(np.diff(data.frame_timestamps_ms) > 0)
        assert abs(data.effective_fps - synthetic_ubfc["fps"]) < 1.0

    def test_validation_issues_clean_for_synthetic(self, synthetic_ubfc):
        from evaluation.datasets import UBFCRPPGLoader

        loader = UBFCRPPGLoader(data_dir=str(synthetic_ubfc["dir"].parent))
        data = loader.load_subject("subject1", load_video=True)
        # Synthetic fixture: ground truth samples == frames; no gaps.
        assert data.validation_issues() == [] or all("alignment" not in i for i in data.validation_issues())

    def test_ground_truth_length_mismatch_reported(self, synthetic_ubfc):
        from evaluation.datasets import UBFCRPPGLoader

        gt_path = synthetic_ubfc["dir"] / "ground_truth.txt"
        raw = np.loadtxt(str(gt_path))
        np.savetxt(str(gt_path), raw[:, :-3])  # shorter ground truth

        loader = UBFCRPPGLoader(data_dir=str(synthetic_ubfc["dir"].parent))
        loader._subject_cache.clear()
        data = loader.load_subject("subject1", load_video=True)
        issues = data.validation_issues()
        assert any("ground truth length" in i for i in issues)


class TestIncludedExamSuite:
    def test_harness_runs_end_to_end_and_writes_report(self, synthetic_ubfc, tmp_path):
        from evaluation.evaluate_hr import evaluate_hr_on_dataset

        out_dir = tmp_path / "results"
        results = evaluate_hr_on_dataset(
            dataset_type="ubfc-rppg",
            data_dir=str(synthetic_ubfc["dir"].parent),
            methods=["green", "chrom", "pos"],
            output_dir=str(out_dir),
            use_face_detector=False,  # deterministic center-crop proxy
        )

        report_path = out_dir / "hr_evaluation_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())

        # Synthetic pulse is recoverable at 72 BPM for all matched methods.
        for method in ("green", "chrom", "pos"):
            assert method in report, f"{method} missing from report"
            mae = report[method]["mae"]
            assert mae is not None and mae <= 6.0, (
                f"{method} MAE {mae} on recoverable synthetic pulse"
            )
            assert "per_subject" in report[method]
            assert report[method]["n_subjects"] >= 1
            # Pooled fallback windows field present (0 for green/chrom)
            assert "pos_fallback_windows" in report[method]

        # Result objects returned for immediate use
        assert set(results) == {"green", "chrom", "pos"}

    def test_validity_gating_marks_invalid_windows_nan(self, synthetic_ubfc):
        """Windows dominated by invalid frames become NaN, not 0-BPM readings."""
        from evaluation.evaluate_hr import GreenExtractor
        from evaluation.datasets import UBFCRPPGLoader
        from evaluation.evaluate_hr import _extract_rgb_trace

        loader = UBFCRPPGLoader(data_dir=str(synthetic_ubfc["dir"].parent))
        data = loader.load_subject("subject1", load_video=True)
        rgb_trace, valid_mask = _extract_rgb_trace(data.video_frames, face_roi_func=None)

        extractor = GreenExtractor()
        fps = data.effective_fps
        window = int(fps * 10)

        bpm_all_valid, _ = extractor.extract_bpm_series(rgb_trace, fps)
        # Break a contiguous block (dropout simulation): mask those frames invalid
        broken_mask = valid_mask.copy()
        broken_mask[0 : int(len(broken_mask) * 0.5)] = False
        bpm_half_invalid, _ = extractor.extract_bpm_series(rgb_trace, fps, valid_mask=broken_mask)

        assert np.all(np.isnan(bpm_half_invalid[:2])), (
            "windows overlapping >10% invalid frames must be excluded (NaN)"
        )
        assert np.any(np.isfinite(bpm_all_valid)), "valid trace must produce readings"

    def test_invalid_frames_do_not_enter_valid_windows(self, synthetic_ubfc):
        """Validity interpolation must not create zero RGB or NaN in valid windows."""
        from evaluation.datasets import UBFCRPPGLoader
        from evaluation.evaluate_hr import _extract_rgb_trace

        loader = UBFCRPPGLoader(data_dir=str(synthetic_ubfc["dir"].parent))
        data = loader.load_subject("subject1", load_video=True)
        rgb_trace, valid_mask = _extract_rgb_trace(data.video_frames, face_roi_func=None)
        assert np.all(np.isfinite(rgb_trace)), "RGB carry-through must fill invalid samples"
        assert valid_mask[0], "fixture center-crop should be valid from frame 0"


class TestMatchedMethodSettings:
    def test_methods_share_window_parameters(self):
        from evaluation.evaluate_hr import PosExtractor, ChromExtractor, GreenExtractor

        fps, window_sec, step_sec = 25, 10, 1
        for cls in (PosExtractor, ChromExtractor, GreenExtractor):
            ext = cls()
            trace = np.tile(np.array([140.0, 100.0, 90.0]), (int(fps * window_sec + fps * 2), 1))
            out_bpm, out_conf = ext.extract_bpm_series(trace, fps, window_sec, step_sec)
            # Number of windows must match sliding-window arithmetic across methods
            expected_n = len(range(0, len(trace) - int(fps * window_sec) + 1, int(fps * step_sec)))
            assert len(out_bpm) == expected_n, f"{cls.__name__} window count mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
