"""
Tests for Evaluation Dataset Loaders
======================================

Tests UBFCRPPGLoader and CustomRecordingLoader with synthetic data
structures, including the 3-row UBFC-rPPG ground truth format fix.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from evaluation.datasets import UBFCRPPGLoader, CustomRecordingLoader, SubjectData


# ---------------------------------------------------------------------------
# UBFCRPPGLoader tests
# ---------------------------------------------------------------------------

class TestUBFCRPPGLoader:
    def test_is_available_returns_false_for_missing_dir(self, tmp_path):
        loader = UBFCRPPGLoader(data_dir=str(tmp_path / "nonexistent"))
        assert loader.is_available() is False

    def test_list_subjects_empty_dir(self, tmp_path):
        loader = UBFCRPPGLoader(data_dir=str(tmp_path))
        assert loader.list_subjects() == []

    def test_list_subjects_with_valid_subject(self, tmp_path):
        """A subject dir with a video file should be listed."""
        subject_dir = tmp_path / "subject01"
        subject_dir.mkdir()
        (subject_dir / "video.avi").touch()
        loader = UBFCRPPGLoader(data_dir=str(tmp_path))
        subjects = loader.list_subjects()
        assert "subject01" in subjects

    def test_list_subjects_skips_dirs_without_video(self, tmp_path):
        """A subject dir without a video file should be skipped."""
        subject_dir = tmp_path / "subject01"
        subject_dir.mkdir()
        (subject_dir / "ground_truth.txt").touch()
        loader = UBFCRPPGLoader(data_dir=str(tmp_path))
        assert loader.list_subjects() == []

    def test_is_available_true_with_valid_subject(self, tmp_path):
        subject_dir = tmp_path / "subject01"
        subject_dir.mkdir()
        (subject_dir / "video.avi").touch()
        loader = UBFCRPPGLoader(data_dir=str(tmp_path))
        assert loader.is_available() is True

    def test_load_ground_truth_bpm_format(self, tmp_path):
        """Single-column BPM values (one per line)."""
        subject_dir = tmp_path / "subject01"
        subject_dir.mkdir()
        gt_path = subject_dir / "ground_truth.txt"
        gt_path.write_text("72.5\n73.0\n74.2\n")

        loader = UBFCRPPGLoader(data_dir=str(tmp_path), ground_truth_format="bpm")
        bpm, ppg = loader._load_ground_truth(subject_dir, video_fps=30)
        assert bpm is not None
        assert ppg is None
        np.testing.assert_allclose(bpm, [72.5, 73.0, 74.2])

    def test_load_ground_truth_3row_ubfc_format(self, tmp_path):
        """
        Real UBFC-rPPG format: 3 rows (PPG waveform, BPM, timestamps).
        Row 0 = PPG, Row 1 = BPM, Row 2 = timestamps.
        """
        subject_dir = tmp_path / "subject01"
        subject_dir.mkdir()
        gt_path = subject_dir / "ground_truth.txt"

        # Create 3-row file
        ppg_row = np.random.rand(100) * 1000  # PPG waveform
        bpm_row = np.random.uniform(60, 100, 100)  # BPM values
        ts_row = np.linspace(0, 3.33, 100)  # timestamps

        data = np.vstack([ppg_row, bpm_row, ts_row])
        np.savetxt(str(gt_path), data)

        loader = UBFCRPPGLoader(data_dir=str(tmp_path), ground_truth_format="auto")
        bpm, ppg = loader._load_ground_truth(subject_dir, video_fps=30)

        # Should extract BPM from row 1 and PPG from row 0
        assert bpm is not None
        assert ppg is not None
        np.testing.assert_allclose(bpm, bpm_row)
        np.testing.assert_allclose(ppg, ppg_row)

    def test_load_ground_truth_3row_does_not_flatten(self, tmp_path):
        """
        Regression test: the old code did raw.flatten() which concatenated
        all 3 rows into one nonsense array. The fix must NOT flatten.
        """
        subject_dir = tmp_path / "subject01"
        subject_dir.mkdir()
        gt_path = subject_dir / "ground_truth.txt"

        # 3 rows with different values
        row0 = np.ones(50) * 100  # PPG
        row1 = np.ones(50) * 75   # BPM
        row2 = np.ones(50) * 0.1  # timestamps

        data = np.vstack([row0, row1, row2])
        np.savetxt(str(gt_path), data)

        loader = UBFCRPPGLoader(data_dir=str(tmp_path), ground_truth_format="auto")
        bpm, ppg = loader._load_ground_truth(subject_dir, video_fps=30)

        # BPM should be 75 (row 1), not a flattened mess
        assert bpm is not None
        np.testing.assert_allclose(bpm, 75.0)

    def test_load_ground_truth_long_1d_array_as_ppg(self, tmp_path):
        """Long 1-D array (>= 1000 elements) should be treated as PPG."""
        subject_dir = tmp_path / "subject01"
        subject_dir.mkdir()
        gt_path = subject_dir / "ground_truth.txt"

        ppg_signal = np.random.rand(2000) * 1000
        np.savetxt(str(gt_path), ppg_signal)

        loader = UBFCRPPGLoader(data_dir=str(tmp_path), ground_truth_format="auto")
        bpm, ppg = loader._load_ground_truth(subject_dir, video_fps=30)
        # Should be treated as PPG and BPM derived from it
        assert ppg is not None
        assert bpm is not None

    def test_load_ground_truth_missing_file(self, tmp_path):
        subject_dir = tmp_path / "subject01"
        subject_dir.mkdir()
        loader = UBFCRPPGLoader(data_dir=str(tmp_path))
        bpm, ppg = loader._load_ground_truth(subject_dir, video_fps=30)
        assert bpm is None
        assert ppg is None

    def test_load_subject_returns_none_for_missing(self, tmp_path):
        loader = UBFCRPPGLoader(data_dir=str(tmp_path))
        result = loader.load_subject("subject01", load_video=False)
        assert result is None

    def test_ppg_to_bpm_basic(self):
        """_ppg_to_bpm should return an array of BPM values."""
        fps = 30
        duration = 30  # seconds
        n_samples = fps * duration
        t = np.linspace(0, duration, n_samples)
        # Simulate 72 BPM signal (1.2 Hz)
        ppg = np.sin(2 * np.pi * 1.2 * t)

        bpm = UBFCRPPGLoader._ppg_to_bpm(ppg, fps)
        assert len(bpm) > 0
        # BPM should be around 72
        median_bpm = np.median(bpm)
        assert 65 < median_bpm < 80, f"Expected ~72 BPM, got {median_bpm}"

    def test_ppg_to_bpm_too_short(self):
        """Very short PPG should return empty array."""
        ppg = np.random.rand(10)
        bpm = UBFCRPPGLoader._ppg_to_bpm(ppg, fps=30)
        assert len(bpm) == 0


# ---------------------------------------------------------------------------
# CustomRecordingLoader tests
# ---------------------------------------------------------------------------

class TestCustomRecordingLoader:
    def test_missing_manifest_raises(self, tmp_path):
        loader = CustomRecordingLoader(data_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            loader._load_manifest()

    def test_list_recordings(self, tmp_path):
        manifest = {
            "recordings": [
                {"id": "rec1", "video": "rec1/video.avi", "ground_truth": "rec1/gt.json"},
                {"id": "rec2", "video": "rec2/video.avi", "ground_truth": "rec2/gt.json"},
            ]
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        loader = CustomRecordingLoader(data_dir=str(tmp_path))
        recordings = loader.list_recordings()
        assert recordings == ["rec1", "rec2"]

    def test_load_recording_missing_video(self, tmp_path):
        manifest = {
            "recordings": [
                {"id": "rec1", "video": "rec1/video.avi", "ground_truth": "rec1/gt.json"},
            ]
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        loader = CustomRecordingLoader(data_dir=str(tmp_path))
        result = loader.load_recording("rec1")
        # Video doesn't exist, should return None
        assert result is None

    def test_load_recording_not_in_manifest(self, tmp_path):
        manifest = {"recordings": []}
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        loader = CustomRecordingLoader(data_dir=str(tmp_path))
        result = loader.load_recording("nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# SubjectData dataclass tests
# ---------------------------------------------------------------------------

class TestSubjectData:
    def test_creation(self):
        data = SubjectData(
            subject_id="test",
            video_frames=None,
            video_fps=30,
            ground_truth_bpm=np.array([72.0]),
            ground_truth_ppg=None,
        )
        assert data.subject_id == "test"
        assert data.video_fps == 30
