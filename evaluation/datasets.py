"""
Dataset Loaders for Evaluation
===============================

Provides standardized loaders for:
- UBFC-rPPG dataset (42 subjects, webcam + pulse oximeter ground truth)
- Custom recordings with manifest files

Each loader returns a consistent format for the evaluation harnesses.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SubjectData:
    """Container for a single subject's recording data."""
    subject_id: str
    video_frames: Optional[np.ndarray]  # (N, H, W, 3) BGR or None if loading lazily
    video_fps: int
    ground_truth_bpm: Optional[np.ndarray]  # per-frame or per-second BPM
    ground_truth_ppg: Optional[np.ndarray]  # raw PPG waveform (if available)
    metadata: Dict = field(default_factory=dict)


class UBFCRPPGLoader:
    """
    Loader for the UBFC-rPPG dataset.

    UBFC-rPPG contains 42 subjects recorded at 30 FPS with a Logitech C920
    webcam, synchronized with a pulse oximeter providing ground-truth PPG.

    Expected directory structure:
        data/ubfc-rppg/
        ├── subject1/
        │   ├── video.avi          (or .mp4)
        │   └── ground_truth.txt   (one BPM value per line, or PPG samples)
        ├── subject2/
        │   └── ...
        └── dataset_info.json      (optional metadata)

    Reference: UBFC-rPPG: A Database for Remote Photoplethysmography
    Signal Processing, Bobbia et al. 2019
    """

    def __init__(self, data_dir: str = "data/ubfc-rppg", ground_truth_format: str = "bpm"):
        """
        Args:
            data_dir: Path to the UBFC-rPPG dataset root directory
            ground_truth_format: How ground truth is stored
                "bpm"   - one BPM value per line
                "ppg"   - raw PPG samples (one per line), BPM computed from peaks
                "auto"  - try to detect format
        """
        self.data_dir = Path(data_dir)
        self.ground_truth_format = ground_truth_format
        self._subject_cache: Dict[str, SubjectData] = {}

    def is_available(self) -> bool:
        """Check if the dataset directory exists and has subjects."""
        if not self.data_dir.exists():
            logger.warning(f"UBFC-rPPG data directory not found: {self.data_dir}")
            return False

        subjects = self.list_subjects()
        if len(subjects) == 0:
            logger.warning(f"No subjects found in {self.data_dir}")
            return False

        return True

    def list_subjects(self) -> List[str]:
        """Return sorted list of available subject IDs."""
        if not self.data_dir.exists():
            return []

        subjects = []
        for entry in sorted(self.data_dir.iterdir()):
            if entry.is_dir() and entry.name.startswith("subject"):
                # Verify it has a video file
                video_files = list(entry.glob("video.*"))
                if video_files:
                    subjects.append(entry.name)
        return subjects

    def _find_video_path(self, subject_dir: Path) -> Optional[Path]:
        """Find the video file in a subject directory."""
        for ext in [".avi", ".mp4", ".mkv", ".mov"]:
            video_path = subject_dir / f"video{ext}"
            if video_path.exists():
                return video_path
        # Try any video file
        for ext in ["*.avi", "*.mp4", "*.mkv", "*.mov"]:
            matches = list(subject_dir.glob(ext))
            if matches:
                return matches[0]
        return None

    def _load_ground_truth(self, subject_dir: Path, video_fps: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load ground truth BPM or PPG waveform.

        Returns:
            (bpm_values, ppg_waveform) — at least one will be non-None
        """
        gt_path = subject_dir / "ground_truth.txt"
        if not gt_path.exists():
            logger.warning(f"No ground truth file for {subject_dir.name}")
            return None, None

        try:
            raw = np.loadtxt(str(gt_path))
        except Exception as e:
            logger.warning(f"Failed to load ground truth from {gt_path}: {e}")
            return None, None

        if self.ground_truth_format == "bpm" or (
            self.ground_truth_format == "auto" and raw.ndim == 1 and len(raw) < 1000
        ):
            # One BPM value per line (or per second)
            return raw, None

        if self.ground_truth_format == "ppg" or (
            self.ground_truth_format == "auto" and (raw.ndim > 1 or len(raw) >= 1000)
        ):
            # Raw PPG waveform — compute BPM from peaks
            ppg = raw.flatten().astype(np.float64)
            bpm_values = self._ppg_to_bpm(ppg, video_fps)
            return bpm_values, ppg

        # Fallback: treat as BPM
        return raw, None

    @staticmethod
    def _ppg_to_bpm(ppg: np.ndarray, fps: int) -> np.ndarray:
        """
        Convert a raw PPG waveform to per-second BPM values using a sliding window.

        Args:
            ppg: 1D PPG signal (one sample per camera frame)
            fps: Camera frame rate

        Returns:
            1D array of BPM values (one per second)
        """
        import scipy.signal

        window_sec = 10
        step_sec = 1
        window_size = fps * window_sec
        step_size = fps * step_sec

        if len(ppg) < window_size:
            return np.array([])

        bpm_values = []
        for start in range(0, len(ppg) - window_size + 1, step_size):
            segment = ppg[start : start + window_size]
            # Detrend
            segment = scipy.signal.detrend(segment)
            # Bandpass 0.7–4.0 Hz
            nyq = fps / 2.0
            try:
                b, a = scipy.signal.butter(4, [0.7 / nyq, 4.0 / nyq], btype="band")
                segment = scipy.signal.filtfilt(b, a, segment)
            except Exception:
                pass
            # FFT peak
            fft_mag = np.abs(np.fft.rfft(segment))
            freqs = np.fft.rfftfreq(len(segment), d=1.0 / fps)
            mask = (freqs >= 0.7) & (freqs <= 4.0)
            if not np.any(mask):
                bpm_values.append(np.nan)
                continue
            peak_freq = freqs[mask][np.argmax(fft_mag[mask])]
            bpm_values.append(peak_freq * 60.0)

        return np.array(bpm_values)

    def load_subject(self, subject_id: str, load_video: bool = True) -> Optional[SubjectData]:
        """
        Load data for a single subject.

        Args:
            subject_id: e.g. "subject1"
            load_video: If True, decode all frames into memory (memory-intensive).
                        If False, video_frames is None and you must use
                        load_subject_frame() for lazy loading.

        Returns:
            SubjectData or None if the subject cannot be loaded
        """
        if subject_id in self._subject_cache:
            return self._subject_cache[subject_id]

        subject_dir = self.data_dir / subject_id
        if not subject_dir.exists():
            logger.warning(f"Subject directory not found: {subject_dir}")
            return None

        # Load video
        video_frames = None
        video_fps = 30  # default

        video_path = self._find_video_path(subject_dir)
        if video_path is None:
            logger.warning(f"No video file found for {subject_id}")
            return None

        if load_video:
            video_frames, video_fps = self._load_video(video_path)
            if video_frames is None:
                return None
        else:
            video_fps = self._probe_fps(video_path)

        # Load ground truth
        gt_bpm, gt_ppg = self._load_ground_truth(subject_dir, video_fps)

        subject_data = SubjectData(
            subject_id=subject_id,
            video_frames=video_frames,
            video_fps=video_fps,
            ground_truth_bpm=gt_bpm,
            ground_truth_ppg=gt_ppg,
            metadata={"video_path": str(video_path)},
        )

        self._subject_cache[subject_id] = subject_data
        return subject_data

    def load_all_subjects(self, load_video: bool = True) -> List[SubjectData]:
        """Load all available subjects."""
        subjects = []
        for subject_id in self.list_subjects():
            data = self.load_subject(subject_id, load_video=load_video)
            if data is not None:
                subjects.append(data)
        logger.info(f"Loaded {len(subjects)} subjects from UBFC-rPPG")
        return subjects

    def _load_video(self, video_path: Path) -> Tuple[Optional[np.ndarray], int]:
        """Decode all frames from a video file."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None, 30

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            logger.error(f"No frames read from {video_path}")
            return None, fps

        return np.array(frames), fps

    def _probe_fps(self, video_path: Path) -> int:
        """Get FPS from a video without loading all frames."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 30
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        cap.release()
        return fps


class CustomRecordingLoader:
    """
    Loader for custom recordings with a manifest file.

    Expected structure:
        data/custom/
        ├── manifest.json
        ├── recording1/
        │   ├── video.avi
        │   └── ground_truth.json
        └── recording2/
            └── ...

    manifest.json format:
    {
        "recordings": [
            {
                "id": "recording1",
                "video": "recording1/video.avi",
                "ground_truth": "recording1/ground_truth.json",
                "fps": 30,
                "metadata": {"subject_age": 25, "lighting": "normal"}
            }
        ]
    }
    """

    def __init__(self, data_dir: str, manifest_path: Optional[str] = None):
        """
        Args:
            data_dir: Root directory for recordings
            manifest_path: Path to manifest JSON. If None, uses data_dir/manifest.json
        """
        self.data_dir = Path(data_dir)
        self.manifest_path = Path(manifest_path) if manifest_path else self.data_dir / "manifest.json"
        self._manifest: Optional[Dict] = None

    def _load_manifest(self) -> Dict:
        """Load and cache the manifest file."""
        if self._manifest is not None:
            return self._manifest

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, "r") as f:
            self._manifest = json.load(f)

        return self._manifest

    def list_recordings(self) -> List[str]:
        """Return list of recording IDs from the manifest."""
        manifest = self._load_manifest()
        return [r["id"] for r in manifest.get("recordings", [])]

    def load_recording(self, recording_id: str) -> Optional[SubjectData]:
        """Load a single recording by ID."""
        manifest = self._load_manifest()

        recording_info = None
        for r in manifest.get("recordings", []):
            if r["id"] == recording_id:
                recording_info = r
                break

        if recording_info is None:
            logger.warning(f"Recording '{recording_id}' not found in manifest")
            return None

        # Load video
        video_path = self.data_dir / recording_info["video"]
        if not video_path.exists():
            logger.warning(f"Video file not found: {video_path}")
            return None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None

        fps = recording_info.get("fps", int(cap.get(cv2.CAP_PROP_FPS)) or 30)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        video_frames = np.array(frames) if frames else None

        # Load ground truth
        gt_bpm = None
        gt_ppg = None
        gt_path = self.data_dir / recording_info.get("ground_truth", "")
        if gt_path.exists():
            try:
                with open(gt_path, "r") as f:
                    gt_data = json.load(f)
                if "bpm_values" in gt_data:
                    gt_bpm = np.array(gt_data["bpm_values"])
                elif "ppg_samples" in gt_data:
                    gt_ppg = np.array(gt_data["ppg_samples"])
            except Exception as e:
                logger.warning(f"Failed to load ground truth for {recording_id}: {e}")

        return SubjectData(
            subject_id=recording_id,
            video_frames=video_frames,
            video_fps=fps,
            ground_truth_bpm=gt_bpm,
            ground_truth_ppg=gt_ppg,
            metadata=recording_info.get("metadata", {}),
        )

    def load_all_recordings(self) -> List[SubjectData]:
        """Load all recordings from the manifest."""
        recordings = []
        for rec_id in self.list_recordings():
            data = self.load_recording(rec_id)
            if data is not None:
                recordings.append(data)
        logger.info(f"Loaded {len(recordings)} recordings from {self.manifest_path}")
        return recordings
