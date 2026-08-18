"""
Respiration Detector
Stateful class that manages the previous-frame reference (needed for optical
flow, which is inherently a two-frame operation) and the rolling motion
buffer, calling motion_analysis.py functions in order.
"""

import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

from src.contactless.face_detection.detect import FaceResult
from src.contactless.respiration.motion_analysis import (
    extract_chest_bbox,
    crop_and_resize_roi,
    compute_vertical_flow,
    detrend_signal,
    bandpass_filter_respiration,
    detect_breathing_peaks,
    compute_bpm_from_peaks,
    compute_motion_quality
)

logger = logging.getLogger(__name__)


@dataclass
class RespirationResult:
    """Container for respiratory rate detection results"""
    bpm: Optional[float]
    confidence: float
    signal_quality: float
    buffer_fill: float
    inference_time_ms: float
    raw_motion_trace: Optional[np.ndarray] = None  # debug visualization only


class RespirationDetector:
    """
    Chest-ROI optical flow respiratory rate detector.

    Manages a rolling buffer of vertical chest motion values extracted via
    dense optical flow on a grayscale chest crop, and applies signal
    processing to extract breathing rate.

    Note: add_frame takes BOTH the raw frame and the FaceResult, unlike
    HeartRateDetector.add_frame(face_result) - the chest ROI has to be
    cropped from the raw frame and does not exist upstream.
    """

    def __init__(
        self,
        fps: int = 30,
        window_seconds: int = 20,
        min_seconds: float = 10.0,
        quality_threshold: float = 0.1,
        roi_size: Tuple[int, int] = (128, 128)
    ):
        """
        Initialize the respiration detector.

        Args:
            fps: Camera frames per second
            window_seconds: Length of rolling buffer in seconds (longer than
                rPPG's 10s: at 6 BPM one breath takes 10s, so 20s gives at
                least 2 full cycles even at the slowest rate)
            min_seconds: Minimum buffer fill before attempting BPM calculation
            quality_threshold: Minimum signal quality to attempt calculation
            roi_size: Fixed size of the grayscale chest ROI for optical flow
        """
        self.fps = fps
        self.window_seconds = window_seconds
        self.min_seconds = min_seconds
        self.quality_threshold = quality_threshold
        self.roi_size = roi_size

        self.buffer_size = int(fps * window_seconds)
        self.min_buffer_size = int(fps * min_seconds)

        # Rolling buffer of vertical motion values
        self._motion_buffer = deque(maxlen=self.buffer_size)

        # Previous frame's grayscale chest crop (t-1 reference for optical flow)
        self._prev_chest_gray: Optional[np.ndarray] = None

        # Cache for last successful result
        self._last_result: Optional[RespirationResult] = None

    def add_frame(self, frame: np.ndarray, face_result: FaceResult) -> RespirationResult:
        """
        Process a frame and update respiratory rate estimation.

        Args:
            frame: BGR image from OpenCV
            face_result: Face detection result (only bbox is used - the chest
                ROI is derived from it and cropped from the raw frame)

        Returns:
            RespirationResult with current estimation
        """
        start_time = time.perf_counter()

        # If no face detected or no bbox, can't compute valid flow across the
        # detection gap - the next real frame pair would produce a spurious
        # flow spike. Don't append to the buffer.
        if not face_result.detected or face_result.bbox is None:
            self._prev_chest_gray = None
            result = self._build_result(
                start_time,
                bpm=None,
                confidence=0.0,
                signal_quality=0.0
            )
            self._last_result = result
            return result

        chest_bbox = extract_chest_bbox(face_result.bbox, frame.shape[:2])

        # Face too close to frame edge - treat the same as "no face"
        if chest_bbox is None:
            self._prev_chest_gray = None
            result = self._build_result(
                start_time,
                bpm=None,
                confidence=0.0,
                signal_quality=0.0
            )
            self._last_result = result
            return result

        curr_gray = crop_and_resize_roi(frame, chest_bbox, self.roi_size)

        # First valid frame (or first after a reset/gap): flow can't be
        # computed yet - nothing is appended to the buffer this call.
        if self._prev_chest_gray is None:
            self._prev_chest_gray = curr_gray
            result = self._build_result(
                start_time,
                bpm=None,
                confidence=0.0,
                signal_quality=0.0
            )
            self._last_result = result
            return result

        motion = compute_vertical_flow(self._prev_chest_gray, curr_gray)
        self._motion_buffer.append(motion)
        self._prev_chest_gray = curr_gray

        buffer_fill = len(self._motion_buffer) / self.buffer_size

        # Buffer not filled to min_seconds yet - return partial result
        if len(self._motion_buffer) < self.min_buffer_size:
            result = self._build_result(
                start_time,
                bpm=None,
                confidence=0.0,
                signal_quality=0.0
            )
            self._last_result = result
            return result

        motion_trace = np.array(self._motion_buffer, dtype=np.float64)

        # Signal quality gate - no real breathing motion detected
        signal_quality = compute_motion_quality(motion_trace)
        if signal_quality < self.quality_threshold:
            result = self._build_result(
                start_time,
                bpm=None,
                confidence=0.0,
                signal_quality=signal_quality
            )
            self._last_result = result
            return result

        # Signal processing pipeline
        try:
            detrended = detrend_signal(motion_trace)
            filtered = bandpass_filter_respiration(detrended, self.fps)
            peaks = detect_breathing_peaks(filtered, self.fps)
            bpm, confidence = compute_bpm_from_peaks(peaks, self.fps)
        except Exception as e:
            logger.warning(f"Signal processing failed: {e}")
            result = self._build_result(
                start_time,
                bpm=None,
                confidence=0.0,
                signal_quality=signal_quality
            )
            self._last_result = result
            return result

        result = self._build_result(
            start_time,
            bpm=bpm,
            confidence=confidence,
            signal_quality=signal_quality
        )
        self._last_result = result
        return result

    def _build_result(
        self,
        start_time: float,
        bpm: Optional[float],
        confidence: float,
        signal_quality: float
    ) -> RespirationResult:
        """Build a RespirationResult from the current buffer state"""
        return RespirationResult(
            bpm=bpm,
            confidence=confidence,
            signal_quality=signal_quality,
            buffer_fill=len(self._motion_buffer) / self.buffer_size,
            inference_time_ms=(time.perf_counter() - start_time) * 1000,
            raw_motion_trace=np.array(self._motion_buffer) if self._motion_buffer else None
        )

    def reset(self) -> None:
        """Clear the motion buffer, previous-frame reference, and last result"""
        self._motion_buffer.clear()
        self._prev_chest_gray = None
        self._last_result = None
        logger.info("Respiration detector reset")

    def get_debug_info(self) -> dict:
        """Get debug information for logging"""
        return {
            "buffer_length": len(self._motion_buffer),
            "buffer_fill": len(self._motion_buffer) / self.buffer_size if self.buffer_size > 0 else 0,
            "fps": self.fps,
            "window_seconds": self.window_seconds
        }
