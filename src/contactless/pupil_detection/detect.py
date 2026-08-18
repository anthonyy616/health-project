"""
Pupil Detector
Stateful class managing per-eye EMA smoothing of pupil dilation readings,
blink-gated updates, and a rolling dilation baseline for the
dilation_change metric. Calls the pure functions in iris_tracker.py in
order; no signal processing lives here.
"""

import numpy as np
import cv2
import time
from dataclasses import dataclass, replace
from typing import Optional, Tuple, List
import logging

from src.contactless.face_detection.detect import FaceResult, FaceDetector
from src.contactless.pupil_detection.iris_tracker import (
    RIGHT_IRIS_INDICES,
    LEFT_IRIS_INDICES,
    compute_iris_diameter_px,
    compute_px_per_mm,
    compute_eye_aspect_ratio,
    is_blinking,
    segment_pupil,
    compute_dilation_mm,
    MIN_PLAUSIBLE_PUPIL_RATIO,
    MAX_PLAUSIBLE_PUPIL_RATIO
)

logger = logging.getLogger(__name__)

# Confidence weighting constants (simple weighted combination of the two
# frame-level quality signals - documented, not magic numbers)
CONF_SEGMENTATION_WEIGHT = 0.5   # both eyes segmented successfully this frame
CONF_RATIO_WEIGHT = 0.5          # pupil:iris ratio inside the plausible range

# A real iris ring is never smaller than this in pixels - anything below is
# missing/collapsed landmarks, and the px/mm scale it would produce is
# garbage. Treat such an eye as unreadable (same path as segmentation failure).
MIN_VALID_IRIS_DIAMETER_PX = 5.0


@dataclass
class PupilResult:
    """Container for pupil dilation detection results"""
    left_pupil_mm: Optional[float]
    right_pupil_mm: Optional[float]
    average_mm: Optional[float]
    dilation_change: Optional[float]  # signed delta from the established baseline
    confidence: float
    is_blinking: bool
    inference_time_ms: float


class PupilDetector:
    """
    Iris-calibrated pupil dilation detector.

    NOTE on the left/right swap: FaceDetector names its eye contour lists
    from the camera view (LEFT_EYE_LANDMARKS = image-left eye), while the
    iris index constants in iris_tracker.py follow MediaPipe's
    subject-perspective naming (RIGHT_IRIS_INDICES = subject's right =
    image-left eye). The pairing below therefore maps:
        image-left eye  (FaceDetector "left", shown left on screen)
                        -> RIGHT_IRIS_INDICES (468..472)
        image-right eye (FaceDetector "right")
                        -> LEFT_IRIS_INDICES (473..477)
    Verified empirically against real FaceDetector output (see
    iris_tracker.py module docstring). If a live run shows the L/R labels
    swapped relative to the camera feed, swap the two iris index lists here.
    """

    def __init__(
        self,
        ema_alpha: float = 0.3,
        blink_ear_threshold: float = 0.2,
        baseline_frames: int = 30,
        eye_crop_size: Tuple[int, int] = (64, 64)
    ):
        """
        Initialize the pupil detector.

        Args:
            ema_alpha: smoothing factor for per-frame mm readings. Higher =
                more responsive, lower = smoother but laggier.
            blink_ear_threshold: average-EAR below which the frame counts
                as a blink and segmentation is skipped.
            baseline_frames: number of initial valid (non-blinking) readings
                averaged to establish the dilation baseline.
            eye_crop_size: (width, height) crop around each iris center for
                the pupil segmentation step (cropped as a square of
                min(width, height) so the iris stays centered).
        """
        self.ema_alpha = ema_alpha
        self.blink_ear_threshold = blink_ear_threshold
        self.baseline_frames = baseline_frames
        self.eye_crop_size = eye_crop_size

        self._left_ema: Optional[float] = None
        self._right_ema: Optional[float] = None
        self._baseline_readings: List[float] = []
        self._baseline_mm: Optional[float] = None
        self._last_result: Optional[PupilResult] = None

    def detect(self, frame: np.ndarray, face_result: FaceResult) -> PupilResult:
        """
        Process a frame and update pupil dilation estimation.

        Args:
            frame: BGR image from OpenCV (needed for pixel-level pupil
                segmentation - landmarks alone are not enough, hence the
                two-argument signature like the respiration module)
            face_result: face detection result with landmarks

        Returns:
            PupilResult with current estimation
        """
        start_time = time.perf_counter()

        # No face / no landmarks - treat as "no valid reading" (equivalent
        # to blinking for display purposes).
        if not face_result.detected or face_result.landmarks is None:
            result = self._build_neutral_result(start_time)
            self._last_result = result
            return result

        landmarks = face_result.landmarks

        # Blink detection: AVERAGE EAR across both eyes (not "either eye")
        # so a single-eye wink does not invalidate the whole reading.
        left_ear = compute_eye_aspect_ratio(landmarks, FaceDetector.LEFT_EYE_LANDMARKS)
        right_ear = compute_eye_aspect_ratio(landmarks, FaceDetector.RIGHT_EYE_LANDMARKS)
        avg_ear = (left_ear + right_ear) / 2.0
        blinking = is_blinking(avg_ear, self.blink_ear_threshold)

        if blinking:
            if self._last_result is not None:
                # Return the cached reading (marked as blinking) so the UI
                # does not flash to "--" every blink.
                return replace(
                    self._last_result,
                    is_blinking=True,
                    inference_time_ms=(time.perf_counter() - start_time) * 1000
                )
            result = self._build_neutral_result(start_time)
            self._last_result = result
            return result

        # Per-eye: iris ring -> px/mm scale -> crop -> pupil segmentation
        left_mm, left_seg_ok, left_ratio_ok = self._process_eye(
            frame, landmarks, RIGHT_IRIS_INDICES, start_time
        )
        right_mm, right_seg_ok, right_ratio_ok = self._process_eye(
            frame, landmarks, LEFT_IRIS_INDICES, start_time
        )

        # Update EMAs (keep previous value when this frame's reading is
        # unavailable - a single bad frame must not blank the display).
        if left_mm is not None:
            self._left_ema = self._ema_update(self._left_ema, left_mm)
        if right_mm is not None:
            self._right_ema = self._ema_update(self._right_ema, right_mm)

        # Average over whichever eyes are available
        available = [v for v in (self._left_ema, self._right_ema) if v is not None]
        average_mm = float(np.mean(available)) if available else None

        # Dilation baseline: freeze the mean of the first baseline_frames
        # valid readings, then report the signed delta afterwards.
        dilation_change: Optional[float] = None
        if average_mm is not None:
            if self._baseline_mm is None:
                self._baseline_readings.append(average_mm)
                if len(self._baseline_readings) >= self.baseline_frames:
                    self._baseline_mm = float(np.mean(self._baseline_readings))
            else:
                dilation_change = average_mm - self._baseline_mm

        confidence = self._compute_confidence(
            left_seg_ok, right_seg_ok, left_ratio_ok, right_ratio_ok, average_mm
        )

        result = PupilResult(
            left_pupil_mm=self._left_ema,
            right_pupil_mm=self._right_ema,
            average_mm=average_mm,
            dilation_change=dilation_change,
            confidence=confidence,
            is_blinking=False,
            inference_time_ms=(time.perf_counter() - start_time) * 1000
        )
        self._last_result = result
        return result

    def _process_eye(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        iris_indices: list,
        start_time: float
    ) -> Tuple[Optional[float], bool, bool]:
        """
        Process a single eye: derive the px/mm scale from its iris ring,
        crop around the iris center, segment the pupil, and convert to mm.

        Returns:
            (dilation_mm, segmented_ok, ratio_ok)
            - dilation_mm: None when segmentation failed or the scale is
              invalid (the caller falls back to the last valid EMA)
            - segmented_ok: pupil found this frame
            - ratio_ok: pupil:iris ratio inside the plausible range
        """
        iris_diameter_px = compute_iris_diameter_px(landmarks, iris_indices)
        px_per_mm = compute_px_per_mm(iris_diameter_px)
        if px_per_mm is None or iris_diameter_px < MIN_VALID_IRIS_DIAMETER_PX:
            return None, False, False

        # Square crop centered on the iris center landmark
        cx, cy = landmarks[iris_indices[0], :2].astype(int)
        half = min(self.eye_crop_size) // 2
        h, w = frame.shape[:2]
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)
        if x2 <= x1 or y2 <= y1:
            return None, False, False

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, False, False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Upper bound on pupil radius: pupil can be at most
        # MAX_PLAUSIBLE_PUPIL_RATIO * iris radius
        expected_radius_px = (iris_diameter_px / 2.0) * MAX_PLAUSIBLE_PUPIL_RATIO

        pupil_radius_px = segment_pupil(gray, expected_radius_px)
        if pupil_radius_px is None:
            return None, False, False

        dilation_mm = compute_dilation_mm(pupil_radius_px, px_per_mm)
        if dilation_mm is None:
            return None, False, False

        pupil_iris_ratio = (pupil_radius_px * 2.0) / iris_diameter_px
        ratio_ok = MIN_PLAUSIBLE_PUPIL_RATIO <= pupil_iris_ratio <= MAX_PLAUSIBLE_PUPIL_RATIO

        return dilation_mm, True, ratio_ok

    def _ema_update(self, ema: Optional[float], new_value: float) -> float:
        """EMA update - first reading primes the filter directly."""
        if ema is None:
            return new_value
        return self.ema_alpha * new_value + (1.0 - self.ema_alpha) * ema

    def _compute_confidence(
        self,
        left_seg_ok: bool,
        right_seg_ok: bool,
        left_ratio_ok: bool,
        right_ratio_ok: bool,
        average_mm: Optional[float]
    ) -> float:
        """
        Frame confidence = weighted combination of:
          (a) whether both eyes segmented successfully this frame, and
          (b) whether each segmented eye's pupil:iris ratio fell inside
              [MIN_PLAUSIBLE_PUPIL_RATIO, MAX_PLAUSIBLE_PUPIL_RATIO] (a
              ratio outside that range means segmentation grabbed the
              wrong blob).
        """
        if average_mm is None:
            return 0.0

        if left_seg_ok and right_seg_ok:
            seg_score = 1.0
        elif left_seg_ok or right_seg_ok:
            seg_score = 0.5
        else:
            seg_score = 0.0

        ratio_scores = []
        if left_seg_ok:
            ratio_scores.append(1.0 if left_ratio_ok else 0.0)
        if right_seg_ok:
            ratio_scores.append(1.0 if right_ratio_ok else 0.0)
        ratio_score = float(np.mean(ratio_scores)) if ratio_scores else 0.0

        return float(np.clip(
            CONF_SEGMENTATION_WEIGHT * seg_score + CONF_RATIO_WEIGHT * ratio_score,
            0.0, 1.0
        ))

    def _build_neutral_result(self, start_time: float) -> PupilResult:
        """A blank result for no-face / no-landmark / first-blink frames"""
        return PupilResult(
            left_pupil_mm=None,
            right_pupil_mm=None,
            average_mm=None,
            dilation_change=None,
            confidence=0.0,
            is_blinking=True,
            inference_time_ms=(time.perf_counter() - start_time) * 1000
        )

    def reset(self) -> None:
        """
        Full reset: clear EMA state, baseline accumulator/value, and the
        cached result. Use when the face has been lost for an extended
        period.
        """
        self._left_ema = None
        self._right_ema = None
        self._baseline_readings = []
        self._baseline_mm = None
        self._last_result = None
        logger.info("Pupil detector reset")

    def reset_baseline(self) -> None:
        """
        Re-establish the dilation baseline WITHOUT a full reset. Keeps the
        EMA-smoothed readings; only the baseline accumulator and value are
        cleared so dilation_change starts over. Useful when lighting
        changes mid-session and the original baseline is no longer valid
        (bound to the 'b' key in --mode pupil).
        """
        self._baseline_readings = []
        self._baseline_mm = None
        logger.info("Pupil baseline reset (EMA readings kept)")

    def get_debug_info(self) -> dict:
        """Get debug information for logging"""
        return {
            "left_ema": self._left_ema,
            "right_ema": self._right_ema,
            "baseline_mm": self._baseline_mm,
            "baseline_frames_collected": len(self._baseline_readings),
        }
