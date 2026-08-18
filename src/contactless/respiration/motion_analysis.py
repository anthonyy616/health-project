"""
Motion Analysis Functions for Respiratory Rate Detection
Pure functions that operate on numpy arrays / OpenCV array ops only - no camera, no buffering, no class state.
"""

import numpy as np
import cv2
import scipy.signal
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Physiological constants for respiration
RESP_MIN_HZ = 0.1  # 6 breaths/min
RESP_MAX_HZ = 0.5  # 30 breaths/min
RESP_MIN_BPM = 6
RESP_MAX_BPM = 30

# Motion quality mapping constants
# Retuned 2026-08-17 from first webcam debug run: observed mean vertical-flow
# values were ~0.0 to -0.3 (a breathing chest produces a mean flow well under
# 1 px/frame), so the original 0.01->2.0 range made the 0.1 quality threshold
# nearly unreachable. 0.001 is the noise floor, 0.5 is a deep-breathing trace.
MIN_MOTION_STD = 0.001
MAX_MOTION_STD = 0.5

# Minimum samples for filtfilt (roughly 3 * max(len(a), len(b)) for order 3 filter, with margin)
MIN_FILTER_SAMPLES = 30

# Standard Farneback optical flow parameters - do not tune without a reason
FARNEBACK_PARAMS = (0.5, 3, 15, 3, 5, 1.2, 0)


def extract_chest_bbox(
    face_bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int],
    width_ratio: float = 1.2,
    height_ratio: float = 1.5,
    gap_ratio: float = 0.15
) -> Optional[Tuple[int, int, int, int]]:
    """
    Derive a chest ROI bounding box below the face bounding box.

    The chest region starts below the face bbox with a small gap
    (gap_ratio * face_h), is width_ratio * face_w wide (centered on the
    face's horizontal center), and height_ratio * face_h tall. All four
    edges are clipped to the frame bounds.

    Args:
        face_bbox: (x, y, w, h) of the detected face
        frame_shape: (H, W) from frame.shape[:2]
        width_ratio: chest width as a multiple of face width
        height_ratio: chest height as a multiple of face height
        gap_ratio: gap below the face as a multiple of face height

    Returns:
        (x, y, w, h) chest bbox clipped to frame bounds, or None if the
        box has zero width or height after clipping (face near frame edge).
    """
    fx, fy, fw, fh = face_bbox
    frame_h, frame_w = frame_shape

    if fw <= 0 or fh <= 0:
        return None

    gap = gap_ratio * fh
    chest_w = width_ratio * fw
    chest_h = height_ratio * fh

    # Center chest horizontally on the face's horizontal center
    face_center_x = fx + fw / 2.0
    chest_x = face_center_x - chest_w / 2.0
    chest_y = fy + fh + gap

    # Clip to frame bounds
    x1 = max(0, int(round(chest_x)))
    y1 = max(0, int(round(chest_y)))
    x2 = min(frame_w, int(round(chest_x + chest_w)))
    y2 = min(frame_h, int(round(chest_y + chest_h)))

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2 - x1, y2 - y1)


def crop_and_resize_roi(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    target_size: Tuple[int, int] = (128, 128)
) -> np.ndarray:
    """
    Crop a bbox from a BGR frame and return a fixed-size grayscale ROI.

    Optical flow only needs intensity, not color, so convert to grayscale
    before resizing. A fixed target size matters because Farneback flow
    between frames requires identical dimensions, and a stable size keeps
    the flow magnitude scale consistent regardless of subject distance.

    Args:
        frame: BGR image from OpenCV
        bbox: (x, y, w, h) region to crop
        target_size: (width, height) to resize the crop to

    Returns:
        Grayscale ROI of shape target_size[::-1], dtype uint8
    """
    x, y, w, h = bbox
    crop = frame[y:y + h, x:x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, target_size)


def compute_vertical_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """
    Compute the mean vertical (y) component of dense optical flow between
    two grayscale frames of identical shape.

    Sign convention: in image coordinates the y-axis grows downward, so a
    rising chest (camera-space upward motion) produces negative flow values.
    This sign does not affect BPM calculation (frequency is sign-independent)
    but matters when interpreting debug output.

    Args:
        prev_gray: previous grayscale ROI
        curr_gray: current grayscale ROI

    Returns:
        Mean of the vertical flow component across the ROI
    """
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, *FARNEBACK_PARAMS)
    return float(flow[..., 1].mean())


def detrend_signal(signal: np.ndarray) -> np.ndarray:
    """
    Remove the linear trend from a 1D motion signal.

    Chest ROI drift (posture shift, slow camera exposure changes) adds a
    low-frequency trend that the bandpass filter alone may not fully remove
    given the very low cutoff (0.1 Hz) needed for respiration.

    Args:
        signal: 1D motion signal array

    Returns:
        Same-length array with the linear trend removed
    """
    return scipy.signal.detrend(signal)


def bandpass_filter_respiration(
    signal: np.ndarray,
    fps: float,
    low_hz: float = RESP_MIN_HZ,
    high_hz: float = RESP_MAX_HZ
) -> np.ndarray:
    """
    Apply a 3rd-order Butterworth bandpass filter for the respiration band.

    Order 3 is deliberately chosen (not the rPPG module's order 4): at these
    very low normalized frequencies relative to a 30fps sample rate, a 4th
    order filter is more prone to numerical instability.

    Args:
        signal: 1D signal array
        fps: frames per second
        low_hz: low frequency cutoff (default 0.1 Hz = 6 BPM)
        high_hz: high frequency cutoff (default 0.5 Hz = 30 BPM)

    Returns:
        Filtered 1D signal array, same length
    """
    if len(signal) < MIN_FILTER_SAMPLES:
        logger.warning(
            f"Signal too short for filtering ({len(signal)} samples, need >= {MIN_FILTER_SAMPLES}), returning unchanged"
        )
        return signal

    nyquist = fps / 2.0
    low_norm = low_hz / nyquist
    high_norm = high_hz / nyquist

    if low_norm >= high_norm or low_norm <= 0 or high_norm >= 1:
        logger.warning(f"Invalid filter parameters: low_norm={low_norm}, high_norm={high_norm}")
        return signal

    b, a = scipy.signal.butter(3, [low_norm, high_norm], btype="band")
    return scipy.signal.filtfilt(b, a, signal)


def detect_breathing_peaks(
    filtered_signal: np.ndarray,
    fps: float,
    min_bpm: float = RESP_MIN_BPM,
    max_bpm: float = RESP_MAX_BPM
) -> np.ndarray:
    """
    Detect breathing peaks in a filtered respiration signal.

    Args:
        filtered_signal: bandpass-filtered 1D signal
        fps: frames per second
        min_bpm: minimum physiological breathing rate
        max_bpm: maximum physiological breathing rate

    Returns:
        1D array of peak sample indices (empty if none found)
    """
    distance = int(fps * 60.0 / max_bpm)  # minimum samples between peaks
    prominence = 0.3 * np.std(filtered_signal)

    peaks, _ = scipy.signal.find_peaks(
        filtered_signal,
        distance=distance,
        prominence=prominence
    )
    return peaks


def compute_bpm_from_peaks(
    peak_indices: np.ndarray,
    fps: float
) -> Tuple[Optional[float], float]:
    """
    Compute breathing rate (BPM) and regularity-based confidence from peaks.

    Args:
        peak_indices: 1D array of peak sample indices
        fps: frames per second

    Returns:
        (bpm, confidence): bpm is None if fewer than 2 peaks; confidence is
        0.0-1.0 based on how evenly spaced the peaks are
    """
    if len(peak_indices) < 2:
        return None, 0.0

    intervals = np.diff(peak_indices).astype(np.float64)
    mean_interval = float(np.mean(intervals))

    if mean_interval == 0:
        return None, 0.0

    bpm = 60.0 * fps / mean_interval
    confidence = float(np.clip(1.0 - (np.std(intervals) / mean_interval), 0.0, 1.0))

    return bpm, confidence


def compute_motion_quality(motion_trace: np.ndarray) -> float:
    """
    Compute a 0.0-1.0 signal quality score from the raw motion trace.

    Unlike an RGB trace, the motion signal's mean is not physically
    meaningful, so raw standard deviation is used directly instead of the
    coefficient of variation. The std range is mapped linearly and clamped
    to [0, 1]; the MIN_MOTION_STD / MAX_MOTION_STD constants are a first
    guess and should be retuned after seeing real webcam motion data.

    Args:
        motion_trace: raw (un-detrended, un-filtered) 1D motion signal

    Returns:
        Float 0.0-1.0 representing signal quality
    """
    if len(motion_trace) == 0:
        return 0.0

    std_val = float(np.std(motion_trace))
    quality = np.clip((std_val - MIN_MOTION_STD) / (MAX_MOTION_STD - MIN_MOTION_STD), 0.0, 1.0)
    return float(quality)
