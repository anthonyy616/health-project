#Tests for Pupil Detection (Iris-Calibrated Pupil Segmentation)

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Ensure ✓ checkmarks print on Windows consoles with cp1252 default encoding
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2

from src.contactless.pupil_detection.iris_tracker import (
    RIGHT_IRIS_INDICES,
    LEFT_IRIS_INDICES,
    AVERAGE_IRIS_DIAMETER_MM,
    compute_iris_diameter_px,
    compute_px_per_mm,
    compute_eye_aspect_ratio,
    is_blinking,
    segment_pupil,
    compute_dilation_mm
)
from src.contactless.pupil_detection.detect import PupilDetector
from src.contactless.face_detection.detect import FaceResult, FaceDetector


# ---------------------------------------------------------------------------
# Synthetic helpers - no real face images or webcam needed
# ---------------------------------------------------------------------------

def place_iris_ring(landmarks, iris_indices, cx, cy, radius):
    """Place a center + 4-cardinal-point iris ring into the landmarks array."""
    landmarks[iris_indices[0], :2] = (cx, cy)
    for k, (dx, dy) in enumerate([(1, 0), (0, -1), (-1, 0), (0, 1)]):
        landmarks[iris_indices[1 + k], :2] = (cx + dx * radius, cy + dy * radius)


def place_eye_contour(landmarks, indices, cx, cy, eye_width, eye_height):
    """
    Place a synthetic eye contour: corners at +-eye_width/2 on the horizontal,
    remaining points alternating between upper/lower lid at +-eye_height/2.
    EAR for this geometry = eye_height / eye_width (see compute_eye_aspect_ratio).
    """
    pts = [(cx - eye_width / 2.0, cy), (cx + eye_width / 2.0, cy)]
    xs = np.linspace(
        cx - eye_width / 2.0 + eye_width / 8.0,
        cx + eye_width / 2.0 - eye_width / 8.0,
        len(indices) - 2
    )
    for i, x in enumerate(xs):
        y = cy - eye_height / 2.0 if i % 2 == 0 else cy + eye_height / 2.0
        pts.append((x, y))
    for idx, (x, y) in zip(indices, pts):
        landmarks[idx, 0] = x
        landmarks[idx, 1] = y


def build_synthetic_face(pupil_radius=5.0, iris_radius=12.0, eye_height=16.0,
                         frame_size=(480, 640)):
    """
    Build a synthetic two-eyed "face":

    - frame: mid-gray BGR image with a dark filled circle (the pupil) at
      each iris center.
    - landmarks: (478, 3) array with real iris ring geometry for both eyes
      and open-eye contours for both eyes.

    Image-left eye uses iris indices 468..472 (RIGHT_IRIS_INDICES per the
    MediaPipe-style naming - it sits inside FaceDetector.LEFT_EYE_LANDMARKS);
    image-right eye uses 473..477. See iris_tracker.py docstring.
    """
    h, w = frame_size
    frame = np.full((h, w, 3), 128, dtype=np.uint8)
    landmarks = np.zeros((478, 3), dtype=np.float64)

    left_center = (int(w * 0.35), int(h * 0.5))    # image-left eye
    right_center = (int(w * 0.65), int(h * 0.5))   # image-right eye

    # Iris ring geometry (the ring ORDER is irrelevant - the diameter
    # function is ordering-agnostic by design)
    place_iris_ring(landmarks, RIGHT_IRIS_INDICES, left_center[0], left_center[1], iris_radius)
    place_iris_ring(landmarks, LEFT_IRIS_INDICES, right_center[0], right_center[1], iris_radius)

    # Dark pupil circles in the frame at both iris centers
    for cx, cy in (left_center, right_center):
        cv2.circle(frame, (cx, cy), int(pupil_radius), (20, 20, 20), -1)

    # Open-eye contours for blink detection (EAR = eye_height / eye_width)
    eye_width = 40.0
    place_eye_contour(landmarks, FaceDetector.LEFT_EYE_LANDMARKS,
                      left_center[0], left_center[1], eye_width, eye_height)
    place_eye_contour(landmarks, FaceDetector.RIGHT_EYE_LANDMARKS,
                      right_center[0], right_center[1], eye_width, eye_height)

    return frame, landmarks


def build_synthetic_face_closed_eyes(pupil_radius=5.0, iris_radius=12.0,
                                     frame_size=(480, 640)):
    """Same geometry as build_synthetic_face but with collapsed (closed) eye contours."""
    frame, landmarks = build_synthetic_face(
        pupil_radius=pupil_radius, iris_radius=iris_radius,
        eye_height=2.0, frame_size=frame_size
    )
    return frame, landmarks


def make_face_result(landmarks):
    return FaceResult(
        detected=True,
        bbox=(100, 50, 440, 340),
        landmarks=landmarks,
        confidence=0.95
    )


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------

def test_compute_iris_diameter_px_known_geometry():
    """A ring of radius 10 must give a diameter of 20"""
    landmarks = np.zeros((478, 3), dtype=np.float64)
    place_iris_ring(landmarks, RIGHT_IRIS_INDICES, 100, 100, 10.0)

    diameter = compute_iris_diameter_px(landmarks, RIGHT_IRIS_INDICES)

    assert abs(diameter - 20.0) < 0.5, f"Expected ~20px diameter, got {diameter}"

    print("✓ test_compute_iris_diameter_px_known_geometry passed")


def test_compute_px_per_mm():
    """30px iris -> 30/11.7 px per mm; non-positive input handled gracefully"""
    px_per_mm = compute_px_per_mm(30.0)
    assert abs(px_per_mm - 30.0 / AVERAGE_IRIS_DIAMETER_MM) < 1e-9, \
        f"px_per_mm {px_per_mm} != 30/11.7"

    # Zero and negative input must not crash and must signal failure
    assert compute_px_per_mm(0.0) is None, "0px iris should return None"
    assert compute_px_per_mm(-5.0) is None, "negative iris should return None"

    print("✓ test_compute_px_per_mm passed")


def test_eye_aspect_ratio_open_vs_closed():
    """Closed-eye EAR must be meaningfully lower than open-eye EAR"""
    landmarks_open = np.zeros((478, 3), dtype=np.float64)
    landmarks_closed = np.zeros((478, 3), dtype=np.float64)

    # Same horizontal spread; open eye has vertical spread 16, closed has 2
    place_eye_contour(landmarks_open, FaceDetector.LEFT_EYE_LANDMARKS, 100, 100, 60.0, 16.0)
    place_eye_contour(landmarks_closed, FaceDetector.LEFT_EYE_LANDMARKS, 100, 100, 60.0, 2.0)

    ear_open = compute_eye_aspect_ratio(landmarks_open, FaceDetector.LEFT_EYE_LANDMARKS)
    ear_closed = compute_eye_aspect_ratio(landmarks_closed, FaceDetector.LEFT_EYE_LANDMARKS)

    assert ear_open > 0.2, f"Open-eye EAR {ear_open} should be above the blink threshold"
    assert ear_closed < 0.1, f"Closed-eye EAR {ear_closed} should be well below the threshold"
    assert ear_open > 2.0 * ear_closed, \
        f"EAR separation insufficient: open {ear_open} vs closed {ear_closed}"

    print("✓ test_eye_aspect_ratio_open_vs_closed passed")


def test_is_blinking_threshold():
    """EAR below threshold is a blink, above is not"""
    assert is_blinking(0.1) is True, "EAR 0.1 (below 0.2) should be a blink"
    assert is_blinking(0.3) is False, "EAR 0.3 (above 0.2) should not be a blink"
    assert is_blinking(0.2) is False, "EAR exactly at threshold should not be a blink (strict <)"

    print("✓ test_is_blinking_threshold passed")


def test_segment_pupil_on_synthetic_circle():
    """A drawn dark circle of radius 10 must be recovered within a few px"""
    crop = np.full((64, 64), 128, dtype=np.uint8)
    cv2.circle(crop, (32, 32), 10, 20, -1)  # dark filled circle, radius 10

    radius = segment_pupil(crop, expected_radius_px=12.0)

    assert radius is not None, "Dark circle should be detected"
    assert abs(radius - 10.0) <= 3.0, f"Radius {radius} not within ±3 of 10"

    print("✓ test_segment_pupil_on_synthetic_circle passed")


def test_segment_pupil_no_dark_region_returns_none():
    """Uniform gray image must not produce a spurious pupil"""
    crop = np.full((64, 64), 128, dtype=np.uint8)

    radius = segment_pupil(crop, expected_radius_px=12.0)

    assert radius is None, f"Uniform image should yield no pupil, got {radius}"

    print("✓ test_segment_pupil_no_dark_region_returns_none passed")


def test_compute_dilation_mm():
    """Radius 2px at 5 px/mm -> 0.8mm diameter; invalid scale handled"""
    mm = compute_dilation_mm(2.0, 5.0)
    assert abs(mm - 0.8) < 1e-9, f"Expected 0.8mm, got {mm}"

    assert compute_dilation_mm(2.0, None) is None, "None scale should return None"
    assert compute_dilation_mm(2.0, 0.0) is None, "Zero scale should return None"
    assert compute_dilation_mm(None, 5.0) is None, "None radius should return None"

    print("✓ test_compute_dilation_mm passed")


# ---------------------------------------------------------------------------
# Detector tests
# ---------------------------------------------------------------------------

def test_detector_blink_returns_cached_result():
    """A blink frame must return the cached reading, not a blank one"""
    detector = PupilDetector()
    frame, landmarks = build_synthetic_face()
    face_result = make_face_result(landmarks)

    # Two open-eye frames to establish a reading
    r1 = detector.detect(frame, face_result)
    r2 = detector.detect(frame, face_result)
    assert r1.average_mm is not None, "Open-eye frame should produce a reading"

    # Closed-eye frame -> blink
    _, closed_landmarks = build_synthetic_face_closed_eyes()
    closed_face = make_face_result(closed_landmarks)
    r3 = detector.detect(frame, closed_face)

    assert r3.is_blinking is True, "Closed-eye frame should be flagged as blinking"
    assert r3.average_mm is not None, "Blink frame must not blank the reading"
    assert abs(r3.average_mm - r2.average_mm) < 0.05, \
        f"Blink frame changed the cached average {r2.average_mm} -> {r3.average_mm}"
    assert r3.left_pupil_mm == r2.left_pupil_mm, "Left pupil should be cached"
    assert r3.right_pupil_mm == r2.right_pupil_mm, "Right pupil should be cached"

    print("✓ test_detector_blink_returns_cached_result passed")


def test_detector_baseline_establishment():
    """dilation_change stays None until the baseline is established, then ~0"""
    detector = PupilDetector(baseline_frames=5)
    frame, landmarks = build_synthetic_face()
    face_result = make_face_result(landmarks)

    # baseline_frames frames: baseline accumulates but delta stays None
    for i in range(5):
        r = detector.detect(frame, face_result)
        assert r.dilation_change is None, f"Frame {i} should have no dilation_change yet"

    # Next frame: baseline is frozen -> delta becomes numeric (~0, consistent readings)
    r = detector.detect(frame, face_result)
    assert r.dilation_change is not None, "dilation_change should appear after baseline"
    assert abs(r.dilation_change) < 0.5, \
        f"Consistent readings should give ~0 delta, got {r.dilation_change}"

    print("✓ test_detector_baseline_establishment passed")


def test_detector_no_face():
    """No face / no landmarks -> all mm fields None, is_blinking True"""
    detector = PupilDetector()

    # Case 1: detected=False
    no_face = FaceResult(detected=False, bbox=None, confidence=0.0, landmarks=None)
    r = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8), no_face)
    assert r.left_pupil_mm is None
    assert r.right_pupil_mm is None
    assert r.average_mm is None
    assert r.dilation_change is None
    assert r.is_blinking is True
    assert r.confidence == 0.0

    # Case 2: detected=True but landmarks missing
    no_landmarks = FaceResult(detected=True, bbox=(10, 10, 100, 100), confidence=0.9, landmarks=None)
    r2 = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8), no_landmarks)
    assert r2.average_mm is None
    assert r2.is_blinking is True

    print("✓ test_detector_no_face passed")


def test_detector_reset():
    """Reset clears EMA state, baseline accumulator/value, and cached result"""
    detector = PupilDetector(baseline_frames=5)
    frame, landmarks = build_synthetic_face()
    face_result = make_face_result(landmarks)

    for _ in range(10):
        detector.detect(frame, face_result)

    assert detector._left_ema is not None, "EMA should be populated before reset"
    assert detector._baseline_mm is not None, "Baseline should be set before reset"
    assert len(detector._baseline_readings) > 0
    assert detector._last_result is not None

    detector.reset()

    assert detector._left_ema is None, "Left EMA should be cleared"
    assert detector._right_ema is None, "Right EMA should be cleared"
    assert detector._baseline_mm is None, "Baseline value should be cleared"
    assert detector._baseline_readings == [], "Baseline accumulator should be empty"
    assert detector._last_result is None, "Cached result should be cleared"

    print("✓ test_detector_reset passed")


def run_all_tests():
    """Run all tests"""
    print("Running pupil detection tests...")
    print("-" * 30)

    test_compute_iris_diameter_px_known_geometry()
    test_compute_px_per_mm()
    test_eye_aspect_ratio_open_vs_closed()
    test_is_blinking_threshold()
    test_segment_pupil_on_synthetic_circle()
    test_segment_pupil_no_dark_region_returns_none()
    test_compute_dilation_mm()
    test_detector_blink_returns_cached_result()
    test_detector_baseline_establishment()
    test_detector_no_face()
    test_detector_reset()

    print("-" * 30)
    print("All tests passed! ✅")


if __name__ == "__main__":
    run_all_tests()
