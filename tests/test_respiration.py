#Tests for Respiratory Rate Detection (Chest Motion Optical Flow)

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

from src.contactless.respiration.motion_analysis import (
    extract_chest_bbox,
    detrend_signal,
    bandpass_filter_respiration,
    detect_breathing_peaks,
    compute_bpm_from_peaks,
    compute_motion_quality
)
from src.contactless.respiration.detect import RespirationDetector
from src.contactless.face_detection.detect import FaceResult


def test_extract_chest_bbox_within_frame():
    """Chest bbox derived from a centered face stays within frame bounds"""
    frame_shape = (480, 640)
    face_bbox = (200, 100, 200, 200)  # near vertical center

    bbox = extract_chest_bbox(face_bbox, frame_shape)

    assert bbox is not None, "Chest bbox should not be None for a centered face"
    x, y, w, h = bbox
    assert w > 0 and h > 0
    assert x >= 0 and y >= 0
    assert x + w <= frame_shape[1], f"Chest right edge {x + w} exceeds frame width"
    assert y + h <= frame_shape[0], f"Chest bottom edge {y + h} exceeds frame height"

    print("✓ test_extract_chest_bbox_within_frame passed")


def test_extract_chest_bbox_clips_at_edge():
    """Chest bbox near the frame edge either clips or returns None, never out-of-bounds"""
    frame_shape = (240, 320)

    # Case 1: face near bottom, naive chest region partially past frame -> clips
    face_bbox_clip = (60, 120, 100, 100)
    bbox = extract_chest_bbox(face_bbox_clip, frame_shape)
    if bbox is not None:
        x, y, w, h = bbox
        assert w > 0 and h > 0
        assert x >= 0 and y >= 0
        assert x + w <= frame_shape[1]
        assert y + h <= frame_shape[0]

    # Case 2: face at very bottom edge, chest region fully below frame -> None
    face_bbox_none = (60, 200, 100, 40)
    bbox = extract_chest_bbox(face_bbox_none, frame_shape)
    assert bbox is None, "Chest bbox fully below the frame should return None"

    print("✓ test_extract_chest_bbox_clips_at_edge passed")


def test_detrend_removes_linear_trend():
    """Detrending a linear ramp leaves a near-zero slope"""
    np.random.seed(42)
    N = 600
    t = np.arange(N, dtype=np.float64)
    signal = (10.0 / N) * t + np.random.normal(0, 0.1, N)  # linear ramp 0->10 + noise

    detrended = detrend_signal(signal)

    orig_slope = np.polyfit(t, signal, 1)[0]
    new_slope = np.polyfit(t, detrended, 1)[0]

    assert abs(new_slope) < abs(orig_slope) * 0.1, \
        f"Detrended slope {new_slope} not much smaller than original {orig_slope}"

    print("✓ test_detrend_removes_linear_trend passed")


def test_bandpass_isolates_respiration_band():
    """Bandpass keeps the in-band sine and removes the DC offset"""
    fps = 30
    N = 600  # 20 seconds
    t = np.linspace(0, N / fps, N)
    offset = 2.0
    signal = offset + np.sin(2 * np.pi * 0.2 * t)  # 0.2 Hz = 12 BPM, in-band

    filtered = bandpass_filter_respiration(signal, fps)

    assert abs(np.mean(filtered)) < abs(np.mean(signal)), \
        "Filtered mean should be closer to 0 than the original DC offset"
    assert np.ptp(filtered) > 0.5, \
        "In-band signal should survive the filter (peak-to-peak not collapsed)"

    print("✓ test_bandpass_isolates_respiration_band passed")


def test_peak_detection_clean_signal():
    """Full chain recovers ~12 BPM from a clean 0.2 Hz sine"""
    fps = 30
    N = 600
    t = np.linspace(0, N / fps, N)
    signal = 2.0 + np.sin(2 * np.pi * 0.2 * t)

    detrended = detrend_signal(signal)
    filtered = bandpass_filter_respiration(detrended, fps)
    peaks = detect_breathing_peaks(filtered, fps)
    bpm, confidence = compute_bpm_from_peaks(peaks, fps)

    assert bpm is not None, "Clean sine should produce a BPM"
    assert abs(bpm - 12.0) <= 2.0, f"BPM {bpm} not within ±2 of 12"
    assert confidence > 0.5, f"Confidence {confidence} not above 0.5 for clean signal"

    print("✓ test_peak_detection_clean_signal passed")


def test_peak_detection_pure_noise():
    """Random noise produces irregular peaks -> low confidence"""
    np.random.seed(7)
    fps = 30
    N = 600
    signal = np.random.normal(0, 1.0, N)

    detrended = detrend_signal(signal)
    filtered = bandpass_filter_respiration(detrended, fps)
    peaks = detect_breathing_peaks(filtered, fps)
    bpm, confidence = compute_bpm_from_peaks(peaks, fps)

    # Do not assert a specific BPM value
    assert confidence < 0.5, f"Confidence {confidence} not below 0.5 for noise"

    print("✓ test_peak_detection_pure_noise passed")


def test_motion_quality_breathing_amplitude():
    """A breathing-amplitude motion trace must clear the quality threshold"""
    # Sine with std ~0.07 simulates a real chest motion trace (mean vertical
    # flow ~0.1 px/frame). Regression test for the 2026-08-17 constant retune:
    # the old 0.01->2.0 mapping made the 0.1 threshold unreachable in practice.
    fps = 30
    N = 600  # 20 seconds
    t = np.linspace(0, N / fps, N)
    motion_trace = 0.1 * np.sin(2 * np.pi * 0.2 * t)  # 0.2 Hz = 12 BPM

    quality = compute_motion_quality(motion_trace)

    assert quality >= 0.1, f"Breathing-amplitude trace quality {quality} should clear the 0.1 threshold"

    print("✓ test_motion_quality_breathing_amplitude passed")


def test_detector_static_scene_low_quality():
    """A static scene produces no motion -> low quality and bpm None"""
    detector = RespirationDetector(fps=30, window_seconds=20, min_seconds=10.0)

    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    face_result = FaceResult(
        detected=True,
        bbox=(100, 100, 200, 200),
        confidence=0.95
    )

    result = None
    for _ in range(320):  # fills past min_seconds (300 samples)
        result = detector.add_frame(frame, face_result)

    assert result.bpm is None, "Static scene should not produce a BPM"
    assert result.signal_quality < detector.quality_threshold, \
        f"Static scene quality {result.signal_quality} should be below threshold"

    print("✓ test_detector_static_scene_low_quality passed")


def test_detector_no_face():
    """No face detected -> bpm None and buffer fill 0.0"""
    detector = RespirationDetector(fps=30)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    face_result = FaceResult(detected=False, bbox=None, confidence=0.0)

    result = detector.add_frame(frame, face_result)

    assert result.bpm is None, "BPM should be None when no face detected"
    assert result.buffer_fill == 0.0, f"Buffer fill should be 0.0, got {result.buffer_fill}"
    assert detector._prev_chest_gray is None

    print("✓ test_detector_no_face passed")


def test_detector_reset():
    """Reset clears the motion buffer and previous-frame reference"""
    detector = RespirationDetector(fps=30)
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    face_result = FaceResult(detected=True, bbox=(100, 100, 200, 200), confidence=0.95)

    for _ in range(100):
        detector.add_frame(frame, face_result)

    assert len(detector._motion_buffer) > 0, "Buffer should have content before reset"
    assert detector._prev_chest_gray is not None

    detector.reset()

    assert len(detector._motion_buffer) == 0, "Buffer should be empty after reset"
    assert detector._prev_chest_gray is None

    print("✓ test_detector_reset passed")


def test_add_frame_signature_takes_frame_and_face_result():
    """add_frame must require both frame and face_result (unlike rPPG's single arg)"""
    detector = RespirationDetector(fps=30)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    face_result = FaceResult(detected=True, bbox=(100, 100, 200, 200), confidence=0.95)

    # Two positional args must work
    result = detector.add_frame(frame, face_result)
    assert result is not None

    # One positional arg must fail - guards against copying the rPPG signature
    try:
        detector.add_frame(frame)
        assert False, "add_frame should require both frame and face_result"
    except TypeError:
        pass

    print("✓ test_add_frame_signature_takes_frame_and_face_result passed")


def run_all_tests():
    """Run all tests"""
    print("Running respiration tests...")
    print("-" * 30)

    test_extract_chest_bbox_within_frame()
    test_extract_chest_bbox_clips_at_edge()
    test_detrend_removes_linear_trend()
    test_bandpass_isolates_respiration_band()
    test_peak_detection_clean_signal()
    test_peak_detection_pure_noise()
    test_motion_quality_breathing_amplitude()
    test_detector_static_scene_low_quality()
    test_detector_no_face()
    test_detector_reset()
    test_add_frame_signature_takes_frame_and_face_result()

    print("-" * 30)
    print("All tests passed! ✅")


if __name__ == "__main__":
    run_all_tests()
