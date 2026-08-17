
#Tests for rPPG Heart Rate Detection

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.contactless.heart_rate.signal_processing import (
    normalize_rgb_trace,
    apply_pos_algorithm,
    bandpass_filter,
    extract_bpm_from_fft,
    compute_signal_quality
)
from src.contactless.heart_rate.rppg import HeartRateDetector
from src.contactless.face_detection.detect import FaceResult


def test_normalize_rgb_trace():
    """Test RGB trace normalization"""
    # Create synthetic RGB trace with constant values plus noise
    N = 300
    rgb_trace = np.array([
        [100 + np.random.normal(0, 1), 80 + np.random.normal(0, 1), 60 + np.random.normal(0, 1)]
        for _ in range(N)
    ])
    
    normalized = normalize_rgb_trace(rgb_trace)
    
    # Check shape preservation
    assert normalized.shape == rgb_trace.shape
    
    # Check that each channel has mean approximately 0
    for i in range(3):
        channel_mean = np.mean(normalized[:, i])
        assert abs(channel_mean) < 1e-6, f"Channel {i} mean is {channel_mean}, expected ~0"
    
    print("✓ test_normalize_rgb_trace passed")


def test_pos_output_shape():
    """Test POS algorithm output shape and normalization"""
    # Create normalized RGB trace
    N = 300
    normalized_rgb = np.random.normal(0, 1, (N, 3))
    
    pulse_signal = apply_pos_algorithm(normalized_rgb)
    
    # Check output shape
    assert pulse_signal.shape == (N,), f"Expected shape ({N},), got {pulse_signal.shape}"
    
    # Check output normalization
    mean_val = np.mean(pulse_signal)
    std_val = np.std(pulse_signal)
    
    assert abs(mean_val) < 1e-6, f"Mean is {mean_val}, expected ~0"
    assert abs(std_val - 1.0) < 1e-6, f"Std is {std_val}, expected ~1"
    
    print("✓ test_pos_output_shape passed")


def test_bandpass_removes_dc():
    """Test that bandpass filter removes DC component"""
    # Create signal with DC component plus sine wave
    N = 300
    fps = 30
    t = np.linspace(0, N/fps, N)
    
    dc_component = 5.0
    sine_wave = np.sin(2 * np.pi * 1.2 * t)  # 1.2 Hz sine wave
    signal = dc_component + sine_wave
    
    filtered = bandpass_filter(signal, fps)
    
    # Filtered signal should have mean closer to 0
    original_mean = np.mean(signal)
    filtered_mean = np.mean(filtered)
    
    assert abs(filtered_mean) < abs(original_mean), \
        f"Filtered mean {filtered_mean} not closer to 0 than original {original_mean}"
    
    print("✓ test_bandpass_removes_dc passed")


def test_bpm_extraction_clean_signal():
    """Test BPM extraction from clean signal"""
    # Generate clean pulse signal at 1.2 Hz (72 BPM)
    N = 300
    fps = 30
    t = np.linspace(0, N/fps, N)
    
    signal = np.sin(2 * np.pi * 1.2 * t)
    
    bpm, confidence = extract_bpm_from_fft(signal, fps)
    
    # Check BPM is within ±3 of 72
    expected_bpm = 72.0
    assert abs(bpm - expected_bpm) <= 3, f"BPM {bpm} not within ±3 of {expected_bpm}"
    
    # Check confidence is above 0.5 for clean signal
    assert confidence > 0.5, f"Confidence {confidence} not above 0.5 for clean signal"
    
    print("✓ test_bpm_extraction_clean_signal passed")


def test_bpm_extraction_pure_noise():
    """Test BPM extraction from pure noise"""
    # Generate random noise
    N = 300
    fps = 30
    signal = np.random.normal(0, 1, N)
    
    bpm, confidence = extract_bpm_from_fft(signal, fps)
    
    # Confidence should be low for noise
    assert confidence < 0.5, f"Confidence {confidence} not below 0.5 for noise"
    
    print("✓ test_bpm_extraction_pure_noise passed")


def test_detector_fills_buffer():
    """Test that detector fills buffer correctly"""
    fps = 30
    window_seconds = 10
    min_seconds = 5.0
    
    detector = HeartRateDetector(fps=fps, window_seconds=window_seconds, min_seconds=min_seconds)
    
    # Create mock FaceResult with fake forehead ROI
    fake_roi = np.random.randint(0, 255, (32, 64, 3), dtype=np.uint8)
    face_result = FaceResult(
        detected=True,
        bbox=(100, 100, 200, 200),
        confidence=0.95,
        face_roi=None,  # Not used for heart rate
        forehead_roi=fake_roi,
        landmarks=None,
        detection_time_ms=10.0
    )
    
    # Add frames up to min_seconds threshold - 1
    min_frames = int(min_seconds * fps)  # 150 frames
    for i in range(min_frames - 1):
        result = detector.add_frame(face_result)
        assert result.bpm is None, f"BPM should be None before min_frames (frame {i})"
    
    # Add one more frame to reach min_seconds threshold
    result = detector.add_frame(face_result)
    
    # Buffer should now be at min_seconds
    buffer_fill = result.buffer_fill
    assert buffer_fill >= 0.5, f"Buffer fill {buffer_fill} should be >= 0.5"
    
    print("✓ test_detector_fills_buffer passed")


def test_detector_no_face():
    """Test detector behavior when no face detected"""
    detector = HeartRateDetector(fps=30, window_seconds=10)
    
    # Create FaceResult with no face detected
    face_result = FaceResult(
        detected=False,
        bbox=None,
        confidence=0.0,
        face_roi=None,
        forehead_roi=None,
        landmarks=None,
        detection_time_ms=5.0
    )
    
    result = detector.add_frame(face_result)
    
    assert result.bpm is None, "BPM should be None when no face detected"
    assert result.buffer_fill == 0.0, f"Buffer fill should be 0.0, got {result.buffer_fill}"
    
    print("✓ test_detector_no_face passed")


def test_detector_reset():
    """Test detector reset functionality"""
    detector = HeartRateDetector(fps=30, window_seconds=10)
    
    # Add some frames
    fake_roi = np.random.randint(0, 255, (32, 64, 3), dtype=np.uint8)
    face_result = FaceResult(
        detected=True,
        bbox=(100, 100, 200, 200),
        confidence=0.95,
        face_roi=None,
        forehead_roi=fake_roi,
        landmarks=None,
        detection_time_ms=10.0
    )
    
    for _ in range(200):
        detector.add_frame(face_result)
    
    # Check buffer has content
    assert len(detector._rgb_buffer) > 0, "Buffer should have content before reset"
    
    # Reset detector
    detector.reset()
    
    # Check buffer is empty
    assert len(detector._rgb_buffer) == 0, "Buffer should be empty after reset"
    
    print("✓ test_detector_reset passed")


def run_all_tests():
    """Run all tests"""
    print("Running rPPG tests...")
    print("-" * 30)
    
    test_normalize_rgb_trace()
    test_pos_output_shape()
    test_bandpass_removes_dc()
    test_bpm_extraction_clean_signal()
    test_bpm_extraction_pure_noise()
    test_detector_fills_buffer()
    test_detector_no_face()
    test_detector_reset()
    
    print("-" * 30)
    print("All tests passed! ✅")


if __name__ == "__main__":
    run_all_tests()
