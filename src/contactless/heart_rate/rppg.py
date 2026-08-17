"""
rPPG Heart Rate Detector
Stateful class that manages the rolling frame buffer and processes rPPG signals
"""

import numpy as np
import cv2
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

from src.contactless.face_detection.detect import FaceResult
from src.contactless.heart_rate.signal_processing import (
    normalize_rgb_trace,
    apply_pos_algorithm,
    bandpass_filter,
    extract_bpm_from_fft,
    compute_signal_quality
)

logger = logging.getLogger(__name__)


@dataclass
class HeartRateResult:
    """Container for heart rate detection results"""
    bpm: Optional[float]
    confidence: float
    signal_quality: float
    buffer_fill: float
    inference_time_ms: float
    raw_rgb_trace: Optional[np.ndarray]


class HeartRateDetector:
    """
    rPPG-based heart rate detector.
    
    Manages a rolling buffer of RGB values from the forehead ROI and applies
    signal processing to extract heart rate.
    """
    
    def __init__(
        self,
        fps: int = 30,
        window_seconds: int = 10,
        min_seconds: float = 5.0,
        quality_threshold: float = 0.1
    ):
        """
        Initialize the heart rate detector.
        
        Args:
            fps: Camera frames per second
            window_seconds: Length of rolling buffer in seconds
            min_seconds: Minimum buffer fill before attempting BPM calculation
            quality_threshold: Minimum signal quality to attempt calculation
        """
        self.fps = fps
        self.window_seconds = window_seconds
        self.min_seconds = min_seconds
        self.quality_threshold = quality_threshold
        
        # Calculate buffer size
        self.buffer_size = int(fps * window_seconds)
        self.min_buffer_size = int(fps * min_seconds)
        
        # Rolling buffer for RGB values
        self._rgb_buffer = deque(maxlen=self.buffer_size)
        
        # Cache for last successful result
        self._last_result: Optional[HeartRateResult] = None
    
    def add_frame(self, face_result: FaceResult) -> HeartRateResult:
        """
        Process a frame and update heart rate estimation.
        
        Args:
            face_result: Face detection result containing forehead ROI
            
        Returns:
            HeartRateResult with current estimation
        """
        start_time = time.perf_counter()
        
        # If no face detected or no forehead ROI, don't add to buffer
        if not face_result.detected or face_result.forehead_roi is None:
            buffer_fill = len(self._rgb_buffer) / self.buffer_size
            result = HeartRateResult(
                bpm=None,
                confidence=0.0,
                signal_quality=0.0,
                buffer_fill=buffer_fill,
                inference_time_ms=0.0,
                raw_rgb_trace=np.array(self._rgb_buffer) if self._rgb_buffer else None
            )
            self._last_result = result
            return result
        
        # Extract mean RGB values from forehead ROI
        # cv2.mean returns (B, G, R, alpha), we need (R, G, B)
        bgr_mean = cv2.mean(face_result.forehead_roi)[:3]
        rgb_mean = (bgr_mean[2], bgr_mean[1], bgr_mean[0])  # Reorder to RGB
        
        # Add to rolling buffer
        self._rgb_buffer.append(rgb_mean)
        
        # Calculate buffer fill percentage
        buffer_fill = len(self._rgb_buffer) / self.buffer_size
        
        # If buffer not full enough, return partial result
        if len(self._rgb_buffer) < self.min_buffer_size:
            result = HeartRateResult(
                bpm=None,
                confidence=0.0,
                signal_quality=0.0,
                buffer_fill=buffer_fill,
                inference_time_ms=(time.perf_counter() - start_time) * 1000,
                raw_rgb_trace=np.array(self._rgb_buffer)
            )
            self._last_result = result
            return result
        
        # Convert buffer to numpy array for processing
        rgb_trace = np.array(self._rgb_buffer, dtype=np.float32)
        
        # Compute signal quality
        signal_quality = compute_signal_quality(rgb_trace)
        
        # If signal quality too low, return result with low confidence
        if signal_quality < self.quality_threshold:
            result = HeartRateResult(
                bpm=None,
                confidence=0.0,
                signal_quality=signal_quality,
                buffer_fill=buffer_fill,
                inference_time_ms=(time.perf_counter() - start_time) * 1000,
                raw_rgb_trace=rgb_trace
            )
            self._last_result = result
            return result
        
        # Apply signal processing pipeline
        try:
            normalized_rgb = normalize_rgb_trace(rgb_trace)
            pulse_signal = apply_pos_algorithm(normalized_rgb)
            filtered_signal = bandpass_filter(pulse_signal, self.fps)
            bpm, confidence = extract_bpm_from_fft(filtered_signal, self.fps)
        except Exception as e:
            logger.warning(f"Signal processing failed: {e}")
            result = HeartRateResult(
                bpm=None,
                confidence=0.0,
                signal_quality=signal_quality,
                buffer_fill=buffer_fill,
                inference_time_ms=(time.perf_counter() - start_time) * 1000,
                raw_rgb_trace=rgb_trace
            )
            self._last_result = result
            return result
        
        # Create result
        result = HeartRateResult(
            bpm=bpm,
            confidence=confidence,
            signal_quality=signal_quality,
            buffer_fill=buffer_fill,
            inference_time_ms=(time.perf_counter() - start_time) * 1000,
            raw_rgb_trace=rgb_trace
        )
        
        self._last_result = result
        return result
    
    def reset(self) -> None:
        """Clear the RGB buffer and last result"""
        self._rgb_buffer.clear()
        self._last_result = None
        logger.info("Heart rate detector reset")
    
    def get_debug_info(self) -> dict:
        """Get debug information for logging"""
        return {
            "buffer_length": len(self._rgb_buffer),
            "buffer_fill": len(self._rgb_buffer) / self.buffer_size if self.buffer_size > 0 else 0,
            "fps": self.fps,
            "window_seconds": self.window_seconds
        }
