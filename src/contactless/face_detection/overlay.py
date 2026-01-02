"""
Video Overlay Module
Renders metrics, face bounding boxes, and status indicators on video frames.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class MetricsData:
    """Container for vital signs metrics to display"""
    age: Optional[int] = None
    age_confidence: float = 0.0
    heart_rate: Optional[float] = None
    hr_confidence: float = 0.0
    respiratory_rate: Optional[float] = None
    resp_confidence: float = 0.0
    pupil_diameter: Optional[float] = None
    pupil_confidence: float = 0.0
    temperature: Optional[float] = None
    temp_confidence: float = 0.0


class VitalsOverlay:
    """
    Renders vital signs metrics overlay on video frames.
    
    Features:
    - Face bounding box with confidence
    - Metrics panel (age, HR, respiration, pupil)
    - Confidence bars
    - FPS counter
    - Color-coded alerts
    
    Usage:
        overlay = VitalsOverlay()
        frame = overlay.draw(frame, face_result, metrics)
    """
    
    # Colors (BGR format)
    COLOR_GREEN = (0, 200, 0)
    COLOR_YELLOW = (0, 200, 200)
    COLOR_RED = (0, 0, 200)
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    COLOR_PANEL_BG = (40, 40, 40)
    COLOR_BOX = (0, 255, 0)
    
    # Alert thresholds
    HR_NORMAL = (50, 100)  # BPM
    RESP_NORMAL = (12, 20)  # BPM
    TEMP_NORMAL = (36.0, 37.5)  # Celsius
    
    def __init__(self, show_landmarks: bool = True, show_mesh: bool = False):
        """
        Initialize overlay renderer.
        
        Args:
            show_landmarks: Show facial landmark points
            show_mesh: Show full face mesh (more detailed)
        """
        self.show_landmarks = show_landmarks
        self.show_mesh = show_mesh
        
        self._fps_buffer = []
        self._fps_update_interval = 0.5  # seconds
        self._last_fps_update = time.time()
        self._current_fps = 0.0
    
    def draw(
        self, 
        frame: np.ndarray, 
        face_result=None,
        metrics: Optional[MetricsData] = None
    ) -> np.ndarray:
        """
        Draw all overlays on frame.
        
        Args:
            frame: BGR image from OpenCV
            face_result: FaceResult from FaceDetector
            metrics: MetricsData with vital signs (None for placeholder "--")
            
        Returns:
            Frame with overlays drawn
        """
        output = frame.copy()
        
        # Update FPS calculation
        self._update_fps()
        
        # Draw face bounding box
        if face_result and face_result.detected:
            output = self._draw_face_box(output, face_result)
            
            # Draw landmarks if enabled
            if self.show_landmarks and face_result.landmarks is not None:
                output = self._draw_landmarks(output, face_result.landmarks)
        
        # Draw metrics panel
        output = self._draw_metrics_panel(output, metrics, face_result)
        
        # Draw FPS and latency
        output = self._draw_performance_info(output, face_result)
        
        return output
    
    def _draw_face_box(self, frame: np.ndarray, face_result) -> np.ndarray:
        """Draw face bounding box with confidence"""
        if face_result.bbox is None:
            return frame
        
        x, y, w, h = face_result.bbox
        
        # Draw box
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.COLOR_BOX, 2)
        
        # Draw confidence badge
        conf_text = f"{face_result.confidence:.0%}"
        cv2.putText(
            frame, conf_text, (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_BOX, 1
        )
        
        return frame
    
    def _draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Draw facial landmarks"""
        for x, y, z in landmarks:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 200, 0), -1)
        return frame
    
    def _draw_metrics_panel(
        self, 
        frame: np.ndarray, 
        metrics: Optional[MetricsData],
        face_result=None
    ) -> np.ndarray:
        """Draw semi-transparent metrics panel in top-left"""
        # Panel dimensions
        panel_x, panel_y = 10, 10
        panel_w, panel_h = 200, 160
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(
            overlay, 
            (panel_x, panel_y), 
            (panel_x + panel_w, panel_y + panel_h),
            self.COLOR_PANEL_BG, 
            -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw panel border
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            self.COLOR_WHITE,
            1
        )
        
        # Title
        cv2.putText(
            frame, "VITAL SIGNS",
            (panel_x + 10, panel_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1
        )
        
        # Draw horizontal line under title
        cv2.line(
            frame,
            (panel_x + 5, panel_y + 30),
            (panel_x + panel_w - 5, panel_y + 30),
            self.COLOR_WHITE, 1
        )
        
        # Metrics
        y_offset = panel_y + 50
        line_height = 28
        
        # Age
        age_text = f"Age: {metrics.age} yrs" if metrics and metrics.age else "Age: --"
        age_color = self.COLOR_WHITE
        self._draw_metric_line(frame, panel_x + 10, y_offset, age_text, age_color,
                               metrics.age_confidence if metrics else 0)
        y_offset += line_height
        
        # Heart Rate
        hr_text = f"HR: {metrics.heart_rate:.0f} BPM" if metrics and metrics.heart_rate else "HR: -- BPM"
        hr_color = self._get_alert_color(metrics.heart_rate, self.HR_NORMAL) if metrics and metrics.heart_rate else self.COLOR_WHITE
        self._draw_metric_line(frame, panel_x + 10, y_offset, hr_text, hr_color,
                               metrics.hr_confidence if metrics else 0)
        y_offset += line_height
        
        # Respiratory Rate
        resp_text = f"Resp: {metrics.respiratory_rate:.0f} BPM" if metrics and metrics.respiratory_rate else "Resp: -- BPM"
        resp_color = self._get_alert_color(metrics.respiratory_rate, self.RESP_NORMAL) if metrics and metrics.respiratory_rate else self.COLOR_WHITE
        self._draw_metric_line(frame, panel_x + 10, y_offset, resp_text, resp_color,
                               metrics.resp_confidence if metrics else 0)
        y_offset += line_height
        
        # Pupil
        pupil_text = f"Pupil: {metrics.pupil_diameter:.1f}mm" if metrics and metrics.pupil_diameter else "Pupil: -- mm"
        self._draw_metric_line(frame, panel_x + 10, y_offset, pupil_text, self.COLOR_WHITE,
                               metrics.pupil_confidence if metrics else 0)
        
        return frame
    
    def _draw_metric_line(
        self, 
        frame: np.ndarray, 
        x: int, 
        y: int, 
        text: str, 
        color: Tuple[int, int, int],
        confidence: float = 0.0
    ):
        """Draw a single metric with confidence bar"""
        # Text
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        
        # Confidence bar (small, right side)
        bar_x = x + 140
        bar_y = y - 8
        bar_w = 40
        bar_h = 6
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        
        # Fill based on confidence
        if confidence > 0:
            fill_w = int(bar_w * confidence)
            fill_color = self.COLOR_GREEN if confidence > 0.7 else (self.COLOR_YELLOW if confidence > 0.4 else self.COLOR_RED)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_color, -1)
    
    def _get_alert_color(self, value: float, normal_range: Tuple[float, float]) -> Tuple[int, int, int]:
        """Get color based on whether value is in normal range"""
        if value is None:
            return self.COLOR_WHITE
        
        low, high = normal_range
        if low <= value <= high:
            return self.COLOR_GREEN
        elif value < low * 0.8 or value > high * 1.2:
            return self.COLOR_RED
        else:
            return self.COLOR_YELLOW
    
    def _draw_performance_info(self, frame: np.ndarray, face_result=None) -> np.ndarray:
        """Draw FPS and latency in bottom-right"""
        h, w = frame.shape[:2]
        
        # FPS
        fps_text = f"FPS: {self._current_fps:.0f}"
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(
            frame, fps_text,
            (w - text_size[0] - 10, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_GREEN, 1
        )
        
        # Detection latency
        if face_result and face_result.detected:
            lat_text = f"Det: {face_result.detection_time_ms:.0f}ms"
            text_size = cv2.getTextSize(lat_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(
                frame, lat_text,
                (w - text_size[0] - 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1
            )
        
        # Face status
        status = "Face: OK" if (face_result and face_result.detected) else "Face: Not Found"
        status_color = self.COLOR_GREEN if (face_result and face_result.detected) else self.COLOR_RED
        cv2.putText(
            frame, status,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1
        )
        
        return frame
    
    def _update_fps(self):
        """Update FPS calculation using moving average"""
        current_time = time.time()
        self._fps_buffer.append(current_time)
        
        # Keep only last 30 timestamps
        if len(self._fps_buffer) > 30:
            self._fps_buffer.pop(0)
        
        # Update FPS display periodically
        if current_time - self._last_fps_update >= self._fps_update_interval:
            if len(self._fps_buffer) >= 2:
                time_span = self._fps_buffer[-1] - self._fps_buffer[0]
                if time_span > 0:
                    self._current_fps = len(self._fps_buffer) / time_span
            self._last_fps_update = current_time
    
    @property
    def fps(self) -> float:
        """Get current FPS"""
        return self._current_fps
