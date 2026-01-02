"""
Camera Utility for Vital Signs Monitoring System
Provides a unified interface for webcam access that works with both
laptop webcam (development) and USB cameras (production/Logitech C920)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Generator
from dataclasses import dataclass
from pathlib import Path
import time

from .config import Config


@dataclass
class FrameData:
    """Container for a captured frame with metadata"""
    frame: np.ndarray
    timestamp: float
    frame_number: int
    width: int
    height: int


class Camera:
    """
    Camera abstraction for webcam access.
    
    Supports:
    - Laptop built-in webcam (development)
    - USB webcams like Logitech C920 (production)
    - Frame rate control
    - Resolution configuration
    
    Usage:
        # Context manager (recommended)
        with Camera() as cam:
            for frame_data in cam.stream():
                process(frame_data.frame)
                
        # Manual control
        cam = Camera(device_id=0)
        cam.start()
        frame = cam.read()
        cam.stop()
    """
    
    def __init__(
        self,
        device_id: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None
    ):
        """
        Initialize camera with optional overrides.
        
        Args:
            device_id: Camera device index (0=default, 1=USB cam)
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Target frames per second
        """
        config = Config()
        cam_config = config.camera
        
        self.device_id = device_id if device_id is not None else cam_config.device_id
        self.width = width if width is not None else cam_config.width
        self.height = height if height is not None else cam_config.height
        self.fps = fps if fps is not None else cam_config.fps
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._start_time = 0.0
        self._is_running = False
    
    def start(self) -> bool:
        """
        Start the camera capture.
        
        Returns:
            True if camera started successfully, False otherwise
        """
        if self._is_running:
            return True
            
        self._cap = cv2.VideoCapture(self.device_id)
        
        if not self._cap.isOpened():
            print(f"Error: Could not open camera {self.device_id}")
            return False
        
        # Set camera properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Verify actual settings (camera may not support requested values)
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self._cap.get(cv2.CAP_PROP_FPS))
        
        if actual_width != self.width or actual_height != self.height:
            print(f"Note: Camera resolution set to {actual_width}x{actual_height} "
                  f"(requested {self.width}x{self.height})")
            self.width = actual_width
            self.height = actual_height
        
        if actual_fps != self.fps:
            print(f"Note: Camera FPS set to {actual_fps} (requested {self.fps})")
            self.fps = actual_fps
        
        self._frame_count = 0
        self._start_time = time.time()
        self._is_running = True
        
        print(f"Camera started: {self.width}x{self.height} @ {self.fps}fps")
        return True
    
    def stop(self) -> None:
        """Stop the camera capture and release resources"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_running = False
        print("Camera stopped")
    
    def read(self) -> Optional[FrameData]:
        """
        Read a single frame from the camera.
        
        Returns:
            FrameData object or None if read failed
        """
        if not self._is_running or self._cap is None:
            return None
            
        ret, frame = self._cap.read()
        
        if not ret:
            return None
        
        self._frame_count += 1
        
        return FrameData(
            frame=frame,
            timestamp=time.time() - self._start_time,
            frame_number=self._frame_count,
            width=frame.shape[1],
            height=frame.shape[0]
        )
    
    def stream(self, max_frames: Optional[int] = None) -> Generator[FrameData, None, None]:
        """
        Generator that yields frames continuously.
        
        Args:
            max_frames: Optional limit on number of frames to yield
            
        Yields:
            FrameData objects
        """
        frames_yielded = 0
        
        while self._is_running:
            frame_data = self.read()
            
            if frame_data is None:
                break
                
            yield frame_data
            frames_yielded += 1
            
            if max_frames and frames_yielded >= max_frames:
                break
    
    def get_actual_fps(self) -> float:
        """Calculate the actual achieved FPS"""
        if self._frame_count == 0 or not self._is_running:
            return 0.0
        elapsed = time.time() - self._start_time
        return self._frame_count / elapsed if elapsed > 0 else 0.0
    
    @property
    def is_running(self) -> bool:
        """Check if camera is currently running"""
        return self._is_running
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Get current resolution as (width, height)"""
        return (self.width, self.height)
    
    def __enter__(self) -> 'Camera':
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit"""
        self.stop()
    
    @staticmethod
    def list_available_cameras(max_check: int = 5) -> list:
        """
        List available camera devices.
        
        Args:
            max_check: Maximum number of device indices to check
            
        Returns:
            List of available device indices
        """
        available = []
        for i in range(max_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available


def test_camera():
    """Test function to verify camera works"""
    print("Testing camera...")
    print(f"Available cameras: {Camera.list_available_cameras()}")
    
    with Camera() as cam:
        print("Press 'q' to quit")
        
        for frame_data in cam.stream(max_frames=300):  # 10 seconds at 30fps
            # Display the frame
            cv2.imshow("Camera Test", frame_data.frame)
            
            # Show FPS every 30 frames
            if frame_data.frame_number % 30 == 0:
                fps = cam.get_actual_fps()
                print(f"Frame {frame_data.frame_number}, FPS: {fps:.1f}")
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()
    print("Camera test complete")


if __name__ == "__main__":
    test_camera()
