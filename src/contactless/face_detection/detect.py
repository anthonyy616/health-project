"""
Face Detection Module using MediaPipe Face Landmarker (Tasks API)
Provides face detection with 478 landmarks and ROI extraction for downstream models.
Compatible with MediaPipe 0.10.x+ (Tasks API)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from pathlib import Path
import time
import urllib.request
import os


@dataclass
class FaceResult:
    """Container for face detection results"""
    detected: bool
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    landmarks: Optional[np.ndarray] = None  # 478 x 3 (x, y, z)
    face_roi: Optional[np.ndarray] = None  # 224x224 cropped face
    forehead_roi: Optional[np.ndarray] = None  # For rPPG
    left_eye_roi: Optional[np.ndarray] = None
    right_eye_roi: Optional[np.ndarray] = None
    confidence: float = 0.0
    detection_time_ms: float = 0.0


class FaceDetector:
    """
    MediaPipe-based face detector with landmark extraction.
    Uses the new Tasks API (MediaPipe 0.10.x+).
    
    Usage:
        detector = FaceDetector()
        result = detector.detect(frame)
        if result.detected:
            # Use result.bbox, result.landmarks, result.face_roi, etc.
    """
    
    # Model URL for download
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    MODEL_PATH = Path(__file__).parent / "face_landmarker.task"
    
    # Landmark indices for specific regions (MediaPipe Face Landmarker has 478 landmarks)
    # Forehead region landmarks
    FOREHEAD_LANDMARKS = [10, 67, 69, 104, 108, 151, 299, 337, 338, 297, 332, 284]
    
    # Left eye landmarks (for bounding box)
    LEFT_EYE_LANDMARKS = [33, 133, 160, 159, 158, 144, 145, 153, 154, 155, 157, 163]
    
    # Right eye landmarks
    RIGHT_EYE_LANDMARKS = [362, 263, 387, 386, 385, 373, 374, 380, 381, 382, 384, 390]
    
    # Face oval landmarks (for bounding box)
    FACE_OVAL_LANDMARKS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                          397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                          172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_faces: int = 1
    ):
        """
        Initialize the face detector.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
            max_faces: Maximum number of faces to detect
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_faces = max_faces
        
        # Download model if needed
        self._ensure_model()
        
        # Initialize MediaPipe Face Landmarker
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python import BaseOptions
        
        base_options = BaseOptions(model_asset_path=str(self.MODEL_PATH))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_detection_time = 0.0
    
    def _ensure_model(self):
        """Download model file if not present"""
        if not self.MODEL_PATH.exists():
            print(f"Downloading face landmarker model to {self.MODEL_PATH}...")
            self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print("Model downloaded successfully!")
    
    def detect(self, frame: np.ndarray) -> FaceResult:
        """
        Detect face and extract landmarks from frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            FaceResult with detection data
        """
        import mediapipe as mp
        
        start_time = time.perf_counter()
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Process frame
        result = self.landmarker.detect(mp_image)
        
        detection_time = (time.perf_counter() - start_time) * 1000
        
        if not result.face_landmarks:
            return FaceResult(detected=False, detection_time_ms=detection_time)
        
        # Get first face
        face_landmarks = result.face_landmarks[0]
        
        # Convert landmarks to numpy array
        landmarks = np.array([
            [lm.x * w, lm.y * h, lm.z * w]
            for lm in face_landmarks
        ])
        
        # Calculate bounding box from face oval
        bbox = self._get_face_bbox(landmarks, w, h)
        
        # Extract ROIs
        face_roi = self._extract_face_roi(frame, bbox)
        forehead_roi = self._extract_forehead_roi(frame, landmarks)
        left_eye_roi, right_eye_roi = self._extract_eye_rois(frame, landmarks)
        
        # Estimate confidence
        confidence = 0.95  # Tasks API doesn't provide per-landmark confidence
        
        return FaceResult(
            detected=True,
            bbox=bbox,
            landmarks=landmarks,
            face_roi=face_roi,
            forehead_roi=forehead_roi,
            left_eye_roi=left_eye_roi,
            right_eye_roi=right_eye_roi,
            confidence=confidence,
            detection_time_ms=detection_time
        )
    
    def _get_face_bbox(
        self, 
        landmarks: np.ndarray, 
        img_w: int, 
        img_h: int,
        padding: float = 0.1
    ) -> Tuple[int, int, int, int]:
        """Calculate face bounding box with padding"""
        # Use all landmarks for more reliable bbox
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Add padding
        w = x_max - x_min
        h = y_max - y_min
        x_min = max(0, int(x_min - w * padding))
        y_min = max(0, int(y_min - h * padding))
        x_max = min(img_w, int(x_max + w * padding))
        y_max = min(img_h, int(y_max + h * padding))
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def _extract_face_roi(
        self, 
        frame: np.ndarray, 
        bbox: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = (224, 224)
    ) -> Optional[np.ndarray]:
        """Extract and resize face region for ML models"""
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return None
        
        face_crop = frame[y:y+h, x:x+w]
        if face_crop.size == 0:
            return None
        
        return cv2.resize(face_crop, target_size)
    
    def _extract_forehead_roi(
        self, 
        frame: np.ndarray, 
        landmarks: np.ndarray,
        target_size: Tuple[int, int] = (64, 32)
    ) -> Optional[np.ndarray]:
        """Extract forehead region for rPPG analysis"""
        # Safely get forehead landmarks (handle if fewer landmarks available)
        valid_indices = [i for i in self.FOREHEAD_LANDMARKS if i < len(landmarks)]
        if len(valid_indices) < 3:
            return None
            
        forehead_points = landmarks[valid_indices, :2].astype(int)
        
        x_min, y_min = forehead_points.min(axis=0)
        x_max, y_max = forehead_points.max(axis=0)
        
        # Ensure valid bounds
        h, w = frame.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        if x_max <= x_min or y_max <= y_min:
            return None
        
        forehead_crop = frame[y_min:y_max, x_min:x_max]
        if forehead_crop.size == 0:
            return None
        
        return cv2.resize(forehead_crop, target_size)
    
    def _extract_eye_rois(
        self, 
        frame: np.ndarray, 
        landmarks: np.ndarray,
        target_size: Tuple[int, int] = (64, 32)
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract left and right eye regions for pupil detection"""
        h, w = frame.shape[:2]
        
        def extract_eye(eye_landmarks):
            valid_indices = [i for i in eye_landmarks if i < len(landmarks)]
            if len(valid_indices) < 3:
                return None
                
            points = landmarks[valid_indices, :2].astype(int)
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            
            # Add padding
            pad_x = int((x_max - x_min) * 0.3)
            pad_y = int((y_max - y_min) * 0.5)
            
            x_min = max(0, x_min - pad_x)
            y_min = max(0, y_min - pad_y)
            x_max = min(w, x_max + pad_x)
            y_max = min(h, y_max + pad_y)
            
            if x_max <= x_min or y_max <= y_min:
                return None
            
            eye_crop = frame[y_min:y_max, x_min:x_max]
            if eye_crop.size == 0:
                return None
            
            return cv2.resize(eye_crop, target_size)
        
        left_eye = extract_eye(self.LEFT_EYE_LANDMARKS)
        right_eye = extract_eye(self.RIGHT_EYE_LANDMARKS)
        
        return left_eye, right_eye
    
    def draw_landmarks(
        self, 
        frame: np.ndarray, 
        face_result: FaceResult,
        draw_mesh: bool = False,
        draw_contours: bool = True
    ) -> np.ndarray:
        """
        Draw face landmarks on frame.
        
        Args:
            frame: Image to draw on
            face_result: Detection result
            draw_mesh: If True, draw full mesh connections
            draw_contours: If True, draw face contours
            
        Returns:
            Frame with landmarks drawn
        """
        if not face_result.detected or face_result.landmarks is None:
            return frame
        
        output = frame.copy()
        
        if draw_contours:
            # Draw face oval contour
            valid_oval = [i for i in self.FACE_OVAL_LANDMARKS if i < len(face_result.landmarks)]
            for idx in valid_oval:
                x, y = int(face_result.landmarks[idx, 0]), int(face_result.landmarks[idx, 1])
                cv2.circle(output, (x, y), 1, (0, 255, 0), -1)
            
            # Draw eye contours
            valid_eyes = [i for i in self.LEFT_EYE_LANDMARKS + self.RIGHT_EYE_LANDMARKS 
                          if i < len(face_result.landmarks)]
            for idx in valid_eyes:
                x, y = int(face_result.landmarks[idx, 0]), int(face_result.landmarks[idx, 1])
                cv2.circle(output, (x, y), 1, (255, 0, 0), -1)
        
        if draw_mesh:
            # Draw all landmarks
            for i, (x, y, z) in enumerate(face_result.landmarks):
                cv2.circle(output, (int(x), int(y)), 1, (0, 200, 0), -1)
        
        return output
    
    def close(self):
        """Release resources"""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
