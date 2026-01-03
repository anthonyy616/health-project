"""
Age Estimator - Inference Wrapper
=================================
High-level interface for age estimation during video inference.
Wraps the trained LightAgeNet model for easy integration with face detection.

Usage:
    from src.contactless.age_estimation.estimator import AgeEstimator
    
    estimator = AgeEstimator()  # Uses default model path
    age, confidence = estimator.estimate(face_roi)  # numpy array
    
    # Or with FaceResult from face detector
    age, confidence = estimator.estimate_from_face_result(face_result)
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Optional, Union
from dataclasses import dataclass
import logging
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.age_detection.light_age_net import LightAgeNet, LightAgeNetV2
from models.age_detection.mobilenet_age import MobileNetV3Age

logger = logging.getLogger(__name__)


@dataclass
class AgeEstimationResult:
    """Result container for age estimation"""
    age: int
    confidence: float
    raw_prediction: float
    inference_time_ms: float


class AgeEstimator:
    """
    High-level age estimator for video/webcam integration.
    
    Handles:
    - Model loading
    - Image preprocessing
    - Inference with confidence estimation
    - Results postprocessing
    
    Usage:
        estimator = AgeEstimator(model_path="models/weights/age_detection/best_model.pt")
        result = estimator.estimate(face_image)
        print(f"Estimated age: {result.age} years (confidence: {result.confidence:.2f})")
    """
    
    # Default paths
    DEFAULT_MODEL_PATH = PROJECT_ROOT / "models/weights/age_detection/best_model.pt"
    
    # Preprocessing constants
    INPUT_SIZE = (224, 224)
    MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
    STD = [0.229, 0.224, 0.225]   # ImageNet std
    
    # Age constraints
    MIN_AGE = 0
    MAX_AGE = 100
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "mobilenet",  # Default to mobilenet for best accuracy
        use_normalization: bool = False,  # Set True for ImageNet normalization
        device: str = "cpu"
    ):
        """
        Initialize the age estimator.
        
        Args:
            model_path: Path to trained weights (.pt file)
            model_type: "light", "v2", or "mobilenet"
            use_normalization: Whether to apply ImageNet normalization
            device: Device for inference ("cpu" or "cuda")
        """
        self.device = torch.device(device)
        self.use_normalization = use_normalization
        self.model_type = model_type
        
        # Create model
        if model_type == "light":
            self.model = LightAgeNet()
        elif model_type == "v2":
            self.model = LightAgeNetV2()
        elif model_type == "mobilenet":
            self.model = MobileNetV3Age(pretrained=False)  # Don't need ImageNet weights for inference
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load weights if available
        if model_path:
            model_path = Path(model_path)
        else:
            model_path = self.DEFAULT_MODEL_PATH
        
        if model_path.exists():
            logger.info(f"Loading model weights from {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model_loaded = True
        else:
            logger.warning(f"Model weights not found at {model_path}. Using random weights.")
            self.model_loaded = False
        
        # Set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Age estimator initialized (model_loaded={self.model_loaded})")
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: BGR or RGB image (H, W, 3)
            
        Returns:
            Tensor of shape (1, 3, 224, 224)
        """
        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            # Assume BGR from OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, self.INPUT_SIZE)
        
        # Convert to tensor
        image = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization if enabled
        if self.use_normalization:
            image = (image - self.MEAN) / self.STD
        
        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))
        
        # Add batch dimension
        tensor = torch.from_numpy(image).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def estimate(self, face_image: np.ndarray) -> AgeEstimationResult:
        """
        Estimate age from face image.
        
        Args:
            face_image: Face ROI as numpy array (H, W, 3)
            
        Returns:
            AgeEstimationResult with age, confidence, and timing
        """
        import time
        
        start_time = time.perf_counter()
        
        # Preprocess
        tensor = self.preprocess(face_image)
        
        # Inference
        with torch.no_grad():
            raw_prediction = self.model(tensor).item()
        
        # Clamp to valid range
        clamped_age = max(self.MIN_AGE, min(self.MAX_AGE, raw_prediction))
        
        # Round to integer
        predicted_age = int(round(clamped_age))
        
        # Estimate confidence (higher if prediction is in typical range)
        confidence = self._estimate_confidence(raw_prediction)
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        return AgeEstimationResult(
            age=predicted_age,
            confidence=confidence,
            raw_prediction=raw_prediction,
            inference_time_ms=inference_time
        )
    
    def estimate_from_face_result(self, face_result) -> Optional[AgeEstimationResult]:
        """
        Estimate age from FaceResult object (from face detector).
        
        Args:
            face_result: FaceResult from FaceDetector
            
        Returns:
            AgeEstimationResult or None if no face ROI available
        """
        if not face_result.detected or face_result.face_roi is None:
            return None
        
        return self.estimate(face_result.face_roi)
    
    def _estimate_confidence(self, raw_age: float) -> float:
        """
        Estimate confidence based on prediction characteristics.
        
        Higher confidence for:
        - Ages in typical range (5-80)
        - Model loaded successfully
        - No extreme predictions
        """
        if not self.model_loaded:
            return 0.3  # Low confidence without trained model
        
        # Base confidence
        confidence = 0.85
        
        # Reduce if outside typical range
        if raw_age < 5 or raw_age > 80:
            confidence -= 0.15
        
        # Reduce if very extreme
        if raw_age < 0 or raw_age > 100:
            confidence -= 0.20
        
        return max(0.1, min(1.0, confidence))
    
    def __call__(self, face_image: np.ndarray) -> Tuple[int, float]:
        """
        Shorthand for estimate().
        
        Returns:
            Tuple of (predicted_age, confidence)
        """
        result = self.estimate(face_image)
        return result.age, result.confidence
    
    def is_loaded(self) -> bool:
        """Check if model weights were loaded successfully"""
        return self.model_loaded
