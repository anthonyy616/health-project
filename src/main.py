"""
Main Entry Point for Vital Signs Monitoring System

Run with: python -m src.main
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils import Config, setup_logger
from src.utils.camera import Camera, test_camera

import cv2


def run_face_detection(camera_id: int = None):
    """
    Run face detection with video overlay.
    Press 'q' to quit, 'l' to toggle landmarks, 'm' to toggle mesh.
    """
    from src.contactless.face_detection import FaceDetector, VitalsOverlay
    
    print("=" * 50)
    print("FACE DETECTION MODE")
    print("=" * 50)
    print("Controls:")
    print("  q - Quit")
    print("  l - Toggle landmarks")
    print("  m - Toggle full mesh")
    print("=" * 50)
    
    # Initialize components
    detector = FaceDetector()
    overlay = VitalsOverlay(show_landmarks=True)
    
    with Camera(device_id=camera_id) as cam:
        for frame_data in cam.stream():
            frame = frame_data.frame
            
            # Detect face
            result = detector.detect(frame)
            
            # Draw overlay
            output = overlay.draw(frame, result)
            
            # Show landmarks if enabled
            if overlay.show_landmarks and result.detected:
                output = detector.draw_landmarks(output, result)
            
            # Display
            cv2.imshow("Face Detection - Press 'q' to quit", output)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                overlay.show_landmarks = not overlay.show_landmarks
                print(f"Landmarks: {'ON' if overlay.show_landmarks else 'OFF'}")
            elif key == ord('m'):
                overlay.show_mesh = not overlay.show_mesh
                print(f"Mesh: {'ON' if overlay.show_mesh else 'OFF'}")
    
    detector.close()
    cv2.destroyAllWindows()
    print("Face detection stopped.")


def run_age_estimation(camera_id: int = None):
    """
    Run face detection with age estimation.
    Shows real-time age predictions on detected faces.
    Press 'q' to quit, 'l' to toggle landmarks.
    """
    from src.contactless.face_detection import FaceDetector, VitalsOverlay
    from src.contactless.face_detection.overlay import MetricsData
    from src.contactless.age_estimation.estimator import AgeEstimator
    
    print("=" * 50)
    print("AGE ESTIMATION MODE")
    print("=" * 50)
    print("Controls:")
    print("  q - Quit")
    print("  l - Toggle landmarks")
    print("=" * 50)
    
    # Initialize components
    detector = FaceDetector()
    overlay = VitalsOverlay(show_landmarks=False)  # Less cluttered for age demo
    
    # Load trained age estimator
    print("\nLoading age estimation model...")
    estimator = AgeEstimator(
        model_path="models/weights/age_detection/best_model.pt"
    )
    if estimator.is_loaded():
        print("✓ Age model loaded successfully!")
    else:
        print("⚠ Warning: Using untrained model (random weights)")
    
    print("\nStarting webcam... Look at the camera!")
    print("-" * 50)
    
    with Camera(device_id=camera_id) as cam:
        for frame_data in cam.stream():
            frame = frame_data.frame
            
            # Detect face
            face_result = detector.detect(frame)
            
            # Create metrics data
            metrics = MetricsData()
            
            # Estimate age if face detected
            if face_result.detected and face_result.face_roi is not None:
                age_result = estimator.estimate(face_result.face_roi)
                metrics.age = age_result.age
                metrics.age_confidence = age_result.confidence
            
            # Draw overlay with metrics
            output = overlay.draw(frame, face_result, metrics)
            
            # Show landmarks if enabled
            if overlay.show_landmarks and face_result.detected:
                output = detector.draw_landmarks(output, face_result)
            
            # Display
            cv2.imshow("Age Estimation - Press 'q' to quit", output)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                overlay.show_landmarks = not overlay.show_landmarks
                print(f"Landmarks: {'ON' if overlay.show_landmarks else 'OFF'}")
    
    detector.close()
    cv2.destroyAllWindows()
    print("Age estimation stopped.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Vital Signs Monitoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        choices=["camera-test", "face-detect", "age", "heart-rate", "dashboard", "all"],
        default="camera-test",
        help="Operation mode (default: camera-test)"
    )
    
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera device ID (overrides config)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logger("main")
    logger.info("=" * 50)
    logger.info("Vital Signs Monitoring System")
    logger.info("=" * 50)
    
    # Load configuration
    config = Config()
    logger.info(f"Project: {config.get('project.name')}")
    logger.info(f"Version: {config.get('project.version')}")
    
    # Run selected mode
    if args.mode == "camera-test":
        logger.info("Running camera test...")
        test_camera()
        
    elif args.mode == "face-detect":
        logger.info("Running face detection...")
        run_face_detection(camera_id=args.camera)
        
    elif args.mode == "age":
        logger.info("Running age estimation...")
        run_age_estimation(camera_id=args.camera)
        
    elif args.mode == "heart-rate":
        logger.info("Running heart rate detection...")
        # TODO: Import and run rPPG heart rate module
        logger.warning("Heart rate module not yet implemented")
        
    elif args.mode == "dashboard":
        logger.info("Starting FastAPI dashboard...")
        # TODO: Import and run dashboard
        logger.warning("Dashboard not yet implemented")
        
    elif args.mode == "all":
        logger.info("Running full vital signs monitoring...")
        # TODO: Run all modules together
        logger.warning("Full system not yet implemented")
    
    logger.info("Shutting down...")


if __name__ == "__main__":
    main()
