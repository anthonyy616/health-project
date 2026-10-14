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

    detector = FaceDetector()
    overlay = VitalsOverlay(show_landmarks=True)

    try:
        with Camera(device_id=camera_id) as cam:
            for frame_data in cam.stream():
                frame = frame_data.frame

                result = detector.detect(frame)
                output = overlay.draw(frame, result)

                if overlay.show_landmarks and result.detected:
                    output = detector.draw_landmarks(output, result)

                cv2.imshow("Face Detection - Press 'q' to quit", output)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('l'):
                    overlay.show_landmarks = not overlay.show_landmarks
                    print(f"Landmarks: {'ON' if overlay.show_landmarks else 'OFF'}")
                elif key == ord('m'):
                    overlay.show_mesh = not overlay.show_mesh
                    print(f"Mesh: {'ON' if overlay.show_mesh else 'OFF'}")
    finally:
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

    detector = FaceDetector()
    overlay = VitalsOverlay(show_landmarks=False)

    print("\nLoading age estimation model...")
    estimator = AgeEstimator(
        model_path="models/weights/age_detection/best_model.pt",
        model_type="mobilenet"
    )
    if estimator.is_loaded():
        print("✓ Age model loaded successfully!")
    else:
        print("⚠ Warning: Using untrained model (random weights)")

    print("\nStarting webcam... Look at the camera!")
    print("-" * 50)

    try:
        with Camera(device_id=camera_id) as cam:
            for frame_data in cam.stream():
                frame = frame_data.frame

                face_result = detector.detect(frame)
                metrics = MetricsData()

                if face_result.detected and face_result.face_roi is not None:
                    age_result = estimator.estimate(face_result.face_roi)
                    metrics.age = age_result.age
                    metrics.age_confidence = age_result.confidence

                output = overlay.draw(frame, face_result, metrics)

                if overlay.show_landmarks and face_result.detected:
                    output = detector.draw_landmarks(output, face_result)

                cv2.imshow("Age Estimation - Press 'q' to quit", output)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('l'):
                    overlay.show_landmarks = not overlay.show_landmarks
                    print(f"Landmarks: {'ON' if overlay.show_landmarks else 'OFF'}")
    finally:
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
        run_heart_rate(camera_id=args.camera)

    elif args.mode == "dashboard":
        logger.info("Starting FastAPI dashboard...")
        logger.warning("Dashboard not yet implemented")

    elif args.mode == "all":
        logger.info("Running full vital signs monitoring...")
        run_all_modules(camera_id=args.camera)

    logger.info("Shutting down...")



def run_heart_rate(camera_id: int = None):
    """
    Run rPPG heart rate detection with video overlay.

    Uses the existing HeartRateDetector from src/contactless.heart_rate.
    Press 'q' to quit.
    """
    from src.contactless.face_detection import FaceDetector, VitalsOverlay
    from src.contactless.face_detection.overlay import MetricsData
    from src.contactless.heart_rate import HeartRateDetector

    print("=" * 50)
    print("HEART RATE (rPPG) MODE")
    print("=" * 50)
    print("Controls:")
    print("  q - Quit")
    print("=" * 50)
    print("\nLook at the camera and stay still for best results.")
    print("Heart rate needs a few seconds of buffered frames before it can report.\n")

    detector = FaceDetector()
    overlay = VitalsOverlay(show_landmarks=False)
    hr_detector = HeartRateDetector(fps=30)

    try:
        with Camera(device_id=camera_id) as cam:
            for frame_data in cam.stream():
                frame = frame_data.frame

                face_result = detector.detect(frame)
                metrics = MetricsData()

                if face_result.detected:
                    hr_result = hr_detector.add_frame(face_result)
                    if hr_result.bpm is not None:
                        metrics.heart_rate = hr_result.bpm
                        metrics.hr_confidence = hr_result.confidence

                output = overlay.draw(frame, face_result, metrics)

                if overlay.show_landmarks and face_result.detected:
                    output = detector.draw_landmarks(output, face_result)

                cv2.imshow("Heart Rate - Press 'q' to quit", output)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
    finally:
        detector.close()
        cv2.destroyAllWindows()
        print("Heart rate detection stopped.")


def run_all_modules(camera_id: int = None):
    """
    Run the full contactless vitals pipeline with overlay.

    Uses the existing VitalsPipeline from src/contactless.pipeline, wired to
    camera acquisition, overlay drawing, and console logging.

    Press 'q' to quit.
    """
    from src.contactless.pipeline import VitalsPipeline
    from src.contactless.face_detection import FaceDetector, VitalsOverlay
    from src.contactless.face_detection.overlay import MetricsData
    from src.utils import setup_logger

    logger = setup_logger("main.all")

    print("=" * 50)
    print("FULL VITALS MONITORING MODE")
    print("=" * 50)
    print("Controls:")
    print("  q - Quit")
    print("  r - Reset pipeline buffers")
    print("=" * 50)
    print("\nThis mode runs face detection, heart rate, respiration, pupil, and")
    print("age estimation together. Heart rate/respiration update every frame;")
    print("age and pupil run on their own skip cadence.\n")

    pipeline = VitalsPipeline(fps=30)
    overlay = VitalsOverlay(show_landmarks=False)

    start_time = __import__("time").monotonic()
    frame_count = 0

    try:
        with Camera(device_id=camera_id) as cam:
            for frame_data in cam.stream():
                frame = frame_data.frame
                frame_count += 1

                reading = pipeline.process_frame(frame)

                metrics = MetricsData()
                if reading.age is not None:
                    metrics.age = reading.age
                    metrics.age_confidence = reading.age_confidence
                if reading.heart_rate_bpm is not None:
                    metrics.heart_rate = reading.heart_rate_bpm
                    metrics.hr_confidence = reading.hr_confidence
                if reading.respiratory_rate_bpm is not None:
                    metrics.respiratory_rate = reading.respiratory_rate_bpm
                    metrics.resp_confidence = reading.resp_confidence
                if reading.pupil_diameter_mm is not None:
                    metrics.pupil_diameter = reading.pupil_diameter_mm
                    metrics.pupil_confidence = reading.pupil_confidence

                output = overlay.draw(frame, pipeline.last_face_result, metrics)

                if (frame_count % 30) == 0:
                    logger.info(
                        "frame=%d age=%s hr=%s resp=%s pupil=%s latency=%.1fms errors=%s",
                        reading.frame_number,
                        reading.age,
                        reading.heart_rate_bpm,
                        reading.respiratory_rate_bpm,
                        reading.pupil_diameter_mm,
                        reading.total_latency_ms,
                        reading.module_errors,
                    )

                cv2.imshow("Vitals - Press 'q' to quit, 'r' to reset", output)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    pipeline.reset()
                    logger.info("Pipeline reset by user")
    finally:
        pipeline.close()
        cv2.destroyAllWindows()
        elapsed = __import__("time").monotonic() - start_time
        logger.info(
            "Full monitoring stopped. frames=%d elapsed=%.1fs", frame_count, elapsed
        )


if __name__ == "__main__":
    main()
