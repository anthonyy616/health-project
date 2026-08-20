"""
Webcam Vitals Test
==================
Run heart rate, respiration, pupil detection, and age estimation
against live webcam feed. Displays real-time overlay with results.

Usage:
    python scripts/run_webcam_vitals.py
    python scripts/run_webcam_vitals.py --camera 1
    python scripts/run_webcam_vitals.py --width 640 --height 480
    python scripts/run_webcam_vitals.py --no-age          # skip age for speed
    python scripts/run_webcam_vitals.py --threading        # use thread pool
"""

import sys
import time
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.contactless.pipeline import VitalsPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run contactless vitals detection via webcam")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Capture height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Expected camera FPS (default: 30)")
    parser.add_argument("--no-age", action="store_true", help="Disable age estimation for speed")
    parser.add_argument("--threading", action="store_true", help="Use thread pool for parallel module execution")
    parser.add_argument("--save-log", type=str, default=None, help="Path to save JSON log of readings")
    return parser.parse_args()


def draw_overlay(frame, reading, fps_actual):
    """Draw vitals overlay on the frame."""
    h, w = frame.shape[:2]

    # Background panel
    panel_w = 320
    panel_h = 220
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 35
    line_h = 30

    # Title
    cv2.putText(frame, "CONTACTLESS VITALS", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y += line_h

    # Face status
    if reading.face_detected:
        cv2.putText(frame, "Face: DETECTED", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(frame, "Face: NOT FOUND", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    y += line_h

    # Heart rate
    if reading.heart_rate_bpm is not None:
        hr_text = f"HR: {reading.heart_rate_bpm:.1f} bpm"
        color = (0, 255, 0) if reading.hr_confidence > 0.5 else (0, 165, 255)
        cv2.putText(frame, hr_text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        cv2.putText(frame, f"conf: {reading.hr_confidence:.2f}", (220, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    else:
        cv2.putText(frame, "HR: --", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (128, 128, 128), 1)
    y += line_h

    # Respiratory rate
    if reading.respiratory_rate_bpm is not None:
        rr_text = f"RR: {reading.respiratory_rate_bpm:.1f} bpm"
        color = (0, 255, 0) if reading.resp_confidence > 0.5 else (0, 165, 255)
        cv2.putText(frame, rr_text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        cv2.putText(frame, f"conf: {reading.resp_confidence:.2f}", (220, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    else:
        cv2.putText(frame, "RR: --", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (128, 128, 128), 1)
    y += line_h

    # Pupil
    if reading.pupil_diameter_mm is not None:
        pupil_text = f"Pupil: {reading.pupil_diameter_mm:.2f} mm"
        cv2.putText(frame, pupil_text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)
    else:
        cv2.putText(frame, "Pupil: --", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (128, 128, 128), 1)
    y += line_h

    # Age
    if reading.age is not None:
        age_text = f"Age: {reading.age} (conf: {reading.age_confidence:.2f})"
        cv2.putText(frame, age_text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 200), 1)
    else:
        cv2.putText(frame, "Age: --", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (128, 128, 128), 1)
    y += line_h

    # FPS + latency
    cv2.putText(frame, f"FPS: {fps_actual:.0f}  latency: {reading.total_latency_ms:.0f}ms",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Module errors
    if reading.module_errors:
        err_text = "Errors: " + ", ".join(reading.module_errors.keys())
        cv2.putText(frame, err_text, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    return frame


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    print(f"Opening camera {args.camera} at {args.width}x{args.height}...")
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera {args.camera}")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {actual_w}x{actual_h}")

    # Build pipeline
    pipeline = VitalsPipeline(
        fps=args.fps,
        use_threading=args.threading,
        age_skip_frames=5 if not args.no_age else 9999,
    )

    if args.no_age:
        pipeline.age_estimator = None
        print("Age estimation DISABLED (--no-age)")

    print("Pipeline ready. Press 'q' to quit, 'r' to reset all detectors.")
    print("-" * 50)

    frame_count = 0
    fps_timer = time.perf_counter()
    fps_display = 0.0
    readings = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Failed to read frame")
                continue

            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)

            # Process through pipeline
            reading = pipeline.process_frame(frame)
            readings.append(reading.to_dict())

            # FPS calculation
            frame_count += 1
            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 1.0:
                fps_display = frame_count / elapsed
                frame_count = 0
                fps_timer = time.perf_counter()

            # Draw overlay
            frame = pipeline.face_detector.draw_landmarks(
                frame, pipeline.last_face_result, draw_contours=True
            ) if pipeline.last_face_result else frame

            output = draw_overlay(frame, reading, fps_display)

            cv2.imshow("Contactless Vitals - Press 'q' to quit", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                pipeline.reset()
                print("Pipeline RESET - all detectors cleared")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.close()

        # Save log if requested
        if args.save_log and readings:
            import json
            log_path = Path(args.save_log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'w') as f:
                json.dump(readings, f, indent=2)
            print(f"Saved {len(readings)} readings to {log_path}")

        print(f"Done. Processed {len(readings)} frames total.")


if __name__ == "__main__":
    main()
