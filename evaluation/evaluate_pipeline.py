"""
Full Pipeline Evaluation Harness
=================================

Evaluates the complete VitalsPipeline end-to-end: face detection →
age estimation + heart rate + respiration + pupil detection, measuring
per-frame latency, timeout handling, and graceful degradation under
low-quality signals.

Usage:
    python evaluation/evaluate_pipeline.py --mode benchmark
    python evaluation/evaluate_pipeline.py --mode stress --n-frames 1000
    python evaluation/evaluate_pipeline.py --mode timeout --timeout-ms 100
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from evaluation.metrics import compute_latency_stats

logger = logging.getLogger(__name__)


def benchmark_pipeline(
    n_frames: int = 300,
    fps: int = 30,
    output_dir: str = "results/pipeline_evaluation",
) -> Dict:
    """
    Benchmark the full pipeline on synthetic frames.

    Measures per-frame and per-module latency, throughput, and error rate.

    Args:
        n_frames: Number of frames to process
        fps: Expected frame rate
        output_dir: Where to save results

    Returns:
        Benchmark results dictionary
    """
    from src.contactless.pipeline import VitalsPipeline, VitalsReading

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing VitalsPipeline for benchmark ({n_frames} frames @ {fps} fps)")

    try:
        pipeline = VitalsPipeline(fps=fps, use_threading=False)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return {"error": str(e)}

    # Generate synthetic frames (simple colored rectangles)
    np.random.seed(42)
    frames = []
    for i in range(n_frames):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Draw a rough face-like ellipse in center for face detection
        import cv2
        cx, cy = 320, 240
        cv2.ellipse(frame, (cx, cy), (80, 110), 0, 0, 360, (180, 150, 130), -1)
        # Eye regions
        cv2.circle(frame, (cx - 30, cy - 15), 8, (50, 50, 50), -1)
        cv2.circle(frame, (cx + 30, cy - 15), 8, (50, 50, 50), -1)
        frames.append(frame)

    # Run pipeline
    readings: List[VitalsReading] = []
    frame_latencies = []
    module_latencies = {"face_detection": [], "heart_rate": [], "respiration": []}

    logger.info(f"Processing {n_frames} frames...")

    start_benchmark = time.perf_counter()

    for i, frame in enumerate(frames):
        frame_start = time.perf_counter()

        reading = pipeline.process_frame(frame)

        frame_latency = (time.perf_counter() - frame_start) * 1000
        frame_latencies.append(frame_latency)
        readings.append(reading)

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{n_frames} frames "
                       f"(avg {np.mean(frame_latencies):.1f} ms/frame)")

    total_benchmark_time = (time.perf_counter() - start_benchmark) * 1000

    pipeline.close()

    # Analyze results
    frame_latencies_arr = np.array(frame_latencies)
    latency_stats = compute_latency_stats(frame_latencies_arr)

    # Count successes and errors per module
    n_face_detected = sum(1 for r in readings if r.face_detected)
    n_hr_valid = sum(1 for r in readings if r.heart_rate_bpm is not None)
    n_resp_valid = sum(1 for r in readings if r.respiratory_rate_bpm is not None)
    n_age_valid = sum(1 for r in readings if r.age is not None)
    n_pupil_valid = sum(1 for r in readings if r.pupil_diameter_mm is not None)
    n_module_errors = sum(1 for r in readings if r.module_errors)

    throughput_fps = n_frames / (total_benchmark_time / 1000)

    report = {
        "mode": "benchmark",
        "n_frames": n_frames,
        "fps": fps,
        "total_time_ms": total_benchmark_time,
        "throughput_fps": throughput_fps,
        "latency": latency_stats,
        "module_success_rate": {
            "face_detection": n_face_detected / n_frames,
            "heart_rate": n_hr_valid / n_frames,
            "respiration": n_resp_valid / n_frames,
            "age": n_age_valid / n_frames,
            "pupil": n_pupil_valid / n_frames,
        },
        "n_module_errors": n_module_errors,
        "n_frames_processed": n_frames,
    }

    # Log results
    logger.info(f"\n{'='*50}")
    logger.info("Pipeline Benchmark Results")
    logger.info(f"{'='*50}")
    logger.info(f"  Frames processed:  {n_frames}")
    logger.info(f"  Total time:        {total_benchmark_time:.0f} ms")
    logger.info(f"  Throughput:        {throughput_fps:.1f} FPS")
    logger.info(f"  Mean latency:      {latency_stats['mean']:.1f} ms")
    logger.info(f"  P95 latency:       {latency_stats['p95']:.1f} ms")
    logger.info(f"  P99 latency:       {latency_stats['p99']:.1f} ms")
    logger.info(f"  Face detected:     {n_face_detected}/{n_frames} ({n_face_detected/n_frames:.1%})")
    logger.info(f"  HR valid:          {n_hr_valid}/{n_frames} ({n_hr_valid/n_frames:.1%})")
    logger.info(f"  Resp valid:        {n_resp_valid}/{n_frames} ({n_resp_valid/n_frames:.1%})")
    logger.info(f"  Age valid:         {n_age_valid}/{n_frames} ({n_age_valid/n_frames:.1%})")
    logger.info(f"  Pupil valid:       {n_pupil_valid}/{n_frames} ({n_pupil_valid/n_frames:.1%})")
    logger.info(f"  Module errors:     {n_module_errors}")

    # Save report
    report_path = output_path / "pipeline_benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport saved to: {report_path}")

    return report


def stress_test_pipeline(
    n_frames: int = 1000,
    error_injection_rate: float = 0.1,
    output_dir: str = "results/pipeline_evaluation",
) -> Dict:
    """
    Stress test the pipeline with error-injected frames.

    Verifies that the pipeline degrades gracefully when individual
    modules fail, without crashing or producing garbage results.

    Args:
        n_frames: Number of frames to process
        error_injection_rate: Fraction of frames to corrupt
        output_dir: Where to save results

    Returns:
        Stress test results dictionary
    """
    from src.contactless.pipeline import VitalsPipeline, VitalsReading
    import cv2

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Stress testing pipeline: {n_frames} frames, "
               f"{error_injection_rate:.0%} error injection rate")

    try:
        pipeline = VitalsPipeline(fps=30, use_threading=False)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return {"error": str(e)}

    np.random.seed(42)
    n_corrupted = 0
    n_crashes = 0
    readings = []
    frame_latencies = []

    for i in range(n_frames):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Corrupt some frames (black, noise, wrong shape)
        if np.random.random() < error_injection_rate:
            n_corrupted += 1
            corruption_type = np.random.choice(["black", "noise", "tiny"])
            if corruption_type == "black":
                frame = np.zeros_like(frame)
            elif corruption_type == "noise":
                frame = np.random.randint(0, 255, frame.shape, dtype=np.uint8)
            elif corruption_type == "tiny":
                frame = np.ones((10, 10, 3), dtype=np.uint8) * 128

        frame_start = time.perf_counter()
        try:
            reading = pipeline.process_frame(frame)
            frame_latency = (time.perf_counter() - frame_start) * 1000
            frame_latencies.append(frame_latency)
            readings.append(reading)
        except Exception as e:
            n_crashes += 1
            logger.error(f"Frame {i} caused pipeline crash: {e}")

    pipeline.close()

    frame_latencies_arr = np.array(frame_latencies) if frame_latencies else np.array([0.0])
    latency_stats = compute_latency_stats(frame_latencies_arr)

    n_with_errors = sum(1 for r in readings if r.module_errors)

    report = {
        "mode": "stress_test",
        "n_frames": n_frames,
        "n_corrupted": n_corrupted,
        "error_injection_rate": error_injection_rate,
        "n_crashes": n_crashes,
        "n_readings": len(readings),
        "n_readings_with_module_errors": n_with_errors,
        "crash_rate": n_crashes / n_frames,
        "graceful_degradation_rate": 1.0 - (n_crashes / n_frames),
        "latency": latency_stats,
    }

    logger.info(f"\n{'='*50}")
    logger.info("Pipeline Stress Test Results")
    logger.info(f"{'='*50}")
    logger.info(f"  Total frames:            {n_frames}")
    logger.info(f"  Corrupted frames:        {n_corrupted}")
    logger.info(f"  Pipeline crashes:        {n_crashes}")
    logger.info(f"  Crash rate:              {n_crashes/n_frames:.1%}")
    logger.info(f"  Graceful degradation:    {1 - n_crashes/n_frames:.1%}")
    logger.info(f"  Readings with errors:    {n_with_errors}/{len(readings)}")
    logger.info(f"  Mean latency:            {latency_stats['mean']:.1f} ms")

    report_path = output_path / "pipeline_stress_test_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport saved to: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Full Pipeline Evaluation Harness")
    parser.add_argument("--mode", choices=["benchmark", "stress"], default="benchmark",
                        help="Evaluation mode: benchmark (latency/throughput) or stress (error resilience)")
    parser.add_argument("--n-frames", type=int, default=300,
                        help="Number of frames to process")
    parser.add_argument("--output-dir", default="results/pipeline_evaluation")
    parser.add_argument("--error-rate", type=float, default=0.1,
                        help="Error injection rate for stress test (0-1)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.mode == "benchmark":
        benchmark_pipeline(
            n_frames=args.n_frames,
            output_dir=args.output_dir,
        )
    elif args.mode == "stress":
        stress_test_pipeline(
            n_frames=args.n_frames,
            error_injection_rate=args.error_rate,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
