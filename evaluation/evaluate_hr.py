"""
Heart Rate Evaluation Harness
==============================

Runs rPPG methods (POS, CHROM, GREEN) against ground-truth data and produces
comprehensive evaluation reports with metrics and plots.

Usage:
    python evaluation/evaluate_hr.py --dataset ubfc-rppg --methods pos,chrom,green
    python evaluation/evaluate_hr.py --dataset custom --data-dir data/custom/ --manifest data/custom/manifest.json
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from evaluation.metrics import (
    compute_mae,
    compute_rmse,
    compute_pearson_r,
    compute_r2,
    compute_bland_altman,
    compute_percentage_within_threshold,
    compute_failure_rate_simple,
    compute_latency_stats,
    evaluate_hr_method,
    HREvaluationResult,
)
from evaluation.plots import (
    plot_predicted_vs_ground_truth,
    plot_bland_altman,
    plot_error_histogram,
    plot_confidence_vs_error,
    plot_per_subject_mae,
    plot_latency_distribution,
    plot_method_comparison,
)
from evaluation.datasets import UBFCRPPGLoader, CustomRecordingLoader

logger = logging.getLogger(__name__)


class PosExtractor:
    """POS rPPG method — wraps existing signal_processing module."""

    def __init__(self):
        from src.contactless.heart_rate.signal_processing import (
            normalize_rgb_trace,
            apply_pos_algorithm,
            bandpass_filter,
            extract_bpm_from_fft,
        )
        self.normalize = normalize_rgb_trace
        self.pos = apply_pos_algorithm
        self.bandpass = bandpass_filter
        self.fft_bpm = extract_bpm_from_fft

    def extract_bpm_series(
        self, rgb_trace: np.ndarray, fps: int, window_sec: int = 10, step_sec: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract BPM values from an RGB trace using a sliding window.

        Args:
            rgb_trace: (N, 3) RGB values per frame
            fps: frames per second
            window_sec: analysis window in seconds
            step_sec: step size in seconds

        Returns:
            (bpm_values, confidences) arrays
        """
        window_size = fps * window_sec
        step_size = fps * step_sec

        if len(rgb_trace) < window_size:
            return np.array([]), np.array([])

        bpm_values = []
        confidences = []

        for start in range(0, len(rgb_trace) - window_size + 1, step_size):
            segment = rgb_trace[start : start + window_size]
            try:
                normalized = self.normalize(segment)
                pulse = self.pos(normalized)
                filtered = self.bandpass(pulse, float(fps))
                bpm, conf = self.fft_bpm(filtered, float(fps))
                bpm_values.append(bpm)
                confidences.append(conf)
            except Exception as e:
                logger.debug(f"POS extraction failed at frame {start}: {e}")
                bpm_values.append(np.nan)
                confidences.append(0.0)

        return np.array(bpm_values), np.array(confidences)


class GreenExtractor:
    """GREEN baseline — normalized green channel."""

    def extract_bpm_series(
        self, rgb_trace: np.ndarray, fps: int, window_sec: int = 10, step_sec: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        import scipy.signal

        window_size = fps * window_sec
        step_size = fps * step_sec

        if len(rgb_trace) < window_size:
            return np.array([]), np.array([])

        bpm_values = []
        confidences = []

        for start in range(0, len(rgb_trace) - window_size + 1, step_size):
            segment = rgb_trace[start : start + window_size]
            try:
                green = segment[:, 1].astype(np.float64)
                # DC normalization
                mean_val = np.mean(green)
                if abs(mean_val) < 1e-6:
                    bpm_values.append(np.nan)
                    confidences.append(0.0)
                    continue
                green_norm = (green - mean_val) / mean_val

                # Bandpass
                nyq = fps / 2.0
                b, a = scipy.signal.butter(4, [0.7 / nyq, 4.0 / nyq], btype="band")
                filtered = scipy.signal.filtfilt(b, a, green_norm)

                # FFT
                fft_mag = np.abs(np.fft.rfft(filtered))
                freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fps)
                mask = (freqs >= 0.7) & (freqs <= 4.0)
                if not np.any(mask):
                    bpm_values.append(np.nan)
                    confidences.append(0.0)
                    continue

                valid_power = fft_mag[mask]
                valid_freqs = freqs[mask]
                peak_idx = np.argmax(valid_power)
                peak_freq = valid_freqs[peak_idx]
                peak_power = valid_power[peak_idx]
                total_power = np.sum(valid_power)

                bpm = peak_freq * 60.0
                conf = peak_power / total_power if total_power > 0 else 0.0

                bpm_values.append(bpm)
                confidences.append(conf)
            except Exception as e:
                logger.debug(f"GREEN extraction failed at frame {start}: {e}")
                bpm_values.append(np.nan)
                confidences.append(0.0)

        return np.array(bpm_values), np.array(confidences)


class ChromExtractor:
    """CHROM (Chrominance) rPPG method — De Haan & Jeanne 2013."""

    def extract_bpm_series(
        self, rgb_trace: np.ndarray, fps: int, window_sec: int = 10, step_sec: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        import scipy.signal

        window_size = fps * window_sec
        step_size = fps * step_sec

        if len(rgb_trace) < window_size:
            return np.array([]), np.array([])

        # CHROM projection matrices
        Xs = np.array([[1.0, -1.0, 0.0]])   # (1, 3)
        Xd = np.array([[0.5, 0.5, -1.0]])    # (1, 3)

        bpm_values = []
        confidences = []

        for start in range(0, len(rgb_trace) - window_size + 1, step_size):
            segment = rgb_trace[start : start + window_size]
            try:
                # DC normalization
                means = np.mean(segment, axis=0)
                norm = np.zeros_like(segment, dtype=np.float64)
                for c in range(3):
                    if abs(means[c]) < 1e-6:
                        norm[:, c] = segment[:, c]
                    else:
                        norm[:, c] = (segment[:, c] - means[c]) / means[c]

                # Chrominance signals
                s = norm @ Xs.T  # (N, 1)
                d = norm @ Xd.T  # (N, 1)

                # Alpha: std ratio
                std_s = np.std(s)
                std_d = np.std(d)
                alpha = std_s / std_d if std_d > 1e-6 else 0.0

                # Pulse signal
                pulse = s[:, 0] - alpha * d[:, 0]

                # Normalize
                p_std = np.std(pulse)
                if p_std > 1e-6:
                    pulse = (pulse - np.mean(pulse)) / p_std

                # Bandpass
                nyq = fps / 2.0
                b, a = scipy.signal.butter(4, [0.7 / nyq, 4.0 / nyq], btype="band")
                filtered = scipy.signal.filtfilt(b, a, pulse)

                # FFT
                fft_mag = np.abs(np.fft.rfft(filtered))
                freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fps)
                mask = (freqs >= 0.7) & (freqs <= 4.0)
                if not np.any(mask):
                    bpm_values.append(np.nan)
                    confidences.append(0.0)
                    continue

                valid_power = fft_mag[mask]
                valid_freqs = freqs[mask]
                peak_idx = np.argmax(valid_power)
                peak_freq = valid_freqs[peak_idx]
                peak_power = valid_power[peak_idx]
                total_power = np.sum(valid_power)

                bpm = peak_freq * 60.0
                conf = peak_power / total_power if total_power > 0 else 0.0

                bpm_values.append(bpm)
                confidences.append(conf)
            except Exception as e:
                logger.debug(f"CHROM extraction failed at frame {start}: {e}")
                bpm_values.append(np.nan)
                confidences.append(0.0)

        return np.array(bpm_values), np.array(confidences)


METHOD_EXTRACTORS = {
    "pos": PosExtractor,
    "green": GreenExtractor,
    "chrom": ChromExtractor,
}


def _extract_rgb_trace(video_frames: np.ndarray, face_roi_func=None) -> np.ndarray:
    """
    Extract mean RGB from each frame.

    If face_roi_func is provided, it's called with each frame to get a crop.
    Otherwise, uses the center 30% of each frame as a rough forehead proxy.
    """
    n_frames = len(video_frames)
    rgb_trace = np.zeros((n_frames, 3), dtype=np.float64)

    for i, frame in enumerate(video_frames):
        if face_roi_func is not None:
            roi = face_roi_func(frame)
            if roi is not None and roi.size > 0:
                mean_bgr = np.mean(roi.reshape(-1, 3).astype(np.float64), axis=0)
                rgb_trace[i] = mean_bgr[::-1]  # BGR -> RGB
            else:
                rgb_trace[i] = [0, 0, 0]
        else:
            # Rough center crop as forehead proxy
            h, w = frame.shape[:2]
            y1, y2 = int(h * 0.15), int(h * 0.45)
            x1, x2 = int(w * 0.3), int(w * 0.7)
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                mean_bgr = np.mean(roi.reshape(-1, 3).astype(np.float64), axis=0)
                rgb_trace[i] = mean_bgr[::-1]
            else:
                rgb_trace[i] = [0, 0, 0]

    return rgb_trace


def evaluate_hr_on_dataset(
    dataset_type: str = "ubfc-rppg",
    data_dir: str = "data/ubfc-rppg",
    manifest_path: Optional[str] = None,
    methods: Optional[List[str]] = None,
    output_dir: str = "results/hr_evaluation",
    use_face_detector: bool = True,
) -> Dict[str, HREvaluationResult]:
    """
    Evaluate heart rate methods on a dataset.

    Args:
        dataset_type: "ubfc-rppg" or "custom"
        data_dir: Path to dataset directory
        manifest_path: Path to manifest (custom dataset only)
        methods: List of method names to evaluate (default: ["pos", "chrom", "green"])
        output_dir: Where to save results
        use_face_detector: Whether to use face detection for ROI extraction

    Returns:
        Dictionary mapping method name to HREvaluationResult
    """
    if methods is None:
        methods = ["pos", "chrom", "green"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    if dataset_type == "ubfc-rppg":
        loader = UBFCRPPGLoader(data_dir=data_dir)
    else:
        loader = CustomRecordingLoader(data_dir=data_dir, manifest_path=manifest_path)

    if not loader.is_available() if hasattr(loader, 'is_available') else not Path(data_dir).exists():
        logger.error(f"Dataset not available at {data_dir}")
        return {}

    # Optionally set up face detector for ROI extraction
    face_roi_func = None
    if use_face_detector:
        try:
            from src.contactless.face_detection.detect import FaceDetector
            detector = FaceDetector()

            def face_roi_func(frame):
                result = detector.detect(frame)
                if result.detected and result.forehead_roi is not None:
                    return result.forehead_roi
                return None
        except ImportError:
            logger.warning("FaceDetector not available; using center-crop ROI")
            use_face_detector = False

    # Initialize extractors
    extractors = {}
    for method_name in methods:
        if method_name not in METHOD_EXTRACTORS:
            logger.warning(f"Unknown method: {method_name}")
            continue
        extractors[method_name] = METHOD_EXTRACTORS[method_name]()

    # Evaluate each subject
    all_results = {m: {"predicted": [], "ground_truth": [], "confidences": [], "subject_ids": []}
                   for m in methods}

    if dataset_type == "ubfc-rppg":
        subject_ids = loader.list_subjects()
    else:
        subject_ids = loader.list_recordings()

    logger.info(f"Evaluating {len(subject_ids)} subjects with methods: {methods}")

    for subject_id in subject_ids:
        if dataset_type == "ubfc-rppg":
            subject_data = loader.load_subject(subject_id, load_video=True)
        else:
            subject_data = loader.load_recording(subject_id)

        if subject_data is None or subject_data.video_frames is None:
            logger.warning(f"Skipping {subject_id}: no video data")
            continue

        if subject_data.ground_truth_bpm is None:
            logger.warning(f"Skipping {subject_id}: no ground truth")
            continue

        # Extract RGB trace
        rgb_trace = _extract_rgb_trace(subject_data.video_frames, face_roi_func)

        # Skip subjects with too many zero frames
        zero_frames = np.sum(np.all(rgb_trace == 0, axis=1))
        if zero_frames > len(rgb_trace) * 0.3:
            logger.warning(f"Skipping {subject_id}: {zero_frames}/{len(rgb_trace)} zero frames")
            continue

        fps = subject_data.video_fps
        gt_bpm = subject_data.ground_truth_bpm

        # Evaluate each method
        for method_name, extractor in extractors.items():
            start_time = time.perf_counter()
            pred_bpm, pred_conf = extractor.extract_bpm_series(rgb_trace, fps)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if len(pred_bpm) == 0 or len(gt_bpm) == 0:
                continue

            # Align predictions with ground truth
            n = min(len(pred_bpm), len(gt_bpm))
            pred_aligned = pred_bpm[:n]
            gt_aligned = gt_bpm[:n]
            conf_aligned = pred_conf[:n]

            # Filter NaN
            valid = ~np.isnan(pred_aligned)
            if np.sum(valid) < 2:
                continue

            all_results[method_name]["predicted"].extend(pred_aligned[valid].tolist())
            all_results[method_name]["ground_truth"].extend(gt_aligned[valid].tolist())
            all_results[method_name]["confidences"].extend(conf_aligned[valid].tolist())
            all_results[method_name]["subject_ids"].extend(
                [subject_id] * int(np.sum(valid))
            )

    # Compute per-method evaluation results
    evaluation_results = {}

    for method_name in methods:
        results = all_results[method_name]
        if len(results["predicted"]) == 0:
            logger.warning(f"No valid predictions for method {method_name}")
            continue

        predicted = np.array(results["predicted"])
        ground_truth = np.array(results["ground_truth"])
        confidences = np.array(results["confidences"])

        eval_result = evaluate_hr_method(predicted, ground_truth, method_name)
        evaluation_results[method_name] = eval_result

        # Generate plots
        plot_predicted_vs_ground_truth(
            predicted, ground_truth,
            title=f"Heart Rate: {method_name.upper()} — Predicted vs Ground Truth",
            xlabel="Ground Truth (BPM)",
            ylabel=f"{method_name.upper()} Predicted (BPM)",
            save_path=str(output_path / f"predicted_vs_gt_{method_name}.png"),
            method_name=method_name,
        )

        plot_bland_altman(
            predicted, ground_truth,
            title=f"Heart Rate: {method_name.upper()} — Bland-Altman",
            save_path=str(output_path / f"bland_altman_{method_name}.png"),
            method_name=method_name,
        )

        plot_error_histogram(
            predicted, ground_truth,
            title=f"Heart Rate: {method_name.upper()} — Error Distribution",
            save_path=str(output_path / f"error_hist_{method_name}.png"),
            method_name=method_name,
        )

        plot_confidence_vs_error(
            confidences, predicted - ground_truth,
            title=f"Heart Rate: {method_name.upper()} — Confidence vs Error",
            save_path=str(output_path / f"confidence_vs_error_{method_name}.png"),
            method_name=method_name,
        )

        # Log summary
        logger.info(f"\n{'='*50}")
        logger.info(f"Heart Rate Evaluation: {method_name.upper()}")
        logger.info(f"{'='*50}")
        logger.info(f"  N valid:    {eval_result.n_valid} / {eval_result.n_total}")
        logger.info(f"  MAE:        {eval_result.mae:.2f} BPM")
        logger.info(f"  RMSE:       {eval_result.rmse:.2f} BPM")
        logger.info(f"  Pearson r:  {eval_result.pearson_r:.3f}")
        logger.info(f"  R²:         {eval_result.r2:.3f}")
        logger.info(f"  Bias:       {eval_result.bland_altman_bias:.2f} BPM")
        logger.info(f"  LoA:        [{eval_result.bland_altman_loa_lower:.2f}, "
                     f"{eval_result.bland_altman_loa_upper:.2f}]")
        logger.info(f"  ±5 BPM:     {eval_result.percentage_within_5bpm:.1%}")
        logger.info(f"  ±10 BPM:    {eval_result.percentage_within_10bpm:.1%}")
        logger.info(f"  Failure:    {eval_result.failure_rate:.1%}")

    # Generate method comparison plot if multiple methods
    if len(evaluation_results) > 1:
        comparison_metrics = {
            "MAE": [evaluation_results[m].mae for m in evaluation_results],
            "RMSE": [evaluation_results[m].rmse for m in evaluation_results],
            "r": [evaluation_results[m].pearson_r for m in evaluation_results],
            "±5 BPM": [evaluation_results[m].percentage_within_5bpm for m in evaluation_results],
        }
        plot_method_comparison(
            list(evaluation_results.keys()),
            comparison_metrics,
            title="Heart Rate Method Comparison",
            save_path=str(output_path / "method_comparison.png"),
        )

    # Save results to JSON
    report = {}
    for method_name, result in evaluation_results.items():
        report[method_name] = {
            "method": result.method,
            "mae": result.mae,
            "rmse": result.rmse,
            "pearson_r": result.pearson_r,
            "r2": result.r2,
            "bland_altman_bias": result.bland_altman_bias,
            "bland_altman_loa_lower": result.bland_altman_loa_lower,
            "bland_altman_loa_upper": result.bland_altman_loa_upper,
            "percentage_within_5bpm": result.percentage_within_5bpm,
            "percentage_within_10bpm": result.percentage_within_10bpm,
            "failure_rate": result.failure_rate,
            "n_valid": result.n_valid,
            "n_total": result.n_total,
        }

    report_path = output_path / "hr_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nFull report saved to: {report_path}")

    return evaluation_results


def main():
    parser = argparse.ArgumentParser(description="Heart Rate Evaluation Harness")
    parser.add_argument("--dataset", choices=["ubfc-rppg", "custom"], default="ubfc-rppg")
    parser.add_argument("--data-dir", default="data/ubfc-rppg")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--methods", default="pos,chrom,green",
                        help="Comma-separated method names")
    parser.add_argument("--output-dir", default="results/hr_evaluation")
    parser.add_argument("--no-face-detection", action="store_true")

    args = parser.parse_args()
    methods = [m.strip() for m in args.methods.split(",")]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    evaluate_hr_on_dataset(
        dataset_type=args.dataset,
        data_dir=args.data_dir,
        manifest_path=args.manifest,
        methods=methods,
        output_dir=args.output_dir,
        use_face_detector=not args.no_face_detection,
    )


if __name__ == "__main__":
    main()
