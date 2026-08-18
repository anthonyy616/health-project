"""
Unified Vitals Pipeline (Stage 5)
=================================
Wraps face detection, age estimation, heart rate, respiration, and pupil
detection behind a single `process_frame()` call, with frame skipping
(age and pupil only) and optional threading.

CRITICAL CONSTRAINT (from the Stage 5 plan - do not skip these two):
HeartRateDetector and RespirationDetector are built around an `fps`
parameter that is used directly in their Butterworth filter cutoffs and
FFT/peak-timing math. If frames were skipped before `add_frame()`, the
*actual* sample rate feeding their buffers would drop while the detectors
still assume 30fps - every BPM calculation would be silently wrong
(aliased). Therefore `heart_rate_detector.add_frame()` and
`respiration_detector.add_frame()` are called on EVERY captured frame.

Frame skipping is applied ONLY to modules whose math is sample-rate
independent:
  - Age estimation: single-shot regression, age barely changes
    frame-to-frame (default: every 5th frame).
  - Pupil detection: single-shot per frame, but its EMA smoothing and
    blink detection are somewhat rate-sensitive, so the default is to run
    it every frame (pupil_skip_frames=1). Only raise the skip count if
    testing shows no dilation-trace artifacts.

One module's exception must never crash the pipeline or prevent the other
modules from running - failures are collected into `module_errors` on the
returned VitalsReading.
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Callable

import numpy as np

from src.contactless.face_detection.detect import FaceDetector
from src.contactless.age_estimation.estimator import AgeEstimator
from src.contactless.heart_rate import HeartRateDetector
from src.contactless.respiration import RespirationDetector
from src.contactless.pupil_detection import PupilDetector

logger = logging.getLogger(__name__)

# Default number of executor worker threads (one per parallelizable module:
# heart rate, respiration, age, pupil). Constructor parameter, not hardcoded
# magic - this is the default value only.
DEFAULT_THREAD_POOL_WORKERS = 4


@dataclass
class VitalsReading:
    """
    Flat, JSON-serializable container for one frame's full vitals reading.

    This is the shape the dashboard API and database layer will consume in
    Phase 3 - `to_dict()` exists now so the shape is settled before anything
    depends on it.
    """
    age: Optional[int] = None
    age_confidence: float = 0.0
    heart_rate_bpm: Optional[float] = None
    hr_confidence: float = 0.0
    respiratory_rate_bpm: Optional[float] = None
    resp_confidence: float = 0.0
    pupil_diameter_mm: Optional[float] = None
    pupil_confidence: float = 0.0
    face_detected: bool = False
    frame_number: int = 0  # 0-indexed: first processed frame is frame 0
    total_latency_ms: float = 0.0
    # Maps module name -> error string for any module that raised this
    # frame (empty dict if none). Surfaces partial failures without crashing.
    module_errors: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-friendly dict (all fields are primitives already)."""
        return asdict(self)


class VitalsPipeline:
    """
    Unified pipeline wrapping the five contactless modules.

    Dependency-injectable for testing: pass lightweight fakes for any
    component; anything left as None is constructed as the real default.
    """

    def __init__(
        self,
        face_detector: Optional[FaceDetector] = None,
        age_estimator: Optional[AgeEstimator] = None,
        heart_rate_detector: Optional[HeartRateDetector] = None,
        respiration_detector: Optional[RespirationDetector] = None,
        pupil_detector: Optional[PupilDetector] = None,
        age_skip_frames: int = 5,
        pupil_skip_frames: int = 1,  # 1 = run every frame (see module docstring)
        use_threading: bool = False,
        fps: int = 30,
        thread_pool_workers: int = DEFAULT_THREAD_POOL_WORKERS,
    ):
        """
        Initialize the pipeline.

        Args:
            face_detector: FaceDetector instance, or None to build the default
            age_estimator: AgeEstimator instance, or None to build the default.
                NOTE: the default is built with model_type="mobilenet" to match
                the actual weights in best_model.pt (Stage 4 finding - the
                plan's EfficientNet assumption does not match the on-disk
                weights; see scripts/optimization/export_age_onnx.py docstring).
            heart_rate_detector: HeartRateDetector, or None for the default
            respiration_detector: RespirationDetector, or None for the default
            pupil_detector: PupilDetector, or None for the default
            age_skip_frames: run age estimation every Nth frame (0-indexed:
                frame 0 runs immediately)
            pupil_skip_frames: run pupil detection every Nth frame (default 1)
            use_threading: run the four dependent modules on a
                ThreadPoolExecutor (see module docstring re: GIL caveats)
            fps: camera frames per second (passed to the HR/resp defaults)
            thread_pool_workers: executor size when use_threading=True
        """
        self.fps = fps
        self.age_skip_frames = age_skip_frames
        self.pupil_skip_frames = pupil_skip_frames
        self.use_threading = use_threading

        # Dependency injection with real defaults
        self.face_detector = face_detector if face_detector is not None else FaceDetector()

        if age_estimator is not None:
            self.age_estimator = age_estimator
        else:
            try:
                self.age_estimator = AgeEstimator(model_type="mobilenet")
            except RuntimeError as e:
                # e.g. weights present but for a different architecture than
                # mobilenet. Degrade gracefully: pipeline works, age is off.
                logger.warning(f"AgeEstimator could not be constructed, age module disabled: {e}")
                self.age_estimator = None

        self.heart_rate_detector = (
            heart_rate_detector if heart_rate_detector is not None
            else HeartRateDetector(fps=fps)
        )
        self.respiration_detector = (
            respiration_detector if respiration_detector is not None
            else RespirationDetector(fps=fps)
        )
        self.pupil_detector = (
            pupil_detector if pupil_detector is not None else PupilDetector()
        )

        # 0-indexed frame counter: the first processed frame is frame 0,
        # so frame 0 % age_skip_frames == 0 runs age immediately.
        self._frame_counter = 0

        # Cached results so skipped frames don't flash to "--"
        self._last_age_result = None
        self._last_pupil_result = None

        # Most recent FaceResult (display aid for overlay drawing - the
        # VitalsReading deliberately does not carry the bbox).
        self.last_face_result = None

        self._executor: Optional[ThreadPoolExecutor] = None
        if use_threading:
            self._executor = ThreadPoolExecutor(max_workers=thread_pool_workers)

        logger.info(
            f"VitalsPipeline initialized (threading={use_threading}, "
            f"age_skip={age_skip_frames}, pupil_skip={pupil_skip_frames})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> VitalsReading:
        """Process one frame through the full pipeline (single entry point).

        Args:
            frame: BGR image from OpenCV

        Returns:
            VitalsReading with whatever succeeded this frame and any module
            errors collected.
        """
        return self._process_frame(frame, use_threading=self.use_threading)

    def process_frame_threaded(self, frame: np.ndarray) -> VitalsReading:
        """Force the threaded execution path.

        Only meaningful when use_threading=True was set at construction; if
        not, logs a warning and falls back to sequential execution.
        """
        if not self.use_threading:
            logger.warning(
                "process_frame_threaded() called but use_threading=False - "
                "falling back to sequential execution."
            )
            return self._process_frame(frame, use_threading=False)
        return self._process_frame(frame, use_threading=True)

    def reset(self) -> None:
        """
        Reset all stateful sub-detectors (heart rate, respiration, pupil),
        the frame counter, and cached age/pupil results. Age estimation and
        face detection have no persistent state to reset.
        """
        self.heart_rate_detector.reset()
        self.respiration_detector.reset()
        self.pupil_detector.reset()
        self._frame_counter = 0
        self._last_age_result = None
        self._last_pupil_result = None
        logger.info("VitalsPipeline reset")

    def get_debug_info(self) -> dict:
        """Aggregate debug info from each sub-detector that exposes it."""
        info = {
            "frame_counter": self._frame_counter,
            "age_skip_frames": self.age_skip_frames,
            "pupil_skip_frames": self.pupil_skip_frames,
            "use_threading": self.use_threading,
            "age_estimator_available": self.age_estimator is not None,
        }
        for name, component in (
            ("face_detection", self.face_detector),
            ("age", self.age_estimator),
            ("heart_rate", self.heart_rate_detector),
            ("respiration", self.respiration_detector),
            ("pupil", self.pupil_detector),
        ):
            if component is not None and hasattr(component, "get_debug_info"):
                info[name] = component.get_debug_info()
        return info

    def close(self) -> None:
        """
        Release resources: shuts down the threaded executor if one was
        created. Call when the pipeline is no longer needed (e.g. on exit of
        --mode all). Component lifecycle stays with their owners.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
            logger.info("VitalsPipeline executor shut down")

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray, use_threading: bool) -> VitalsReading:
        start_time = time.perf_counter()
        errors: Dict[str, str] = {}

        frame_number = self._frame_counter
        self._frame_counter += 1

        # 1. Face detection - always, sequential, both modes. Everything
        #    downstream depends on its output.
        try:
            face_result = self.face_detector.detect(frame)
        except Exception as e:
            logger.exception(f"Face detection failed: {e}")
            self.last_face_result = None
            return self._build_reading(
                frame_number=frame_number,
                face_detected=False,
                errors={"face_detection": str(e)},
                start_time=start_time,
            )

        self.last_face_result = face_result

        # No face -> nothing downstream can run. Not an error, just no data.
        if not face_result.detected:
            return self._build_reading(
                frame_number=frame_number,
                face_detected=False,
                errors=errors,
                start_time=start_time,
            )

        # 2-6. Build the dependent-module jobs. HR and respiration are ALWAYS
        #      included (critical constraint, see module docstring); age and
        #      pupil are included only on their skip-cadence frames.
        jobs: Dict[str, Callable[[], object]] = {}

        if self.heart_rate_detector is not None:
            jobs["heart_rate"] = lambda: self.heart_rate_detector.add_frame(face_result)

        if self.respiration_detector is not None:
            jobs["respiration"] = lambda: self.respiration_detector.add_frame(frame, face_result)

        if (
            self.age_estimator is not None
            and face_result.face_roi is not None
            and frame_number % self.age_skip_frames == 0
        ):
            jobs["age"] = lambda: self.age_estimator.estimate(face_result.face_roi)

        if (
            self.pupil_detector is not None
            and frame_number % self.pupil_skip_frames == 0
        ):
            jobs["pupil"] = lambda: self.pupil_detector.detect(frame, face_result)

        # 7. Execute all jobs - either inline (sequential) or on the shared
        #    executor (threaded). One module's exception must never prevent
        #    the others from running: each is caught and recorded.
        results: Dict[str, object] = {}
        if use_threading and self._executor is not None:
            futures = {name: self._executor.submit(fn) for name, fn in jobs.items()}
            for name in jobs:
                try:
                    results[name] = futures[name].result()
                except Exception as e:
                    logger.exception(f"Pipeline module '{name}' failed: {e}")
                    errors[name] = str(e)
        else:
            for name, fn in jobs.items():
                try:
                    results[name] = fn()
                except Exception as e:
                    logger.exception(f"Pipeline module '{name}' failed: {e}")
                    errors[name] = str(e)

        # Cache handling: on skip frames (or after a module failure) reuse
        # the last good result so the display doesn't flash to "--".
        age_result = results.get("age")
        if "age" in results:
            self._last_age_result = age_result
        else:
            age_result = self._last_age_result

        pupil_result = results.get("pupil")
        if "pupil" in results:
            self._last_pupil_result = pupil_result
        else:
            pupil_result = self._last_pupil_result

        hr_result = results.get("heart_rate")
        resp_result = results.get("respiration")

        return self._build_reading(
            frame_number=frame_number,
            face_detected=True,
            errors=errors,
            start_time=start_time,
            age_result=age_result,
            hr_result=hr_result,
            resp_result=resp_result,
            pupil_result=pupil_result,
        )

    def _build_reading(
        self,
        frame_number: int,
        face_detected: bool,
        errors: Dict[str, str],
        start_time: float,
        age_result=None,
        hr_result=None,
        resp_result=None,
        pupil_result=None,
    ) -> VitalsReading:
        """Assemble a VitalsReading from whichever sub-results are available."""
        return VitalsReading(
            age=age_result.age if age_result is not None else None,
            age_confidence=age_result.confidence if age_result is not None else 0.0,
            heart_rate_bpm=hr_result.bpm if hr_result is not None else None,
            hr_confidence=hr_result.confidence if hr_result is not None else 0.0,
            respiratory_rate_bpm=resp_result.bpm if resp_result is not None else None,
            resp_confidence=resp_result.confidence if resp_result is not None else 0.0,
            pupil_diameter_mm=pupil_result.average_mm if pupil_result is not None else None,
            pupil_confidence=pupil_result.confidence if pupil_result is not None else 0.0,
            face_detected=face_detected,
            frame_number=frame_number,
            total_latency_ms=(time.perf_counter() - start_time) * 1000,
            module_errors=errors,
        )
