#Tests for Stage 5: Unified VitalsPipeline
#
# These tests verify the pipeline's ORCHESTRATION logic (frame skipping,
# per-module error isolation, result assembly) using lightweight fakes -
# no real models, no camera, no ML inference. The dependency-injectable
# constructor makes this possible.

import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Ensure ✓ checkmarks print on Windows consoles with cp1252 default encoding
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.contactless.pipeline import VitalsPipeline
from src.contactless.face_detection.detect import FaceResult
from src.contactless.age_estimation.estimator import AgeEstimationResult
from src.contactless.heart_rate import HeartRateResult
from src.contactless.respiration import RespirationResult
from src.contactless.pupil_detection import PupilResult

MODEL_WEIGHTS_PATH = project_root / "models/weights/age_detection/best_model.pt"

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Fakes - match each real component's public interface, return fixed values
# ---------------------------------------------------------------------------

class FakeFaceDetector:
    def __init__(self, detected=True):
        self.calls = 0
        self.detected = detected

    def detect(self, frame):
        self.calls += 1
        return FaceResult(
            detected=self.detected,
            bbox=(10, 10, 100, 100),
            face_roi=np.zeros((224, 224, 3), dtype=np.uint8),
            confidence=0.95,
        )

    def close(self):
        pass


class FakeAgeEstimator:
    def __init__(self):
        self.calls = 0

    def estimate(self, face_roi):
        self.calls += 1
        return AgeEstimationResult(age=24, confidence=0.9, raw_prediction=24.0, inference_time_ms=1.0)

    def is_loaded(self):
        return True


class FakeHeartRateDetector:
    def __init__(self):
        self.calls = 0
        self.resets = 0

    def add_frame(self, face_result):
        self.calls += 1
        return HeartRateResult(
            bpm=71.0, confidence=0.8, signal_quality=0.5,
            buffer_fill=1.0, inference_time_ms=1.0, raw_rgb_trace=None
        )

    def reset(self):
        self.resets += 1

    def get_debug_info(self):
        return {"buffer_fill": 1.0}


class FakeFailingHeartRateDetector(FakeHeartRateDetector):
    """add_frame raises - used for the error-isolation test."""

    def add_frame(self, face_result):
        self.calls += 1
        raise RuntimeError("boom")


class FakeRespirationDetector:
    def __init__(self):
        self.calls = 0
        self.resets = 0

    def add_frame(self, frame, face_result):
        self.calls += 1
        return RespirationResult(
            bpm=15.0, confidence=0.7, signal_quality=0.5,
            buffer_fill=1.0, inference_time_ms=1.0, raw_motion_trace=None
        )

    def reset(self):
        self.resets += 1

    def get_debug_info(self):
        return {"buffer_fill": 1.0}


class FakePupilDetector:
    def __init__(self):
        self.calls = 0
        self.resets = 0

    def detect(self, frame, face_result):
        self.calls += 1
        return PupilResult(
            left_pupil_mm=4.3, right_pupil_mm=4.4, average_mm=4.35,
            dilation_change=0.12, confidence=0.74, is_blinking=False,
            inference_time_ms=1.0
        )

    def reset(self):
        self.resets += 1

    def get_debug_info(self):
        return {"baseline_mm": 4.2}


def build_pipeline(**kwargs):
    """Pipeline with all fakes (any real module can be overridden via kwargs)."""
    defaults = dict(
        face_detector=FakeFaceDetector(),
        age_estimator=FakeAgeEstimator(),
        heart_rate_detector=FakeHeartRateDetector(),
        respiration_detector=FakeRespirationDetector(),
        pupil_detector=FakePupilDetector(),
    )
    defaults.update(kwargs)
    return VitalsPipeline(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pipeline_assembles_full_reading():
    """All fakes succeed -> all fields populated, no module errors."""
    pipeline = build_pipeline()
    reading = pipeline.process_frame(FRAME)

    assert reading.face_detected is True
    assert reading.age == 24
    assert reading.age_confidence == 0.9
    assert reading.heart_rate_bpm == 71.0
    assert reading.hr_confidence == 0.8
    assert reading.respiratory_rate_bpm == 15.0
    assert reading.resp_confidence == 0.7
    assert abs(reading.pupil_diameter_mm - 4.35) < 1e-9
    assert reading.pupil_confidence == 0.74
    assert reading.frame_number == 0  # 0-indexed: first frame is frame 0
    assert reading.module_errors == {}

    # JSON-friendly shape (dashboard/DB contract)
    json.dumps(reading.to_dict())

    print("✓ test_pipeline_assembles_full_reading passed")


def test_pipeline_age_skipped_on_non_multiple_frames():
    """age_skip_frames=5: age runs only on frame 0 of the first 4 calls;
    skipped frames return the cached age, not None."""
    age = FakeAgeEstimator()
    pipeline = build_pipeline(age_estimator=age, age_skip_frames=5)

    readings = [pipeline.process_frame(FRAME) for _ in range(4)]

    assert age.calls == 1, f"Age called {age.calls} times, expected 1 (only frame 0)"
    assert all(r.age == 24 for r in readings), "Skipped frames must reuse the cached age"
    assert readings[0].frame_number == 0

    # Frame 4 is still skipped (4 % 5 != 0)
    pipeline.process_frame(FRAME)
    assert age.calls == 1

    # Frame 5 runs age again (5 % 5 == 0)
    pipeline.process_frame(FRAME)
    assert age.calls == 2

    print("✓ test_pipeline_age_skipped_on_non_multiple_frames passed")


def test_pipeline_hr_and_resp_run_every_frame_regardless_of_age_skip():
    """THE critical constraint: heart rate and respiration are called on
    EVERY frame, never skipped, no matter the age skip cadence."""
    hr = FakeHeartRateDetector()
    resp = FakeRespirationDetector()
    pipeline = build_pipeline(
        heart_rate_detector=hr, respiration_detector=resp, age_skip_frames=5
    )

    for _ in range(6):
        pipeline.process_frame(FRAME)

    assert hr.calls == 6, f"HR called {hr.calls} times, expected 6 (every frame)"
    assert resp.calls == 6, f"Respiration called {resp.calls} times, expected 6 (every frame)"

    print("✓ test_pipeline_hr_and_resp_run_every_frame_regardless_of_age_skip passed")


def test_pipeline_isolates_module_failure():
    """A failing heart rate detector must not crash the pipeline or stop the
    other modules - the error lands in module_errors."""
    failing_hr = FakeFailingHeartRateDetector()
    pipeline = build_pipeline(heart_rate_detector=failing_hr)

    reading = pipeline.process_frame(FRAME)  # must NOT raise

    assert reading.module_errors == {"heart_rate": "boom"}
    assert reading.heart_rate_bpm is None
    assert reading.age == 24, "Age should still work despite HR failure"
    assert reading.respiratory_rate_bpm == 15.0, "Respiration should still work despite HR failure"
    assert abs(reading.pupil_diameter_mm - 4.35) < 1e-9, "Pupil should still work despite HR failure"

    print("✓ test_pipeline_isolates_module_failure passed")


def test_pipeline_face_not_detected_short_circuits_gracefully():
    """No face -> no downstream module is called; reading is clean defaults."""
    hr = FakeHeartRateDetector()
    resp = FakeRespirationDetector()
    age = FakeAgeEstimator()
    pupil = FakePupilDetector()
    pipeline = build_pipeline(
        face_detector=FakeFaceDetector(detected=False),
        age_estimator=age, heart_rate_detector=hr,
        respiration_detector=resp, pupil_detector=pupil,
    )

    reading = pipeline.process_frame(FRAME)  # must NOT raise

    assert reading.face_detected is False
    assert age.calls == 0 and hr.calls == 0 and resp.calls == 0 and pupil.calls == 0, \
        "No downstream module should be called when no face is detected"
    assert reading.age is None
    assert reading.heart_rate_bpm is None
    assert reading.respiratory_rate_bpm is None
    assert reading.pupil_diameter_mm is None
    assert reading.module_errors == {}

    print("✓ test_pipeline_face_not_detected_short_circuits_gracefully passed")


def test_pipeline_reset_clears_frame_counter_and_cached_values():
    """reset() zeroes the frame counter, clears cached age/pupil results, and
    resets all stateful sub-detectors."""
    hr = FakeHeartRateDetector()
    resp = FakeRespirationDetector()
    pupil = FakePupilDetector()
    pipeline = build_pipeline(
        heart_rate_detector=hr, respiration_detector=resp, pupil_detector=pupil
    )

    for _ in range(3):
        pipeline.process_frame(FRAME)

    assert pipeline._frame_counter == 3
    assert pipeline._last_age_result is not None, "Age should be cached after frame 0"
    assert pipeline._last_pupil_result is not None

    pipeline.reset()

    assert pipeline._frame_counter == 0
    assert pipeline._last_age_result is None, "Cached age must be cleared"
    assert pipeline._last_pupil_result is None, "Cached pupil result must be cleared"
    assert hr.resets == 1 and resp.resets == 1 and pupil.resets == 1, \
        "Stateful sub-detectors must be reset"

    # Next frame after reset is frame 0 again -> age runs immediately
    age_calls_before = pipeline.age_estimator.calls
    pipeline.process_frame(FRAME)
    assert pipeline._frame_counter == 1
    assert pipeline.age_estimator.calls == age_calls_before + 1

    print("✓ test_pipeline_reset_clears_frame_counter_and_cached_values passed")


def test_threaded_and_sequential_produce_equivalent_results():
    """Threading must change HOW the work runs, not WHAT gets computed."""
    sequential = build_pipeline()
    threaded = build_pipeline(use_threading=True)

    frames = [FRAME] * 12  # includes age-skip boundaries (frames 0, 5, 10)

    seq_readings = [sequential.process_frame(f) for f in frames]
    thr_readings = [threaded.process_frame(f) for f in frames]

    def comparable(reading):
        d = reading.to_dict()
        d.pop("total_latency_ms")  # timing legitimately differs between paths
        return d

    for i, (a, b) in enumerate(zip(seq_readings, thr_readings)):
        assert comparable(a) == comparable(b), \
            f"Frame {i}: threaded reading differs from sequential: {a} vs {b}"

    threaded.close()

    print("✓ test_threaded_and_sequential_produce_equivalent_results passed")


def test_default_constructor_builds_real_components():
    """VitalsPipeline() with no args builds real detectors without crashing.
    process_frame smoke test is skipped (gracefully) if the trained weights
    file isn't present."""
    pipeline = VitalsPipeline()

    assert pipeline.face_detector is not None
    assert pipeline.heart_rate_detector is not None
    assert pipeline.respiration_detector is not None
    assert pipeline.pupil_detector is not None

    if MODEL_WEIGHTS_PATH.exists():
        # Blank frame -> no face -> short-circuit, no crash, no module errors
        reading = pipeline.process_frame(np.zeros((240, 320, 3), dtype=np.uint8))
        assert reading.face_detected is False
        assert reading.module_errors == {}
    else:
        print("SKIPPED: process_frame smoke test (best_model.pt not present)")

    pipeline.close()

    print("✓ test_default_constructor_builds_real_components passed")


def run_all_tests():
    """Run all tests"""
    print("Running pipeline tests...")
    print("-" * 30)

    test_pipeline_assembles_full_reading()
    test_pipeline_age_skipped_on_non_multiple_frames()
    test_pipeline_hr_and_resp_run_every_frame_regardless_of_age_skip()
    test_pipeline_isolates_module_failure()
    test_pipeline_face_not_detected_short_circuits_gracefully()
    test_pipeline_reset_clears_frame_counter_and_cached_values()
    test_threaded_and_sequential_produce_equivalent_results()
    test_default_constructor_builds_real_components()

    print("-" * 30)
    print("All tests passed! ✅")


if __name__ == "__main__":
    run_all_tests()
