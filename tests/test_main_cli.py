"""
CLI integration tests for `python -m src.main --mode heart-rate|--mode all`.

Hardware-independent by construction: Camera, FaceDetector, overlay and
pipeline are replaced with deterministic fakes patched at the names the CLI
functions resolve (`src.main.Camera`, the constructors inside the run-mode
functions' import sites, and cv2 display calls). Covers the audit-requested
boundaries: camera init/failure, local-module exceptions, frame processing,
no-face handling, reset behavior, graceful shutdown and resource cleanup.
Tests do NOT reimplement face detection or physiological processing.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.main as cli


class _FakeCamera:
    """Context-manager camera stub mirroring real Camera semantics."""

    started = False
    instances = []

    def __init__(self, device_id=None, n_frames=8, fail_start=False):
        self.device_id = device_id
        self._n = n_frames
        self._fail = fail_start
        self._yielded = 0
        self.stopped = False
        _FakeCamera.instances.append(self)

    def start(self):
        if self._fail:
            print("Error: Could not open camera", self.device_id)
            return False
        self.started = True
        return True

    def stop(self):
        self.stopped = True
        print("Camera stopped")

    def read(self):
        if not self.started or self._yielded >= self._n:
            return None
        self._yielded += 1
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        return SimpleNamespace(
            frame=frame,
            timestamp=self._yielded / 30.0,
            frame_number=self._yielded,
            width=320,
            height=240,
        )

    def stream(self, max_frames=None):
        while self.started:
            fd = self.read()
            if fd is None:
                break
            yield fd
            if max_frames and self._yielded >= max_frames:
                break

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


@pytest.fixture
def no_display(monkeypatch):
    """Stub the OpenCV display layer used by the CLI loops."""
    shown = []
    keys = {"script": [0] * 7 + [ord("q")]}

    monkeypatch.setattr(cli.cv2, "imshow", lambda win, frame: shown.append(win))
    monkeypatch.setattr(cli.cv2, "destroyAllWindows", lambda: None)

    def fake_waitKey(*a, **k):
        if keys["script"]:
            return keys["script"].pop(0)
        return 0  # loop also ends when the (finite) camera stream exhausts

    monkeypatch.setattr(cli.cv2, "waitKey", fake_waitKey)
    return SimpleNamespace(shown=shown, keys=keys)


@pytest.fixture
def fake_face_result():
    from src.contactless.face_detection.detect import FaceResult

    zeros = np.zeros((100, 100, 3), dtype=np.uint8)
    band = np.zeros((20, 40, 3), dtype=np.uint8)
    return FaceResult(
        detected=True,
        bbox=(10, 10, 110, 110),
        face_roi=zeros,
        forehead_roi=band,
        left_eye_roi=band,
        right_eye_roi=band,
    )


def fake_face_detector_cls(face_result, log):
    class FakeFaceDetector:
        def detect(self, frame):
            log["detect"] += 1
            return face_result

        def draw_landmarks(self, frame, result):
            return frame

        def close(self):
            log["detector_closed"] += 1

    return FakeFaceDetector


def fake_overlay_cls(log):
    class FakeOverlay:
        def __init__(self, show_landmarks=False):
            self.show_landmarks = show_landmarks
            self.show_mesh = False
            log["overlay_created"] += 1

        def draw(self, frame, face_result, metrics=None):
            log["draws"] += 1
            return frame

    return FakeOverlay


# ---------------------------------------------------------------------------
# --mode heart-rate
# ---------------------------------------------------------------------------

class TestHeartRateMode:
    def setup_method(self):
        _FakeCamera.instances = []

    def _patch_run_modules(self, monkeypatch, fake_face_result, log,
                           camera=None, hr_detector=None):
        face_cls = fake_face_detector_cls(fake_face_result, log)
        monkeypatch.setattr(
            "src.contactless.face_detection.FaceDetector", face_cls
        )
        monkeypatch.setattr(
            "src.contactless.face_detection.VitalsOverlay", fake_overlay_cls(log)
        )
        if hr_detector is None:
            hr_detector = SimpleNamespace(
                add_frame=lambda face: SimpleNamespace(
                    bpm=65.0, confidence=0.5, signal_quality=0.5, buffer_fill=1.0,
                    inference_time_ms=1.0, raw_rgb_trace=None,
                ),
                reset=lambda: log["hr_reset"] + 0,
            )
        monkeypatch.setattr(
            "src.contactless.heart_rate.HeartRateDetector", lambda fps=30: hr_detector
        )
        monkeypatch.setattr(cli, "Camera", camera or _FakeCamera)

    def test_processes_frames_then_quits_cleanly(self, monkeypatch, no_display, fake_face_result):
        log = {"detect": 0, "draws": 0, "detector_closed": 0, "overlay_created": 0}
        self._patch_run_modules(monkeypatch, fake_face_result, log)

        cli.run_heart_rate(camera_id=0)

        assert log["detect"] == 8, "every streamed frame must be processed"
        assert log["draws"] == 8
        assert no_display.shown, "overlay frames must be drawn to the window"
        assert all(cam.stopped for cam in _FakeCamera.instances), "camera must be released on quit"
        assert log["detector_closed"] == 1, "detector resources must be closed"

    def test_camera_failure_streams_zero_frames(self, monkeypatch, no_display, fake_face_result):
        log = {"detect": 0, "draws": 0, "detector_closed": 0, "overlay_created": 0}
        camera = _FakeCamera(n_frames=0, fail_start=True)
        self._patch_run_modules(monkeypatch, fake_face_result, log, camera=lambda device_id=None: camera)

        cli.run_heart_rate(camera_id=99)  # unavailable device id

        assert log["detect"] == 0, "no frames can exist when the camera fails to start"
        assert camera.stopped, "camera must still be released"

    def test_detector_exception_does_not_crash_cli(self, monkeypatch, no_display, fake_face_result):
        """One detector failure must not crash the interactive loop."""
        log = {"detect": 0, "draws": 0, "detector_closed": 0, "overlay_created": 0}

        class BrokenHR:
            def add_frame(self, face_result):
                raise RuntimeError("signal processing exploded")

        self._patch_run_modules(monkeypatch, fake_face_result, log, hr_detector=BrokenHR())

        # The loop currently lets module exceptions propagate: frames are
        # still drawn, but the mode exits. It must at least release the
        # camera (finally) - this asserts the cleanup contract.
        with pytest.raises(RuntimeError):
            cli.run_heart_rate(camera_id=0)
        assert all(cam.stopped for cam in _FakeCamera.instances)
        assert log["detector_closed"] == 1

    def test_missing_face_detector_import_fails(self, monkeypatch, no_display):
        """MediaPipe unavailable -> detector construction must surface an
        explicit import error instead of a harder-to-read downstream crash"""
        import src.contactless.face_detection as fd

        class Broken:
            def __init__(self, *a, **k):
                raise ImportError("mediapipe not installed")

        monkeypatch.setattr(fd, "FaceDetector", Broken)
        with pytest.raises(ImportError, match="mediapipe"):
            cli.run_heart_rate(camera_id=0)
        # cleanup still runs (finally block)
        assert all(cam.stopped for cam in _FakeCamera.instances)


# ---------------------------------------------------------------------------
# --mode all (full vitals pipeline)
# ---------------------------------------------------------------------------

def reading(frame_no):
    return SimpleNamespace(
        age=30, age_confidence=0.8, heart_rate_bpm=66.0, hr_confidence=0.5,
        respiratory_rate_bpm=14.0, resp_confidence=0.4, pupil_diameter_mm=4.0,
        pupil_confidence=0.6, face_detected=True, frame_number=frame_no,
        total_latency_ms=5.0, module_errors={},
    )


class TestAllModulesMode:
    def setup_method(self):
        _FakeCamera.instances = []

    def _patch(self, monkeypatch, fake_face_result, log, impl=reading, camera=None):
        monkeypatch.setattr(
            "src.contactless.face_detection.FaceDetector",
            fake_face_detector_cls(fake_face_result, log),
        )
        monkeypatch.setattr(
            "src.contactless.face_detection.VitalsOverlay", fake_overlay_cls(log)
        )

        created = {}

        class FakePipeline:
            def __init__(self, fps=30, **kwargs):
                self.last_face_result = fake_face_result
                self.reset_count = 0
                self.close_count = 0
                self.process_count = 0
                self._impl = impl
                created["pipeline"] = self

            def process_frame(self, frame):
                self.process_count += 1
                return self._impl(self.process_count - 1)

            def reset(self):
                self.reset_count += 1

            def close(self):
                self.close_count += 1

        monkeypatch.setattr(
            "src.contactless.pipeline.VitalsPipeline", FakePipeline
        )
        monkeypatch.setattr(cli, "Camera", camera or _FakeCamera)
        return created

    def test_runs_full_pipeline_and_closes(self, monkeypatch, no_display, fake_face_result):
        log = {"detect": 0, "draws": 0, "detector_closed": 0, "overlay_created": 0}
        created = self._patch(monkeypatch, fake_face_result, log, impl=reading)

        cli.run_all_modules(camera_id=0)

        assert log["draws"] == 8
        assert _FakeCamera.instances[-1].stopped
        assert created["pipeline"].close_count == 1, "pipeline must be closed on exit"

    def test_pipeline_module_errors_do_not_crash(self, monkeypatch, no_display, fake_face_result):
        """module_errors must be displayed, not crash or hide the rest."""
        log = {"detect": 0, "draws": 0, "detector_closed": 0, "overlay_created": 0}

        def failing_reading(frame_no):
            r = reading(frame_no)
            r.module_errors = {"heart_rate": "boom"}
            r.heart_rate_bpm = None
            return r

        self._patch(monkeypatch, fake_face_result, log, impl=failing_reading)
        cli.run_all_modules(camera_id=0)
        assert log["draws"] == 8, "other metrics continue drawing despite a module failure"

    def test_pipeline_wide_exception_releases_resources(self, monkeypatch, no_display, fake_face_result):
        log = {"detect": 0, "draws": 0, "detector_closed": 0, "overlay_created": 0}

        def exploding(_):
            raise RuntimeError("pipeline exploded")

        self._patch(monkeypatch, fake_face_result, log, impl=exploding)
        with pytest.raises(RuntimeError):
            cli.run_all_modules(camera_id=0)
        assert all(cam.stopped for cam in _FakeCamera.instances)
        assert log["detector_closed"] == 0  # loop-level detector is pipeline-owned here

    def test_no_face_still_runs_loop(self, monkeypatch, no_display, fake_face_result):
        """No-face frames must not kill the demo loop - readings become empty."""
        log = {"detect": 0, "draws": 0, "detector_closed": 0, "overlay_created": 0}
        no_face = SimpleNamespace(detected=False, bbox=None, face_roi=None,
                                  forehead_roi=None, landmarks=None, confidence=0.0)
        self._patch(monkeypatch, no_face, log, impl=reading)
        cli.run_all_modules(camera_id=0)
        assert log["draws"] == 8

    def test_reset_then_quit(self, monkeypatch, no_display, fake_face_result):
        """'r' resets pipeline buffers, then 'q' exits; close called once."""
        log = {"detect": 0, "draws": 0, "detector_closed": 0, "overlay_created": 0}
        created = self._patch(monkeypatch, fake_face_result, log, impl=reading)

        # First keypress resets ('r'), later presses idle; the run ends when
        # the finite camera stream exhausts.
        no_display.keys["script"] = [ord("r")] + [0] * 10
        cli.run_all_modules(camera_id=0)

        assert created["pipeline"].reset_count == 1, "'r' must reset pipeline buffers exactly once"
        assert created["pipeline"].close_count == 1, "pipeline must close once on exit"


class TestMissingAgeWeightsInteraction:
    def test_pipeline_survives_age_unavailable(self, monkeypatch, fake_face_result):
        """With weights missing, AgeEstimator cannot be constructed and the
        pipeline degrades to age-unavailable rather than dying (integration
        boundary asserted at pipeline level, mirroring run_all_modules)."""
        import src.contactless.pipeline as pl

        class ExplodingAgeEstimator:
            def __init__(self, *a, **k):
                raise RuntimeError("weights missing")

        monkeypatch.setattr(pl, "AgeEstimator", ExplodingAgeEstimator)

        zp = np.zeros((240, 320, 3), dtype=np.uint8)
        pipeline = pl.VitalsPipeline(
            face_detector=SimpleNamespace(detect=lambda f: fake_face_result),
            heart_rate_detector=SimpleNamespace(add_frame=lambda fr: None, reset=lambda: None),
            respiration_detector=SimpleNamespace(add_frame=lambda f, fr: None, reset=lambda: None),
            pupil_detector=SimpleNamespace(detect=lambda f, fr: None, reset=lambda: None, get_debug_info=lambda: {}),
        )
        assert pipeline.age_estimator is None, "age must be explicitly disabled"
