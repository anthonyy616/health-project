"""
Tests for the Phase G measurement envelope, timing semantics and
staleness policy (timestamp edge cases and stale-data behavior).
"""

import sys
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contactless.measurements import (
    UNIT_BY_METRIC,
    FreshnessPolicy,
    Measurement,
    MeasurementStatus,
    apply_freshness,
    measurement_from_reading,
)


def make_reading(frame_no=0, hr=66.0):
    return SimpleNamespace(
        age=30, age_confidence=0.8,
        heart_rate_bpm=hr, hr_confidence=0.5,
        respiratory_rate_bpm=14.0, resp_confidence=0.4,
        pupil_diameter_mm=4.0, pupil_confidence=0.6,
        face_detected=True, frame_number=frame_no,
        total_latency_ms=5.0, module_errors={},
    )


class TestMeasurementContract:
    def test_valid_adaptation(self):
        mono, utc = time.perf_counter_ns(), time.time()
        m = measurement_from_reading(make_reading(), "heart_rate", mono, utc)
        assert m.value == 66.0
        assert m.unit == UNIT_BY_METRIC["heart_rate"] == "bpm"
        assert m.status == MeasurementStatus.VALID
        assert m.acquired_mono_ns == mono
        assert m.acquired_at_utc_s == utc
        assert m.measurement_age_s() < 5.0

    def test_all_metrics_adapt(self):
        mono, utc = time.perf_counter_ns(), time.time()
        for metric in UNIT_BY_METRIC:
            m = measurement_from_reading(make_reading(), metric, mono, utc)
            assert m.status == MeasurementStatus.VALID
            assert m.value is not None

    def test_no_face_is_distinct_state(self):
        r = make_reading()
        r.face_detected = False
        mono, utc = time.perf_counter_ns(), time.time()
        m = measurement_from_reading(r, "heart_rate", mono, utc)
        assert m.status == MeasurementStatus.NO_FACE
        assert m.value is None, "NO_FACE must not masquerade as a number"

    def test_module_error_is_error_state(self):
        r = make_reading()
        r.module_errors = {"heart_rate": "boom"}
        mono, utc = time.perf_counter_ns(), time.time()
        m = measurement_from_reading(r, "heart_rate", mono, utc)
        assert m.status == MeasurementStatus.ERROR
        assert m.value is None
        assert "boom" in m.failure_reason

    def test_none_value_maps_insufficient_data(self):
        r = make_reading()
        r.heart_rate_bpm = None
        mono, utc = time.perf_counter_ns(), time.time()
        m = measurement_from_reading(r, "heart_rate", mono, utc)
        assert m.status == MeasurementStatus.INSUFFICIENT_DATA

    def test_to_dict_is_serializable(self):
        mono, utc = time.perf_counter_ns(), time.time()
        m = measurement_from_reading(make_reading(), "pupil_diameter", mono, utc)
        d = m.to_dict()
        assert d["status"] == "valid"
        assert d["unit"] == "mm"
        import json
        json.dumps(d)  # must not raise


class TestFreshness:
    def test_recent_measurement_not_stale(self):
        mono, utc = time.perf_counter_ns(), time.time()
        m = measurement_from_reading(make_reading(), "heart_rate", mono, utc)
        policy = FreshnessPolicy(heart_rate=10.0)
        assert not m.is_stale_under(policy)

    def test_old_measurement_stale(self):
        mono, utc = time.perf_counter_ns(), time.time()
        m = measurement_from_reading(make_reading(), "heart_rate", mono, utc - 30.0)
        assert m.is_stale_under(FreshnessPolicy(heart_rate=10.0))

    def test_apply_freshness_downgrades_but_does_not_mutate(self):
        mono, utc = time.perf_counter_ns(), time.time()
        fresh = measurement_from_reading(make_reading(), "heart_rate", mono, utc)
        old = measurement_from_reading(make_reading(), "respiratory_rate", mono, utc - 60.0)
        result = apply_freshness(
            {"heart_rate": fresh, "respiratory_rate": old},
            policy=FreshnessPolicy(heart_rate=10.0, respiratory_rate=20.0),
        )
        assert result["heart_rate"].status == MeasurementStatus.VALID
        assert result["respiratory_rate"].status == MeasurementStatus.STALE
        assert result["respiratory_rate"].value is None, "stale values must clear"
        # Originals untouched
        assert old.status == MeasurementStatus.VALID

    def test_per_metric_ttl(self):
        policy = FreshnessPolicy(heart_rate=5.0, age=60.0)
        assert policy.for_metric("heart_rate") == 5.0
        assert policy.for_metric("age") == 60.0

    def test_dropped_frames_and_resume_implies_staleness(self):
        """Long gap between acquisitions -> previous reading must expire."""
        mono, utc = time.perf_counter_ns(), time.time()
        m = measurement_from_reading(make_reading(), "heart_rate", mono, utc)
        # Simulate capture stall of 15 s
        aged = apply_freshness(
            {"heart_rate": m},
            policy=FreshnessPolicy(heart_rate=10.0),
            now_utc_s=utc + 15.0,
        )
        assert aged["heart_rate"].status == MeasurementStatus.STALE


class TestIrregularTimestamps:
    def test_measured_gap_does_not_break_adaptation(self):
        """Variable-FPS: acquisitions at irregular intervals still adapt."""
        stamps = [0.0, 0.033, 0.1, 0.28, 0.31]  # dropped frames at 30 fps
        readings = [make_reading(frame_no=i) for i in range(len(stamps))]
        out = []
        for t, r in zip(stamps, readings):
            mono = int(t * 1e9)
            out.append(measurement_from_reading(r, "heart_rate", mono, t))
        gaps = [out[i + 1].acquired_at_utc_s - out[i].acquired_at_utc_s for i in range(4)]
        assert any(g > 0.05 for g in gaps), "fixture must contain irregular gaps"
        assert all(m.value == 66.0 for m in out)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
