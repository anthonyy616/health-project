"""
Tests for the Phase J multimodal fusion engine on synthetic measurements.

These verify FUSION SOFTWARE CORRECTNESS (source gating, weighting,
disagreement policy, stale rejection) on synthetic inputs. Any claim that
fusion actually improves accuracy vs single sources requires an
independent reference evaluation and stays an outstanding requirement.
"""

import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.contactless.measurements import (
    FreshnessPolicy,
    Measurement,
    MeasurementStatus,
)
from src.central.fusion.fusion import FusionConfig, FusionEngine, FusionMode


def make_m(metric="heart_rate", value=70.0, quality=0.8, source="contactless_camera",
           status=MeasurementStatus.VALID, age_s=0.0, now=None):
    now = now if now is not None else time.time()
    return Measurement(
        metric=metric,
        status=status,
        value=value,
        unit="bpm" if metric == "heart_rate" else "breaths_per_min",
        source=source,
        acquired_mono_ns=0,
        acquired_at_utc_s=now - age_s,
        computed_at_utc_s=now,
        signal_quality=quality,
        confidence=quality,
    )


class TestSourceGating:
    def test_both_none_is_disconnected_not_zero(self):
        engine = FusionEngine()
        result = engine.fuse("heart_rate", None, None)
        assert result.value is None
        assert result.status == MeasurementStatus.DISCONNECTED
        assert result.source_used == "none"

    def test_camera_only_mode(self):
        engine = FusionEngine(config=FusionConfig(mode=FusionMode.CAMERA_ONLY))
        result = engine.fuse(
            "heart_rate",
            make_m(),
            make_m(value=80.0, source="wearable_ble"),
        )
        assert result.value == 70.0
        assert result.source_used == "camera"

    def test_missing_source_falls_back(self):
        engine = FusionEngine()
        result = engine.fuse("heart_rate", make_m(), None)
        assert result.value == 70.0
        assert result.source_used == "camera"
        # and reversed
        result = engine.fuse("heart_rate", None, make_m(source="wearable_ble", value=75.0))
        assert result.value == 75.0
        assert result.source_used == "wearable"

    def test_stale_source_rejected(self):
        engine = FusionEngine(config=FusionConfig(
            freshness=FreshnessPolicy(heart_rate=10.0)
        ))
        now = time.time()
        stale_cam = make_m(age_s=30.0, now=now)
        result = engine.fuse("heart_rate", stale_cam, None, now_utc_s=now + 1.0)
        assert result.value is None
        assert result.camera_value == 70.0, "rejected values stay visible for diagnostics"
        assert result.source_used == "none"
        assert "quality/freshness" in result.failure_reason

    def test_low_quality_source_rejected(self):
        engine = FusionEngine()
        result = engine.fuse("heart_rate", make_m(quality=0.05), make_m(source="wearable_ble", value=72.0, quality=0.9))
        assert result.source_used == "wearable"
        assert result.value == 72.0

    def test_error_status_rejected(self):
        engine = FusionEngine()
        result = engine.fuse(
            "heart_rate",
            make_m(status=MeasurementStatus.ERROR),
            make_m(source="wearable_ble", value=72.0),
        )
        assert result.value == 72.0


class TestBlending:
    def test_equal_quality_equal_weights(self):
        engine = FusionEngine()
        result = engine.fuse(
            "heart_rate",
            make_m(value=70.0, quality=0.8),
            make_m(value=74.0, source="wearable_ble", quality=0.8),
        )
        assert result.source_used == "blended"
        assert result.value == 72.0
        assert result.weights["camera"] == pytest.approx(0.5)
        assert result.weights["wearable"] == pytest.approx(0.5)

    def test_quality_weighted_blend(self):
        # Both above the min-SQI trust floor (0.2), asymmetric quality;
        # wide tolerance so 68 vs 92 stays a blend case.
        engine = FusionEngine(config=FusionConfig(
            disagreement_tolerance=25.0, min_source_quality=0.2
        ))
        result = engine.fuse(
            "heart_rate",
            make_m(value=68.0, quality=0.9),
            make_m(value=92.0, source="wearable_ble", quality=0.3),
        )
        assert result.source_used == "blended"
        assert result.value == pytest.approx(0.75*68.0 + 0.25*92.0)
        assert result.weights["camera"] == pytest.approx(0.75)
        assert result.weights["wearable"] == pytest.approx(0.25)

    def test_below_trust_floor_source_dropped(self):
        """Measured SQI under min_source_quality loses trust entirely."""
        engine = FusionEngine()
        result = engine.fuse(
            "heart_rate",
            make_m(value=70.0, quality=0.9),
            make_m(value=90.0, source="wearable_ble", quality=0.1),
        )
        assert result.source_used == "camera"
        assert result.value == 70.0

    def test_missing_sqi_defaults_equal(self):
        engine = FusionEngine()
        cam = make_m(value=70.0)
        cam.signal_quality = None
        wear = make_m(value=74.0, source="wearable_ble")
        wear.signal_quality = None
        result = engine.fuse("heart_rate", cam, wear)
        assert result.value == 72.0


class TestDisagreement:
    def test_disagreement_prefers_better_source(self):
        engine = FusionEngine(config=FusionConfig(disagreement_tolerance=10.0))
        result = engine.fuse(
            "heart_rate",
            make_m(value=60.0, quality=0.95),
            make_m(value=110.0, source="wearable_ble", quality=0.2),
        )
        assert result.disagreement_flagged is True
        assert result.source_used == "camera"
        assert result.value == 60.0

    def test_disagreement_prefers_better_wearable(self):
        engine = FusionEngine(config=FusionConfig(disagreement_tolerance=10.0))
        result = engine.fuse(
            "heart_rate",
            make_m(value=60.0, quality=0.1),
            make_m(value=110.0, source="wearable_ble", quality=0.95),
        )
        assert result.value == 110.0
        assert result.source_used == "wearable"

    def test_no_disagreement_within_tolerance_blends(self):
        engine = FusionEngine(config=FusionConfig(disagreement_tolerance=10.0))
        result = engine.fuse(
            "heart_rate",
            make_m(value=68.0, quality=0.7),
            make_m(value=72.0, source="wearable_ble", quality=0.7),
        )
        assert result.disagreement_flagged is False
        assert result.source_used == "blended"
        assert result.value == 70.0


class TestWearableTelemetryAdapter:
    def test_packet_to_measurements_shape(self):
        from src.wearable.mock_transport import MockWearableTransport
        from src.wearable.telemetry import packet_to_measurements

        gen = MockWearableTransport(heart_rate_bpm=82.0, seed=9)
        packet = gen.next_packet()
        measurements = packet_to_measurements(packet, received_at_s=time.time())
        assert "heart_rate" in measurements
        hr = measurements["heart_rate"]
        assert hr.value == packet.hr_bpm
        assert hr.signal_quality == packet.ppg_sqi
        assert "spo2_unvalidated" not in measurements  # not emitted without flag
        if "motion" in measurements:
            assert measurements["motion"].value == packet.motion_score

    def test_spo2_always_marked_unvalidated(self):
        from src.wearable.telemetry import packet_to_measurements
        from src.wearable.protocol import WearablePacket, SensorStatus

        p = WearablePacket(boot_id=1, sequence=1, device_ticks_ms=0, spo2_pct=97)
        result = packet_to_measurements(p, received_at_s=time.time())
        spo2 = result["spo2_unvalidated"]
        assert spo2.status != MeasurementStatus.VALID
        assert "non_clinical" in spo2.provenance
        assert spo2.confidence == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
