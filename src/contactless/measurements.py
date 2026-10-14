"""
Measurement envelope and timing semantics (Phase G)
===================================================

Introduces the normalized measurement contract from the architecture
review: every physiological quantity is a `Measurement` carrying value,
units, source, timestamps (monotonic + wall clock), validity status,
quality/confidence, window and latency metadata.

Design boundaries honored:
- `VitalsReading` (pipeline) remains the backward-compatible flat shape;
  `Measurement.from_vitals_reading` adapts it, so existing CLI/tests/
  `to_dict()` consumers are unaffected.
- Clock semantics: `acquired_mono_ns` is the CAMPAIGN clock frame
  timestamp (monotonic), `acquired_at_utc` is the wall clock at
  acquisition; comparison across computers requires NTP-synchronized
  wall clocks or a negotiated offset (BLE sync handles that separately).
- Freshness policy is explicit (`FreshnessPolicy`), never implicit:
  stale, low-quality and unavailable are distinct statuses as demanded
  by the audit pack. Nothing here invents readings when data is missing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class MeasurementStatus(str, Enum):
    """Explicit validity lifecycle of a measurement.

    Consumer-visible distinct states (audit requirement): no reading may
    ever mask one state as another (e.g. stale data shown as live).
    """

    VALID = "valid"
    LOW_SIGNAL = "low_signal"           # computed but quality below gate
    INSUFFICIENT_DATA = "insufficient_data"  # buffer/window too short
    NO_FACE = "no_face"                  # contactless ROIs unavailable
    MOTION_ARTIFACT = "motion_artifact"
    STALE = "stale"                      # computed once but now expired
    DISCONNECTED = "disconnected"        # source stream down (camera/BLE)
    ERROR = "error"                      # processing failed this frame


UNIT_BY_METRIC = {
    "heart_rate": "bpm",
    "respiratory_rate": "breaths_per_min",
    "pupil_diameter": "mm",
    "age": "years",
}

_SQI_BY_METRIC_ATTR = {
    "heart_rate": "hr_sq",
    "respiratory_rate": "resp_sq",
    "pupil_diameter": "pupil_confidence",
    "age": "age_confidence",
}


@dataclass
class Measurement:
    """One normalized physiological measurement.

    `value` is None whenever status is not VALID/LOW_SIGNAL — the absence
    of data stays visible instead of proxy zero-values.
    """

    metric: str
    status: MeasurementStatus
    value: Optional[float]
    unit: str
    source: str  # "contactless_camera" | "wearable_ble" | "fused" | ...
    acquired_mono_ns: int           # monotonic frame clock (perf_counter_ns)
    acquired_at_utc_s: float        # wall clock at acquisition (epoch seconds)
    computed_at_utc_s: float        # wall clock when this Measurement was built
    signal_quality: Optional[float] = None
    confidence: Optional[float] = None
    failure_reason: Optional[str] = None
    window: Optional[Dict[str, float]] = None   # {start_s, end_s} relative to session
    latency_ms: Optional[float] = None
    frame_number: Optional[int] = None
    session_id: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def measurement_age_s(self, now_utc_s: Optional[float] = None) -> float:
        """Age since acquisition, wall-clock based."""
        now = now_utc_s if now_utc_s is not None else time.time()
        return float(now - self.acquired_at_utc_s)

    def is_stale_under(self, policy: "FreshnessPolicy") -> bool:
        return self.measurement_age_s() > policy.for_metric(self.metric)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["status"] = self.status.value
        return out


@dataclass
class FreshnessPolicy:
    """Configured expiry rules per metric (seconds)."""

    heart_rate: float = 10.0
    respiratory_rate: float = 20.0
    pupil_diameter: float = 3.0
    age: float = 60.0

    def for_metric(self, metric: str) -> float:
        return getattr(self, metric, 5.0)


def now_pair() -> Tuple[int, float]:
    """(monotonic_ns, wall_utc_s) acquired together for one timestamp."""
    return time.perf_counter_ns(), time.time()


def _value_for_metric(reading, metric: str) -> Optional[float]:
    return getattr(reading, _SQI_BY_METRIC_ATTR.get(metric, metric), None)


_VALUE_ATTR = {
    "heart_rate": "heart_rate_bpm",
    "respiratory_rate": "respiratory_rate_bpm",
    "pupil_diameter": "pupil_diameter_mm",
    "age": "age",
}

# Module-error keys in VitalsReading.module_errors (see pipeline.py).
ERROR_KEY_BY_METRIC = {
    "heart_rate": "heart_rate",
    "respiratory_rate": "respiration",
    "pupil_diameter": "pupil",
    "age": "age",
}


def measurement_from_reading(
    reading,
    metric: str,
    acquired_mono_ns: int,
    acquired_at_utc_s: float,
    source: str = "contactless_camera",
    session_id: Optional[str] = None,
) -> Measurement:
    """Build a Measurement for one metric from a VitalsReading.

    The reading itself stays untouched (backward compatible); this is the
    adapter the backend/dashboard will consume.
    """
    value = getattr(reading, _VALUE_ATTR[metric])
    confidence_attr = _SQI_BY_METRIC_ATTR[metric]
    confidence = getattr(reading, confidence_attr, None)
    module_errors = getattr(reading, "module_errors", None) or {}
    error = module_errors.get(ERROR_KEY_BY_METRIC[metric])

    if error is not None:
        status = MeasurementStatus.ERROR
        value = None
        failure = error
    elif not getattr(reading, "face_detected", True):
        status = MeasurementStatus.NO_FACE
        value = None
        failure = "no face detected"
    elif value is None:
        status = MeasurementStatus.INSUFFICIENT_DATA
        failure = "no measurement for this frame"
    else:
        status = MeasurementStatus.VALID
        failure = None

    computed_at = time.time()
    return Measurement(
        metric=metric,
        status=status,
        value=value,
        unit=UNIT_BY_METRIC[metric],
        source=source,
        acquired_mono_ns=acquired_mono_ns,
        acquired_at_utc_s=acquired_at_utc_s,
        computed_at_utc_s=computed_at,
        confidence=confidence,
        failure_reason=failure,
        latency_ms=getattr(reading, "total_latency_ms", None),
        frame_number=getattr(reading, "frame_number", None),
        session_id=session_id,
        provenance={"metric": metric, "adapter": "measurement_from_reading"},
    )


def apply_freshness(
    measurements: Dict[str, Measurement],
    policy: Optional[FreshnessPolicy] = None,
    now_utc_s: Optional[float] = None,
) -> Dict[str, Measurement]:
    """Downgrade VALID/LOW_SIGNAL measurements past their TTL to STALE.

    Staleness is explicit: consumers see the STALE status (or the absence
    of a value); nothing silently keeps displaying expired numbers.
    Returns a NEW dict; inputs are not mutated.
    """
    policy = policy or FreshnessPolicy()
    now = now_utc_s if now_utc_s is not None else time.time()
    out: Dict[str, Measurement] = {}
    for metric, m in measurements.items():
        ttl = policy.for_metric(metric)
        if m.status in (MeasurementStatus.VALID, MeasurementStatus.LOW_SIGNAL) \
                and (now - m.acquired_at_utc_s) > ttl:
            stale = Measurement(
                metric=m.metric,
                status=MeasurementStatus.STALE,
                value=None,
                unit=m.unit,
                source=m.source,
                acquired_mono_ns=m.acquired_mono_ns,
                acquired_at_utc_s=m.acquired_at_utc_s,
                computed_at_utc_s=m.computed_at_utc_s,
                signal_quality=m.signal_quality,
                confidence=m.confidence,
                failure_reason=f"measurement age {now - m.acquired_at_utc_s:.1f}s exceeds TTL {ttl:.1f}s",
                window=m.window,
                latency_ms=m.latency_ms,
                frame_number=m.frame_number,
                session_id=m.session_id,
                provenance=m.provenance,
            )
            out[metric] = stale
        else:
            out[metric] = m
    return out
