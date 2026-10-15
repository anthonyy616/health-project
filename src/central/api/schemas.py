"""
Pydantic request/response schemas (Phase K).

Validation rules mirror the Measurement contract: finite numeric values,
bounded ranges, explicit statuses, no phantom readings (value=None allowed
only for non-VALID statuses).
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


def _ensure_finite(value: Optional[float], field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite, got {value}")


class SessionCreate(BaseModel):
    display_name: str = Field(default="Unnamed session", max_length=120)
    consent_recorded: bool = False


class SessionOut(BaseModel):
    id: str
    display_name: str
    started_at: float
    ended_at: Optional[float] = None
    consent_recorded: bool


_ALLOWED_STATUSES = {
    "valid", "low_signal", "insufficient_data", "no_face",
    "motion_artifact", "stale", "disconnected", "error",
}
_STATUS_WITHOUT_VALUE = _ALLOWED_STATUSES - {"valid", "low_signal"}


class MeasurementIn(BaseModel):
    metric: str = Field(min_length=3, max_length=40, pattern=r"^[a-z0-9_]+$")
    status: str
    value: Optional[float] = None
    unit: str = Field(min_length=1, max_length=24)
    source: str = Field(min_length=3, max_length=40, pattern=r"^[a-z0-9_]+$")
    acquired_at_utc: Optional[float] = None      # defaults to server now
    acquired_mono_ns: Optional[int] = None
    signal_quality: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    failure_reason: Optional[str] = Field(default=None, max_length=500)
    latency_ms: Optional[float] = Field(default=None, ge=0.0)
    frame_number: Optional[int] = Field(default=None, ge=0)
    provenance: Dict[str, str] = Field(default_factory=dict)

    @field_validator("status")
    @classmethod
    def _validate_status(cls, v: str) -> str:
        if v not in _ALLOWED_STATUSES:
            raise ValueError(f"status must be one of {sorted(_ALLOWED_STATUSES)}")
        return v

    @field_validator("value")
    @classmethod
    def _validate_value(cls, v):
        _ensure_finite(v, "value")
        return v

    @model_validator(mode="after")
    def _validate_semantics(self) -> "MeasurementIn":
        if self.status in _STATUS_WITHOUT_VALUE and self.value is not None:
            raise ValueError(
                f"status '{self.status}' must not carry a value; rejected to keep "
                "absence distinct from readings"
            )
        if self.status == "valid" and self.value is None:
            raise ValueError("status 'valid' requires a finite value")
        # Physiological plausibility gates (not clinical gates): reject
        # obviously corrupted payloads early.
        if self.value is not None:
            if self.metric == "heart_rate" and not (10.0 <= self.value <= 250.0):
                raise ValueError("heart_rate value outside 10-250 BPM")
            if self.metric == "respiratory_rate" and not (2.0 <= self.value <= 60.0):
                raise ValueError("respiratory_rate value outside 2-60 breaths/min")
            if self.metric == "age" and not (0.0 <= self.value <= 120.0):
                raise ValueError("age value outside 0-120 years")
        return self

    @model_validator(mode="after")
    def _validate_acquisition_time(self) -> "MeasurementIn":
        now = time.time()
        acquired = self.acquired_at_utc if self.acquired_at_utc is not None else now
        if acquired > now + 60.0:
            raise ValueError("acquired_at_utc cannot be more than 60 s in the future")
        if acquired < now - 24 * 3600.0:
            raise ValueError("acquired_at_utc cannot be more than 24 h in the past")
        return self


class MeasurementOut(BaseModel):
    metric: str
    status: str
    value: Optional[float]
    unit: str
    source: str
    acquired_at_utc: float
    signal_quality: Optional[float]
    confidence: Optional[float]
    failure_reason: Optional[str]
    latency_ms: Optional[float]
    frame_number: Optional[int]
    provenance: Dict[str, str]


class AlertIn(BaseModel):
    severity: str  # info | warning | critical
    kind: str      # threshold | quality | disconnect | stale
    message: str = Field(min_length=3, max_length=500)
    metric: Optional[str] = None
    value: Optional[float] = None

    @field_validator("severity")
    @classmethod
    def _severity(cls, v):
        if v not in {"info", "warning", "critical"}:
            raise ValueError("severity must be info|warning|critical")
        return v

    @field_validator("kind")
    @classmethod
    def _kind(cls, v):
        if v not in {"threshold", "quality", "disconnect", "stale"}:
            raise ValueError("kind must be threshold|quality|disconnect|stale")
        return v

    @field_validator("value")
    @classmethod
    def _value(cls, v):
        _ensure_finite(v, "value")
        return v
