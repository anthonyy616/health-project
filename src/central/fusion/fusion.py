"""
Multimodal sensor fusion (Phase J)
==================================

Blends contactless (camera) and wearable measurements under an explicit
trust policy. Follows the audit pack's guardrails:

- Camera-only / wearable-only / combined modes; a missing source falls
  back rather than failing (and the fallback is visible in the result).
- Invalid + stale sources are rejected BEFORE blending - never force a
  result from a dead sensor.
- Simple equal-weight blending, then quality weighting. NO Kalman filter:
  D04 in the decision register fixes "confidence-weighted blending
  before Kalman", and equally, no fusion-accuracy claim is made without
  an independent reference evaluation (dataset needed).
- Disagreement detection: if sources disagree beyond the configured
  tolerance, the engine flags it and (configurable) trusts the
  higher-quality source instead of averaging away a real signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

from src.contactless.measurements import (
    FreshnessPolicy,
    Measurement,
    MeasurementStatus,
    apply_freshness,
)


class FusionMode(str, Enum):
    CAMERA_ONLY = "camera_only"
    WEARABLE_ONLY = "wearable_only"
    COMBINED = "combined"


@dataclass
class FusionResult:
    """Outcome of one fusion decision for one metric."""

    metric: str
    value: Optional[float]
    status: MeasurementStatus
    source_used: str  # "camera" | "wearable" | "blended" | "none"
    weights: Dict[str, float]
    camera_value: Optional[float]
    wearable_value: Optional[float]
    disagreement_flagged: bool = False
    failure_reason: Optional[str] = None

    def to_dict(self):
        return {
            "metric": self.metric,
            "value": self.value,
            "status": self.status.value,
            "source_used": self.source_used,
            "weights": self.weights,
            "camera_value": self.camera_value,
            "wearable_value": self.wearable_value,
            "disagreement_flagged": self.disagreement_flagged,
            "failure_reason": self.failure_reason,
        }


@dataclass
class FusionConfig:
    """Fusion policy parameters (documented defaults, no magic values)."""

    mode: FusionMode = FusionMode.COMBINED
    # Sources disagree when |a - b| exceeds tolerance_bpm; then the higher
    # SQI source wins instead of averaging a conflict away.
    disagreement_tolerance: float = 15.0
    # A source with SQI below this is trusted with zero weight.
    min_source_quality: float = 0.2
    freshness: FreshnessPolicy = field(default_factory=FreshnessPolicy)


class FusionEngine:
    """Stateless per-decision fusion (freshness handled per call)."""

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self._last_biases = []

    def fuse(self, metric: str, camera: Optional[Measurement], wearable: Optional[Measurement], now_utc_s: Optional[float] = None) -> FusionResult:
        """Fuse one metric from at-most-two candidate measurements.

        Args:
            metric: e.g. "heart_rate".
            camera/wearable: Measurement envelopes (may be None when the
                source is disconnected or the metric absent).
            now_utc_s: injectable clock for deterministic tests.

        Returns:
            FusionResult - `value` is None whenever no trustworthy source
            exists; statuses surface exactly why.
        """
        fresh = apply_freshness(
            {"camera": camera, "wearable": wearable},
            policy=self.config.freshness,
            now_utc_s=now_utc_s,
        ) if (camera is not None or wearable is not None) else {}

        # Raw (pre-freshness) values stay visible in results for diagnostics
        # even when a source is rejected as stale.
        raw_camera_value = camera.value if camera is not None else None
        raw_wearable_value = wearable.value if wearable is not None else None

        camera_f = fresh.get("camera", camera)
        wearable_f = fresh.get("wearable", wearable)

        camera_ok = self._usable(camera_f)
        wearable_ok = self._usable(wearable_f)

        mode = self.config.mode
        if mode == FusionMode.CAMERA_ONLY:
            return self._single(metric, camera_f, camera_ok, "camera")
        if mode == FusionMode.WEARABLE_ONLY:
            return self._single(metric, wearable_f, wearable_ok, "wearable")

        if camera_ok and not wearable_ok:
            return self._single(metric, camera_f, True, "camera")
        if wearable_ok and not camera_ok:
            return self._single(metric, wearable_f, True, "wearable")
        if not camera_ok and not wearable_ok:
            return FusionResult(
                metric=metric,
                value=None,
                status=MeasurementStatus.DISCONNECTED
                if (camera_f is None and wearable_f is None)
                else MeasurementStatus.ERROR,
                source_used="none",
                weights={"camera": 0.0, "wearable": 0.0},
                camera_value=raw_camera_value,
                wearable_value=raw_wearable_value,
                failure_reason=(
                    "no sources present (both disconnected)" if (camera_f is None and wearable_f is None)
                    else "no trustworthy source after quality/freshness gating"
                ),
            )

        # Both usable: disagreement check.
        cam_val = float(camera_f.value)
        wear_val = float(wearable_f.value)
        disagreement = abs(cam_val - wear_val) > self.config.disagreement_tolerance

        if disagreement:
            # Trust the higher-quality single source rather than blending
            # across a conflict that may indicate motion bias on one source.
            if (camera_f.signal_quality or 0.0) >= (wearable_f.signal_quality or 0.0):
                return self._single(metric, camera_f, True, "camera", disagreement_flag=True)
            return self._single(metric, wearable_f, True, "wearable", disagreement_flag=True)

        # Quality-weighted blend (default equal when SQI missing on both).
        w_cam = camera_f.signal_quality if camera_f.signal_quality is not None else 0.5
        w_wear = wearable_f.signal_quality if wearable_f.signal_quality is not None else 0.5
        w_cam = self._usable_weight(w_cam)
        w_wear = self._usable_weight(w_wear)
        total = w_cam + w_wear
        if total <= 0.0:
            return self._single(metric, camera_f, True, "camera")

        value = (w_cam * cam_val + w_wear * wear_val) / total
        return FusionResult(
            metric=metric,
            value=round(value, 4),
            status=MeasurementStatus.VALID,
            source_used="blended",
            weights={"camera": round(w_cam / total, 4), "wearable": round(w_wear / total, 4)},
            camera_value=cam_val,
            wearable_value=wear_val,
            disagreement_flagged=False,
        )

    # -- helpers -----------------------------------------------------------
    def _usable(self, m: Optional[Measurement]) -> bool:
        if m is None:
            return False
        if m.status not in (MeasurementStatus.VALID, MeasurementStatus.LOW_SIGNAL):
            return False
        if m.value is None:
            return False
        # Absent SQI is treated as neutral-trust, not as known-bad:
        # blending then behaves as equal weight instead of dropping the
        # source. Only a MEASURED low quality (< min_source_quality)
        # removes trust.
        if m.signal_quality is None:
            return True
        return m.signal_quality >= self.config.min_source_quality

    def _usable_weight(self, q: float) -> float:
        return max(0.0, min(1.0, q))

    def _raw_value(self, m: Optional[Measurement]) -> Optional[float]:
        return None if m is None else m.value

    def _single(self, metric: str, m: Optional[Measurement], ok: bool, source: str, disagreement_flag: bool = False) -> FusionResult:
        if not ok or m is None:
            status = m.status if m is not None else MeasurementStatus.DISCONNECTED
            return FusionResult(
                metric=metric,
                value=None,
                status=status,
                source_used=source,
                weights={"camera": 0.0, "wearable": 0.0},
                camera_value=self._raw_value(m) if source == "camera" else None,
                wearable_value=self._raw_value(m) if source == "wearable" else None,
                disagreement_flagged=disagreement_flag,
                failure_reason=f"{source} source unusable",
            )
        return FusionResult(
            metric=metric,
            value=m.value,
            status=m.status,
            source_used=source,
            weights={"camera": 1.0 if source == "camera" else 0.0,
                     "wearable": 1.0 if source == "wearable" else 0.0},
            camera_value=self._raw_value(m) if source == "camera" else None,
            wearable_value=self._raw_value(m) if source == "wearable" else None,
            disagreement_flagged=disagreement_flag,
            failure_reason=None,
        )
