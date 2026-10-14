"""
Wearable packet → Measurement adapter (Phase I to Phase J bridge).

Converts decoded WearablePackets into the shared Measurement envelope so
the fusion engine and backend treat contactless and wearable values with
the same contract.

Integrity constraints carried over from the protocol:
- Only fields whose valid-flag was set produce values.
- SpO2 is never produced as a clinical number: it maps to a provenance-
  marked, non-validated measurement (research-only).
"""

from __future__ import annotations

import time
from typing import Optional

from src.contactless.measurements import Measurement, MeasurementStatus
from src.wearable.protocol import SensorStatus, WearablePacket

STATUS_BY_SENSOR = {
    SensorStatus.OK: MeasurementStatus.VALID,
    SensorStatus.CALIBRATING: MeasurementStatus.LOW_SIGNAL,
    SensorStatus.I2C_RECOVERY: MeasurementStatus.LOW_SIGNAL,
    SensorStatus.LOW_POWER: MeasurementStatus.LOW_SIGNAL,
    SensorStatus.PPG_SENSOR_ERROR: MeasurementStatus.ERROR,
    SensorStatus.IMU_SENSOR_ERROR: MeasurementStatus.ERROR,
}


def packet_to_measurements(
    packet: WearablePacket,
    received_at_s: float,
    session_id: Optional[str] = None,
    source: str = "wearable_ble",
) -> dict:
    """Map one packet to {metric: Measurement} entries.

    Device clock → host clock conversion is intentionally crude here
    (device_ticks_ms is monotonic-per-boot only). Real offset/drift
    estimation between the wearable clock and the host clock is a Phase J
    fusion concern and stays explicit there; packets carry both stamps.
    """
    measurements: dict = {}
    base_status = STATUS_BY_SENSOR.get(packet.sensor_status, MeasurementStatus.ERROR)

    if packet.hr_bpm is not None:
        status = base_status
        measurements["heart_rate"] = Measurement(
            metric="heart_rate",
            status=status,
            value=float(packet.hr_bpm),
            unit="bpm",
            source=source,
            acquired_mono_ns=None,
            acquired_at_utc_s=received_at_s - packet.device_ticks_ms / 1000.0,
            computed_at_utc_s=time.time(),
            signal_quality=packet.ppg_sqi,
            confidence=packet.ppg_sqi,
            failure_reason=None if status == MeasurementStatus.VALID else f"sensor_status={packet.sensor_status.name}",
            session_id=session_id,
            provenance={"boot_id": packet.boot_id, "sequence": packet.sequence, "device_ticks_ms": packet.device_ticks_ms},
        )

    if packet.motion_score is not None:
        measurements["motion"] = Measurement(
            metric="motion",
            status=MeasurementStatus.VALID,
            value=float(packet.motion_score),
            unit="fraction",
            source=source,
            acquired_mono_ns=None,
            acquired_at_utc_s=received_at_s - packet.device_ticks_ms / 1000.0,
            computed_at_utc_s=time.time(),
            signal_quality=None,
            confidence=None,
            session_id=session_id,
            provenance={"boot_id": packet.boot_id, "sequence": packet.sequence},
        )

    if packet.spo2_pct is not None:
        # NON-CLINICAL: explicitly research-only measurement.
        measurements["spo2_unvalidated"] = Measurement(
            metric="spo2_unvalidated",
            status=MeasurementStatus.LOW_SIGNAL,
            value=float(packet.spo2_pct),
            unit="percent",
            source=source,
            acquired_mono_ns=None,
            acquired_at_utc_s=received_at_s - packet.device_ticks_ms / 1000.0,
            computed_at_utc_s=time.time(),
            confidence=0.0,
            failure_reason="SpO2 has no independent clinical calibration; research use only",
            session_id=session_id,
            provenance={"boot_id": packet.boot_id, "non_clinical": True},
        )

    return measurements


def battery_from_packet(packet: WearablePacket) -> Optional[float]:
    return float(packet.battery_pct) if packet.battery_pct is not None else None
