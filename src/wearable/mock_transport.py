"""
Mock BLE transport with simulated wearable sensors (Phase I test support).

PRODUCES SIMULATED DATA ONLY. The simulated stream exists to test host-side
acquisition software (parsing, ordering, timeouts, fusion wiring) before
hardware exists; nothing here validates MAX30102/MPU6050 hardware behaviour
and simulated packets are labelled SIMULATED throughout.

The generator is deterministic: a seeded electrocardiogram-like PPG at a
configurable heart rate, with a sinusoidal respiration-locked amplitude
modulation and configurable motion (IMU) disturbance illusions.
"""

from __future__ import annotations

import math
import random
import time
from typing import List, Optional

from src.wearable.protocol import SensorStatus, WearablePacket


class MockWearableTransport:
    """Deterministic simulated wearable that emits valid packets over time.

    Usage (host-side tests / software integration):
        transport = MockWearableTransport(heart_rate_bpm=72, packet_period_s=0.05)
        packets = transport.tick(until_s=5.0)
    """

    def __init__(
        self,
        heart_rate_bpm: float = 70.0,
        packet_period_s: float = 0.05,
        sensor_status: SensorStatus = SensorStatus.OK,
        seed: int = 7,
        boot_id: Optional[int] = None,
    ):
        if heart_rate_bpm <= 0:
            raise ValueError("heart_rate_bpm must be positive")
        self.heart_rate_bpm = heart_rate_bpm
        self.packet_period_s = packet_period_s
        self.sensor_status = sensor_status
        self.boot_id = boot_id if boot_id is not None else random.Random(seed).randrange(1, 2**32)
        self._sequence = 0
        self._elapsed_s = 0.0
        self._rng = random.Random(seed)

    @property
    def elapsed_s(self) -> float:
        return self._elapsed_s

    def _synthetic_values(self, t: float):
        """Deterministic HR/quality/motion at time t (SIN PPG SIMULATION)."""
        pulse_hz = self.heart_rate_bpm / 60.0
        phase = 2 * math.pi * pulse_hz * t
        # hr read with small deterministic beat-to-beat variation
        hr = self.heart_rate_bpm + 0.8 * math.sin(2 * math.pi * 0.25 * t)
        hr = int(round(hr))
        # PPG quality degrades when simulated motion is present
        quality = 0.9 + 0.05 * math.sin(2 * math.pi * 0.11 * t)
        motion = 0.05 + 0.03 * math.sin(2 * math.pi * 0.5 * t)
        return hr, quality, motion

    def next_packet(self) -> WearablePacket:
        t = self.elapsed_s
        hr, quality, motion = self._synthetic_values(t)
        packet = WearablePacket(
            boot_id=self.boot_id,
            sequence=self._sequence,
            device_ticks_ms=int(round(t * 1000.0)),
            hr_bpm=hr,
            ppg_sqi=quality,
            motion_score=motion,
            sensor_status=self.sensor_status,
            battery_pct=87,
        )
        self._sequence += 1
        self._elapsed_s += self.packet_period_s
        return packet

    def tick(self, duration_s: float) -> List[WearablePacket]:
        """Emit packets covering duration_s of simulated time."""
        count = max(0, int(round(duration_s / self.packet_period_s)))
        return [self.next_packet() for _ in range(count)]

    def simulate_reboot(self):
        """Device reboots: new boot_id, sequence and ticks reset to zero."""
        self.boot_id = (self.boot_id + 1) % (2**32)
        self._sequence = 0
        self._elapsed_s = 0.0
