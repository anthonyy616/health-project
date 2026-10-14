"""
Wearable BLE GATT protocol (Phase I)
====================================

Versioned, explicitly-encoded packet schema shared between the ESP32
firmware (firmware/esp32/) and the Python receiver in this package.

Packet layout (little-endian, fixed 23 bytes for v1):

    offset  size  field
    --------------------------
    0       1     protocol_version (= 1)
    1       4     boot_id (random uint32 assigned at MCU boot)
    5       4     sequence (uint32, monotonically increasing, wraps at 2^32)
    9       4     device_ticks_ms (uint32, ms since boot)
    13      2     flags (uint16 bitmask: fields below are VALID when set)
    15      1     hr_bpm (uint8, 0 = unset; valid when FLAG_HR)
    16      1     spo2_pct (uint8, 0 = unset; valid when FLAG_SPO2)
    17      1     ppg_sqi (uint8, device side 0-255 -> 0-1 on host)
    18      1     sensor_status (uint8, SensorStatus enum)
    19      3     motion_score .. battery_pct+flags_low (packed below)
        19      1     motion_score (uint8, 0-255 -> 0-1)
        20      1     battery_pct (uint8, 0-100; invalid when !FLAG_BATTERY)
        21      1     flags_low reserved (0)
    21      2     crc16-xmodem (uint16) over bytes[0:21]

Intentionally conservative packet size (<= MTU-3 typical bridging) with no
fragmentation logic required; a separate raw-PPG characteristic is
documented in firmware/esp32/README.md for debugging.

Software-integrity boundary: packets decoded on the host are data; MAX30102
SpO2 is NOT clinically validated and sensor_status must be surfaced.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Tuple

PROTOCOL_VERSION = 1
# Body format (little-endian, no padding): version, boot_id, sequence,
# device_ticks_ms, flags, then six uint8 fields; bytes 21..22 carry the CRC.
PACKET_FORMAT = "<BIIIHBBBBBB"
PACKET_SIZE = 23
CRC_SPAN = 21  # crc16-xmodem computed over bytes[0:21] (the body)

FLAG_HR = 0x0001
FLAG_SPO2 = 0x0002
FLAG_BATTERY = 0x0004
FLAG_PPG_VALID = 0x0008
FLAG_MOTION_VALID = 0x0010


class SensorStatus(IntEnum):
    OK = 0
    PPG_SENSOR_ERROR = 1
    IMU_SENSOR_ERROR = 2
    I2C_RECOVERY = 3
    LOW_POWER = 4
    CALIBRATING = 5


@dataclass
class WearablePacket:
    """Decoded (or pre-encode) wearable packet payload."""

    boot_id: int
    sequence: int
    device_ticks_ms: int
    hr_bpm: Optional[int] = None
    spo2_pct: Optional[int] = None
    ppg_sqi: Optional[float] = None          # 0..1
    motion_score: Optional[float] = None     # 0..1
    sensor_status: SensorStatus = SensorStatus.OK
    battery_pct: Optional[int] = None

    def __post_init__(self):
        # Range clamps raise: the encoder refuses to silently corrupt data.
        # 8-bit fields are pre-quantized so pack->parse->pack round-trips
        # are bit-identical (wire is the source of truth).
        for name in ("ppg_sqi", "motion_score"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, round(round(value * 255.0) / 255.0, 8))
        for name in ("boot_id", "sequence", "device_ticks_ms"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0 or value > 0xFFFFFFFF:
                raise ValueError(f"{name} must be uint32, got {value!r}")
        for name, lo, hi in (
            ("ppg_sqi", 0.0, 1.0),
            ("motion_score", 0.0, 1.0),
        ):
            value = getattr(self, name)
            if value is not None and not (lo - 1e-9 <= value <= hi + 1e-9):
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        for name in ("hr_bpm", "spo2_pct", "battery_pct"):
            value = getattr(self, name)
            if value is not None and not (0 <= value <= 255):
                raise ValueError(f"{name} must be uint8, got {value}")

    def pack(self) -> bytes:
        flags = 0
        flags |= FLAG_HR if self.hr_bpm is not None else 0
        flags |= FLAG_SPO2 if self.spo2_pct is not None else 0
        flags |= FLAG_BATTERY if self.battery_pct is not None else 0
        flags |= FLAG_PPG_VALID if self.ppg_sqi is not None else 0
        flags |= FLAG_MOTION_VALID if self.motion_score is not None else 0

        body = struct.pack(
            PACKET_FORMAT,
            PROTOCOL_VERSION,
            self.boot_id & 0xFFFFFFFF,
            self.sequence & 0xFFFFFFFF,
            self.device_ticks_ms & 0xFFFFFFFF,
            flags & 0xFFFF,
            self.hr_bpm if self.hr_bpm is not None else 0,
            self.spo2_pct if self.spo2_pct is not None else 0,
            int(round((self.ppg_sqi if self.ppg_sqi is not None else 0.0) * 255.0)) & 0xFF,
            int(self.sensor_status) & 0xFF,
            int(round((self.motion_score if self.motion_score is not None else 0.0) * 255.0)) & 0xFF,
            self.battery_pct if self.battery_pct is not None else 0,
        )
        assert len(body) == CRC_SPAN, (len(body), CRC_SPAN)
        crc = crc16_xmodem(body)
        return body + struct.pack("<H", crc)


def crc16_xmodem(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if (crc & 0x8000) else (crc << 1)
            crc &= 0xFFFF
    return crc


def wrap_delta(a: int, b: int) -> int:
    """Unsigned sequence wrap-aware delta (b - a) mod 2^32."""
    return (b - a) & 0xFFFFFFFF


class PacketError(ValueError):
    """Raised for malformed/rejected packets."""


def parse_packet(raw: bytes) -> WearablePacket:
    """Decode one packet; raises PacketError on bad size/CRC/version."""
    if len(raw) != PACKET_SIZE:
        raise PacketError(f"packet size {len(raw)} != {PACKET_SIZE}")
    payload, crc_bytes = raw[:CRC_SPAN], raw[CRC_SPAN:]
    (crc,) = struct.unpack("<H", crc_bytes)
    if crc16_xmodem(payload) != crc:
        raise PacketError("CRC mismatch")
    version = payload[0]
    if version != PROTOCOL_VERSION:
        raise PacketError(f"unsupported protocol version {version}")
    unpacked = struct.unpack(PACKET_FORMAT, payload)
    (
        _version, boot_id, sequence, ticks, flags,
        hr, spo2, ppg_sq_u8, status_u8, motion_u8, battery,
    ) = unpacked  # 11 fields from the 21-byte body
    return WearablePacket(
        boot_id=boot_id,
        sequence=sequence,
        device_ticks_ms=ticks,
        hr_bpm=hr if (flags & FLAG_HR) else None,
        spo2_pct=spo2 if (flags & FLAG_SPO2) else None,
        ppg_sqi=ppg_sq_u8 / 255.0 if (flags & FLAG_PPG_VALID) else None,
        motion_score=motion_u8 / 255.0 if (flags & FLAG_MOTION_VALID) else None,
        sensor_status=SensorStatus(status_u8) if status_u8 <= max(SensorStatus) else SensorStatus.OK,
        battery_pct=battery if (flags & FLAG_BATTERY) else None,
    )


SERVICE_UUID = "0000A5A5-0000-1000-8000-00805F9B34FB"
CHARACTERISTIC_UUID = "0000A5A6-0000-1000-8000-00805F9B34FB"
