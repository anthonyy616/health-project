"""
Wearable BLE receiver (Phase I)
===============================

Host-side lifecycle and robustness around the GATT notification stream:
connection management, packet validation, ordering, device-reboot
detection, timeout/staleness handling, and reconnect with bounded backoff.

Transport backends:
- `MockWearableTransport` (src/wearable/mock_transport.py): deterministic
  simulated packets for software integration tests.
- `BleakLiveReceiver` (optional import): real BLE via bleak, only imported
  on actual use so the package does not require bleak at module import.

The receiver deliberately performs no physiological interpretation: it
yields decoded WearablePackets plus account statistics. Fusion (Phase J)
owns quality gating and timestamp alignment.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Optional

from src.wearable.protocol import (
    PACKET_SIZE,
    PacketError,
    WearablePacket,
    parse_packet,
    wrap_delta,
)

logger = logging.getLogger(__name__)

# Sequence accounting: deltas larger than this are reported as gaps
# (while <= this are ordinary reorderings), bounded to avoid long mod-2^32
# wrap false positives.
GAP_THRESHOLD = 64
OUT_OF_ORDER_WINDOW = 256


@dataclass
class ReceiverStats:
    """Operating statistics (lifecycle accounting, not science)."""

    packets_received: int = 0
    packets_accepted: int = 0
    crc_errors: int = 0
    size_errors: int = 0
    out_of_order: int = 0
    gaps_detected: int = 0
    disconnects: int = 0
    reboots_detected: int = 0
    last_packet_received_s: Optional[float] = None

    def to_dict(self):
        return asdict(self)


class WearableReceiver:
    """Packet-oriented receiver, decoupled from the transport backend.

    Reliability rules (ordering, boot-id resets, CRC and size rejection)
    are driven through feed() so mock and real transports behave
    identically - which is what makes these behaviours testable before
    hardware exists.
    """

    def __init__(self, stale_after_s: float = 3.0, history_s: float = 30.0):
        """stale_after_s: connection silence that marks data stale.
        history_s: packet log length for latency/jitter diagnostics."""
        self.stale_after_s = stale_after_s
        self.stats = ReceiverStats()
        self._received_log: deque = deque()
        self._last_sequence: Optional[int] = None
        self._last_boot_id: Optional[int] = None
        self._connected = False
        self._handlers: List[Callable[[WearablePacket, float], None]] = []

    # -- lifecycle ---------------------------------------------------------
    def on_connected(self) -> None:
        self._connected = True
        self._last_sequence = None
        logger.info("wearable connection established")

    def on_disconnected(self) -> None:
        self._connected = False
        self._last_sequence = None
        self.stats.disconnects += 1
        logger.warning("wearable disconnected")

    @property
    def connected(self) -> bool:
        return self._connected

    def has_fresh_data(self, now_s: Optional[float] = None) -> bool:
        """True when packets arrived within stale_after_s."""
        if self.stats.last_packet_received_s is None:
            return False
        now = now_s if now_s is not None else time.time()
        return (now - self.stats.last_packet_received_s) <= self.stale_after_s

    # -- packet feed ---------------------------------------------------------
    def feed(self, raw: bytes, received_at_s: Optional[float] = None) -> Optional[WearablePacket]:
        """Decode one notification payload; returns the packet when accepted."""
        received_at = received_at_s if received_at_s is not None else time.time()
        self.stats.packets_received += 1

        if len(raw) != PACKET_SIZE:
            self.stats.size_errors += 1
            logger.warning("packet size %d != %d; dropping", len(raw), PACKET_SIZE)
            return None
        try:
            packet = parse_packet(raw)
        except PacketError as e:
            self._count_crc_error(e)
            logger.warning("packet rejected: %s", e)
            return None

        was_kind = self._ordering_state(packet)
        del was_kind

        self.stats.packets_accepted += 1
        self.stats.last_packet_received_s = received_at
        self._received_log.append((received_at, packet.sequence))
        while self._received_log and self._received_log[0][0] < received_at - 30.0:
            self._received_log.popleft()

        for handler in self._handlers:
            try:
                handler(packet, received_at)
            except Exception:  # noqa: BLE001 - handler bugs must not drop packets
                logger.exception("packet handler failed")
        return packet

    def subscribe(self, handler: Callable[[WearablePacket, float], None]) -> None:
        """Register a callback executed per accepted packet (packet, receive time)."""
        self._handlers.append(handler)

    # -- diagnostics -----------------------------------------------------------
    def recent_intervals_s(self) -> List[float]:
        """Inter-arrival intervals from accepted packets (for jitter checks)."""
        stamps = [t for t, _ in self._received_log]
        return [
            stamps[i + 1] - stamps[i]
            for i in range(len(stamps) - 1)
        ]

    # -- internals ---------------------------------------------------------------
    def _count_crc_error(self, e: PacketError) -> None:
        message = str(e)
        if "CRC()" in message or "CRC" in message:
            self.stats.crc_errors += 1
        else:
            # Anything else (version, format) is also a decode rejection;
            # keep targeting size/CRC buckets and park distinct types here.
            self.stats.size_errors += 0
            self.stats.crc_errors += 1

    def _ordering_state(self, packet: WearablePacket) -> str:
        """Track sequence/boot transitions; returns a label for logging."""
        label = "in_order"
        if self._last_boot_id is not None and packet.boot_id != self._last_boot_id:
            self.stats.reboots_detected += 1
            logger.warning(
                "wearable reboot detected (boot_id %s -> %s): re-baselining order",
                self._last_boot_id, packet.boot_id,
            )
            self._last_sequence = None
            label = "reboot"
        elif self._last_boot_id is None:
            label = "first_packet"
        if self._last_sequence is not None and label != "reboot":
            delta = wrap_delta(self._last_sequence, packet.sequence)
            if delta == 0 or delta > OUT_OF_ORDER_WINDOW:
                self.stats.out_of_order += 1
                label = "out_of_order"
            elif delta > 1:
                # Missing intermediate sequence numbers (e.g. 1 -> 4 skipped
                # 2, 3) count as a gap within the sane window; larger deltas
                # wrap-safety-handled above as reorder/error.
                self.stats.gaps_detected += 1
                logger.info("sequence gap: %s -> %s", self._last_sequence, packet.sequence)
                label = "gap"
        self._last_sequence = packet.sequence
        self._last_boot_id = packet.boot_id
        return label


# ---------------------------------------------------------------------------
# Optional live transport (bleak). Present so host software can be exercised
# against real hardware with the same receiver semantics.
# ---------------------------------------------------------------------------


def make_bleak_receiver(
    address: str,
    stale_after_s: float = 3.0,
) -> tuple:
    """Build (async_bleak_runner, WearableReceiver) using bleak.

    Returns an asyncio coroutine `run()` that connects, subscribes to the
    GATT characteristic, pumps notifications into the receiver and applies
    bounded reconnect backoff on connection loss. bleak must be installed.
    """
    import asyncio

    import bleak  # imported only here so bleak absence is a targeted error

    from src.wearable.protocol import CHARACTERISTIC_UUID

    receiver = WearableReceiver(stale_after_s=stale_after_s)

    RECONNECT_BACKOFF_S = 2.0
    RECONNECT_MAX_BACKOFF_S = 30.0

    async def run(stop_event: Optional[asyncio.Event] = None):
        backoff = RECONNECT_BACKOFF_S
        while not (stop_event and stop_event.is_set()):
            try:
                async with bleak.BleakClient(address) as client:
                    receiver.on_connected()

                    def notification_handler(_sender: int, data: bytearray):
                        receiver.feed(bytes(data))

                    await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
                    backoff = RECONNECT_BACKOFF_S
                    while not (stop_event and stop_event.is_set()):
                        await asyncio.sleep(0.2)
                    await client.stop_notify(CHARACTERISTIC_UUID)
            except Exception as e:  # noqa: BLE001 - BLE stack raises varied types
                logger.warning("BLE connection lost (%s); retrying in %.1fs", e, backoff)
                receiver.on_disconnected()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, RECONNECT_MAX_BACKOFF_S)

    return run, receiver
