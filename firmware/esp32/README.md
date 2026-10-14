# ESP32 Wearable Firmware (Phase I)

Target: ESP32-WROOM-32 DevKit prototype (sacrificial) → final ESP32-S3.
Sensors: MAX30102 pulse oximeter (I²C 0x57) + MPU-6050 IMU (I²C 0x68).
Transport: BLE GATT notifications matching `src/wearable/protocol.py` v1.

> **Integrity note:** this firmware is the reference implementation for the
> documented protocol. It has NOT been flashed or validated on physical
> hardware yet — the Python side (protocol, receiver, mock transport) is
> software-verified with simulated packets. Physical validation is an
> explicitly tracked outstanding requirement (see `docs/IMPLEMENTATION-PROGRESS.md`).

## Repository layout (proposed)

```
firmware/esp32/
├── platformio.ini              # PlatformIO project (espressif32 + libs)
├── include/
│   └── protocol.h              # mirrors src/wearable/protocol.py byte layout
└── src/
    └── main.cpp                 # acquisition loop, I²C init/recovery, BLE GATT
```

## Status

- `platformio.ini`: written (build targets for WROOM-32 and S3 variants).
- `include/protocol.h`: reference v1 packet encoder (little-endian, CRC16-XMODEM).
- `src/main.cpp`: acquisition + GATT notify skeleton, deterministic test
  packet mode (`SIMULATE=1`) producing the exact packets the Python
  receiver already ingests in tests.

Do not claim hardware functionality until the flashing checklist below is
executed on real boards and a packet round-trip is captured with the
receiver.

## MAX30102 wiring (prototype, 3.3 V logic only)

| MAX30102 (GY-MAX30102 / Sparkfun) | ESP32-WROOM-32        |
|----------------------------------|-----------------------|
| VIN                              | 3V3                   |
| GND                              | GND                   |
| SCL                              | GPIO 22 (I²C default) |
| SDA                              | GPIO 21 (I²C default) |
| INT/IRD/RD (optional)            | GPIO 19 (interrupt)   |

Notes: onboard pull-ups are typically present on breakouts; do not add
strong second pull-ups. Red LED = 660 nm, IR = 880 nm heater current is
set by firmware; finger-clip placement first (wrist SpO₂ is NOT validated).

## MPU-6050 (GY-521) wiring

| MPU-6050 | ESP32      |
|----------|------------|
| VCC      | 3V3        |
| GND      | GND        |
| SCL      | GPIO 22    |
| SDA      | GPIO 21    |
| AD0      | GND (addr 0x68) |
| XDA/XCL  | unconnected |

Both devices share the same I²C bus. Motion score in firmware = simple
accelerometer magnitude deviation, NOT respiration measurement.

## Flashing & debug checklist (to execute on hardware)

1. Install: `pip install platformio`; connect ESP32 via USB.
2. `pio run -e wroom32 -t flash -t monitor` (baud 115200).
3. Confirm boot banner prints sensor discovery (MAX30102 ID 0x15, MPU who-am-i 0x68).
4. In `SIMULATE=0` mode verify a real PPG amp-vs-time plot when the finger
   is placed on the sensor window.
5. On host: `python -c "from src.wearable.client import ..."` with the
   BleakLiveReceiver` connecting by MAC/UUID, capture 60 s, verify:
   - decoded sequence monotonic (wrap-aware)
   - zero CRC errors over >1000 packets
   - reboot recovery: press MCU reset, session continues with new boot_id.
6. Record logs in `test_logs/wearable-hardware-*` (NOT committed if they
   contain personal data).

## BLE GATT

- Service UUID: `0000A5A5-0000-1000-8000-00805F9B34FB`
- Notifications characteristic: `0000A5A6-...` (23-byte packet, notify)
- Optional raw-PPG characteristic for debugging: emit 6-byte samples
  (uint16 IR, uint16 RED, uint16 ticks/4).

Same protocol is intended for the ESP32-S3 final variant (BLE API differs;
`Ethernet`/`NimBLE-Arduino` selection decided in platformio.ini env).

## Safety constraints

- No bare lithium-battery improvisation; USB power only until protection
  circuitry is validated.
- LED current limits mirrored from MAX30102 datasheet; do not exceed
  datasheet thermal envelope on continuous wrist placement.
- No mains-connected exposed prototypes worn by people.
- SpO₂ derived values are research-only until calibration against a
  clinical reference device; refuse to label them medical data.
