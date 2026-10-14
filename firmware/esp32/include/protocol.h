/*
 * Wearable BLE packet protocol v1 - firmware reference encoder.
 *
 * BYTE-LAYOUT CONTRACT: must byte-for-byte mirror
 * `src/wearable/protocol.py` (little-endian, 23 bytes, CRC16-XMODEM over
 * bytes 0..20). The Python suite (tests/test_wearable.py) holds the
 * golden round-trip fixtures; if you change one side you must change the
 * other and rerun those tests together with a captured hardware log.
 */
#ifndef WEARABLE_PROTOCOL_H
#define WEARABLE_PROTOCOL_H

#include <Arduino.h>
#include <string.h>

#define PROTOCOL_VERSION 1
#define PACKET_SIZE 23

/* flags bitmask (host: FLAG_HR etc.) */
#define FLAG_HR           0x0001
#define FLAG_SPO2         0x0002
#define FLAG_BATTERY      0x0004
#define FLAG_PPG_VALID    0x0008
#define FLAG_MOTION_VALID 0x0010

enum SensorStatus : uint8_t {
  STATUS_OK               = 0,
  STATUS_PPG_SENSOR_ERROR = 1,
  STATUS_IMU_SENSOR_ERROR = 2,
  STATUS_I2C_RECOVERY     = 3,
  STATUS_LOW_POWER        = 4,
  STATUS_CALIBRATING      = 5,
};

struct WearablePacket {
  uint32_t boot_id;         // random at boot
  uint32_t sequence;        // monotonic, wraps at 2^32
  uint32_t device_ticks_ms; // ms since boot
  bool     hr_valid;        uint8_t  hr_bpm;       // bpm (0..255 clamp)
  bool     spo2_valid;      uint8_t  spo2_pct;
  bool     ppg_sq_valid;    float    ppg_sq_01;    // 0..1
  bool     motion_valid;    float    motion_01;    // 0..1
  SensorStatus status;
  bool     battery_valid;   uint8_t  battery_pct;
};

/* CRC16-XMODEM (poly 0x1021, init 0x0000), over the first 21 bytes. */
static inline uint16_t crc16_xmodem(const uint8_t* data, size_t len) {
  uint16_t crc = 0;
  for (size_t i = 0; i < len; ++i) {
    crc ^= (uint16_t)data[i] << 8;
    for (int b = 0; b < 8; ++b) {
      crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    }
  }
  return crc;
}

/* Pack into a 23-byte buffer (values clamped; refuses hr > 255). */
static inline void pack_packet(WearablePacket& p, uint8_t out[PACKET_SIZE]) {
  memset(out, 0, PACKET_SIZE);
  uint16_t flags = 0;
  if (p.hr_valid)      flags |= FLAG_HR;
  if (p.spo2_valid)    flags |= FLAG_SPO2;
  if (p.battery_valid) flags |= FLAG_BATTERY;
  if (p.ppg_sq_valid)  flags |= FLAG_PPG_VALID;
  if (p.motion_valid)  flags |= FLAG_MOTION_VALID;

  uint8_t hr        = p.hr_valid      ? (uint8_t)min((int)p.hr_bpm, 255) : 0;
  uint8_t spo2      = p.spo2_valid    ? p.spo2_pct : 0;
  int8_t  ppg_sq    = p.ppg_sq_valid  ? (int8_t)roundf(clamp(p.ppg_sq_01, 0.0f, 1.0f) * 255.0f) : 0;
  uint8_t motion_sq = p.motion_valid  ? (uint8_t)roundf(clamp(p.motion_01, 0.0f, 1.0f) * 255.0f) : 0;
  uint8_t battery   = p.battery_valid ? p.battery_pct : 0;

  /* Body bytes 0..20, little-endian struct matching Python "<BIIIHBBBBBB". */
  out[0]  = PROTOCOL_VERSION;
  memcpy(&out[1],  &p.boot_id, 4);
  memcpy(&out[5],  &p.sequence, 4);
  memcpy(&out[9],  &p.device_ticks_ms, 4);
  memcpy(&out[13], &flags, 2);
  out[15] = hr;
  out[16] = spo2;
  out[17] = (uint8_t)ppg_sq;
  out[18] = (uint8_t)p.status;
  out[19] = motion_sq;
  out[20] = battery;

  uint16_t crc = crc16_xmodem(out, 21);
  out[21] = (uint8_t)(crc & 0xFF);
  out[22] = (uint8_t)(crc >> 8);
}

#endif  // WEARABLE_PROTOCOL_H
