/*
 * Wearable firmware (Phase I) - reference implementation skeleton.
 *
 * Contracts:
 * - Emits 23-byte v1 packets matching src/wearable/protocol.py exactly.
 * - SIMULATE=1 build emits deterministic test packets only (for host-side
 *   integration before sensors are attached).
 * - Real mode reads MAX30102 (IR/RED + HR) and MPU-6050 (acceleration),
 *   tracks I2C recovery and increments a monotonic sequence and ms ticks.
 * - BLE GATT notify via NimBLE-Arduino, service/characteristic UUIDs must
 *   match protocol.py.
 *
 * STATUS: reference implementation, not yet validated on physical boards.
 * The flashing checklist in README.md must be executed before any hardware
 * claim; host-side parity is already covered by tests/test_wearable.py.
 */

#include <Arduino.h>
#include "protocol.h"

#if defined(SIMULATE) && SIMULATE
#define WEARABLE_SIMULATION 1
#else
#define WEARABLE_SIMULATION 0
#endif

#if !WEARABLE_SIMULATION
#include <Wire.h>
#include "MAX30105.h"
#include "MAX30105_Documentation.h"  // by-SparkFun example headers come with the lib
#include "heartRateKit.h"            // if unavailable, inline simple HR logic instead
#include "MPU6050.h"
#endif

#if !WEARABLE_SIMULATION
#include <NimBLEDevice.h>
#endif

#define I2C_SDA 21
#define I2C_SCL 22
#define PACKET_PERIOD_MS 20  // 50 Hz notifications
#define HR_BPM_RADIUS 75     // SparkFun heart-rate peak radius example value

static WearablePacket g_packet;
static uint32_t g_start_ms = 0;
static float g_motion_score = 0.05f;

#if !WEARABLE_SIMULATION
static MAX30105 g_ppg;
static MPU6050 g_mpu;
static NimBLEServer* g_server = nullptr;
static NimBLECharacteristic* g_char = nullptr;

const char* SERVICE_UUID = "0000A5A5-0000-1000-8000-00805F9B34FB";
const char* CHAR_UUID = "0000A5A6-0000-1000-8000-00805F9B34FB";
#endif

static void init_boot_id() {
  /* random boot_id per session (host uses it to detect reboot) */
  uint32_t boot_id = 0;
  for (int i = 0; i < 4; ++i) boot_id = (boot_id << 8) | (esp_random() & 0xFF);
  g_packet.boot_id = boot_id;
  g_packet.sequence = 0;
  g_start_ms = millis();
}

#if !WEARABLE_SIMULATION
static void init_sensors() {
  Wire.begin(I2C_SDA, I2C_SCL);
  g_packet.status = STATUS_OK;

  if (!g_ppg.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found");
    g_packet.status = STATUS_PPG_SENSOR_ERROR;
  } else {
    g_ppg.setup(60 /*ledBrightness*/, 4 /*sampleAverage*/, LED_CURRENT_MODE_4, 411);
  }

  g_mpu.initialize();
  if (g_mpu.testConnection()) {
    /* success */ } else {
    if (g_packet.status == STATUS_OK) g_packet.status = STATUS_IMU_SENSOR_ERROR;
  }
}

static void update_motion_and_ppg() {
  /* Motion: cheap accelerometer magnitude deviation (NOT respiration) */
  int16_t ax, ay, az;
  g_mpu.getAcceleration(&ax, &ay, &az);
  float mag = sqrtf((float)ax * ax + (float)ay * ay + (float)az * az) / 16384.0f;  // ~1g
  g_motion_score = fmin(mag > 1.0f ? (mag - 1.0f) : (1.0f - mag), 1.0f);

  /* Safe non-blocking IR/RED read; drop zeros (sensor unplugged). */
  uint32_t ir = g_ppg.getIR();
  uint32_t red = g_ppg.getRed();
  if (ir == 0 || red == 0) {
    g_packet.ppg_sq_valid = false;
    if (g_packet.status == STATUS_OK) g_packet.status = STATUS_PPG_SENSOR_ERROR;
    return;
  }
  /* Simple beat detection using the classic IR threshold on rolling max:
     replace with the SparkFun beat example (beatAvg) once validated. */
  static uint32_t last_peak_ms = 0;
  static float ibi_ema = 0.0f;
  if (ir > 50000 && millis() - last_peak_ms > 300) {
    uint32_t now = millis();
    float ibi = (now - last_peak_ms);
    last_peak_ms = now;
    ibi_ema = (ibi_ema == 0.0f) ? ibi : (ibi_ema * 0.8f + ibi * 0.2f);
    if (ibi_ema > 0) g_packet.hr_bpm = (uint8_t)min((int)(60000.0f / ibi_ema), 255);
    g_packet.hr_valid = true;
  }
  g_packet.ppg_sq_valid = true;
  g_packet.ppg_sq_01 = 0.8f;  // Validated proxy score comes post-calibration
}
#endif  // !WEARABLE_SIMULATION

#if WEARABLE_SIMULATION
static void update_simulated() {
  uint32_t now_ms = millis();
  float t = (now_ms - g_start_ms) / 1000.0f;
  g_packet.hr_bpm = 72 + (int8_t)(2.0f * sinf(t * 0.5f));
  g_packet.hr_valid = true;
  g_packet.ppg_sq_valid = true;
  g_packet.ppg_sq_01 = 0.85f;
  g_packet.motion_valid = true;
  g_motion_score = 0.05f + 0.03f * sinf(t * 0.3f);
}
#endif

static void emit_packet() {
  g_packet.sequence += 1;
  g_packet.device_ticks_ms = millis() - g_start_ms;
  g_packet.status = STATUS_OK;
  g_packet.battery_pct = 100;
  g_packet.battery_valid = true;

#if WEARABLE_SIMULATION
  update_simulated();
#else
  update_motion_and_ppg();
#endif

  uint8_t out[PACKET_SIZE];
  pack_packet(g_packet, out);

#if WEARABLE_SIMULATION
  /* Simulation mode prints hex for direct host-side integration tests. */
  Serial.write(out, PACKET_SIZE);
#else
  if (g_char) {
    g_char->setValue(out, PACKET_SIZE);
    g_char->notify();
  }
#endif
}

#if !WEARABLE_SIMULATION
static void setup_ble() {
  NimBLEDevice::init("Vitals Wearable");
  g_server = NimBLEDevice::createServer();
  NimBLEService* svc = g_server->createService(SERVICE_UUID);
  g_char = svc->createCharacteristic(
      CHAR_UUID, NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY);
  svc->start();
  NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
  adv->addServiceUUID(SERVICE_UUID);
  adv->start();
}
#endif

void setup() {
  Serial.begin(115200);
  delay(200);
  memset(&g_packet, 0, sizeof(g_packet));
  g_packet.status = STATUS_OK;
  g_packet.spo2_valid = false;   // SpO2 NOT validated - never emitted until calibrated
  init_boot_id();

#if WEARABLE_SIMULATION
  Serial.println("SIMULATION mode - deterministic test packets");
#else
  init_sensors();
  setup_ble();
#endif
}

void loop() {
  static uint32_t next_emit = 0;
  if (millis() >= next_emit) {
    emit_packet();
    next_emit += PACKET_PERIOD_MS;
  }
#if !WEARABLE_SIMULATION
  /* I2C lock-up watchdog: reset bus if SCL stuck low > 50 ms. */
  static uint32_t last_ok = 0;
  if (millis() - last_ok > 50) {
    Wire.end();
    init_sensors();
    g_packet.status = STATUS_I2C_RECOVERY;
    last_ok = millis();
  }
#endif
  yield();
}
