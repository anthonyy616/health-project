# Wearable Vital Signs Band — Hardware Roadmap

**For:** Vital Signs Monitoring System (final-year project)
**Audience:** Zero prior hardware experience assumed. Every step is spelled out.
**Scope:** Everything from "what is a breadboard" to a finished, worn, working band.

---

## Table of Contents

0. [How to Use This Document](#0-how-to-use-this-document)
1. [Hardware Goals Recap](#1-hardware-goals-recap)
2. [Constraints](#2-constraints)
3. [System Architecture](#3-system-architecture)
4. [Bill of Materials](#4-bill-of-materials)
5. [Absolute Basics — Read This Before Touching Anything](#5-absolute-basics--read-this-before-touching-anything)
6. [Simulate First: Wokwi](#6-simulate-first-wokwi)
7. [Phase 1 — Breadboard Prototype (Cheap ESP32)](#7-phase-1--breadboard-prototype-cheap-esp32)
8. [Phase 2 — Move to ESP32-S3](#8-phase-2--move-to-esp32-s3)
9. [Phase 3 — Soldering & Permanent Assembly](#9-phase-3--soldering--permanent-assembly)
10. [Phase 4 — Final Band Assembly](#10-phase-4--final-band-assembly)
11. [Communication Architecture](#11-communication-architecture)
12. [End-to-End Data Flow](#12-end-to-end-data-flow)
13. [Testing & Validation Protocol](#13-testing--validation-protocol)
14. [Troubleshooting Guide](#14-troubleshooting-guide)
15. [Safety Reference Sheet](#15-safety-reference-sheet)
16. [Timeline](#16-timeline)
17. [Glossary](#17-glossary)
18. [Appendix: Pin Maps & Checklists](#18-appendix-pin-maps--checklists)

---

## 0. How to Use This Document

Work through this **in order**. Each phase assumes the previous one is done and understood — don't skip to soldering before the simulator and breadboard stages work. If something in a later phase doesn't make sense, the answer is almost always in an earlier section (usually [Section 5](#5-absolute-basics--read-this-before-touching-anything) or the [Glossary](#17-glossary)).

Every code snippet here is a **minimum working example** for that one step — just enough to prove the wiring and concept work, not production firmware. You build up complexity one working step at a time.

---

## 1. Hardware Goals Recap

You're building a wrist/arm-worn band that:
1. Reads heart rate (and SpO2) from a **MAX30102** pulse oximeter (I2C).
2. Reads motion/orientation from an **MPU6050** IMU (I2C) — used to detect motion artifacts and, per your plan, as a respiration backup signal.
3. Runs on an **ESP32-S3** microcontroller.
4. Sends readings over **BLE** to your Raspberry Pi / laptop for fusion with the camera-based readings.
5. Shows live readings on a small **screen mounted on the band itself**, in addition to sending them to the computer.
6. Runs off a small **LiPo battery** for a reasonable number of hours.

---

## 2. Constraints

These shape every decision below — refer back here when something seems like an arbitrary choice.

| Constraint | Detail | Why it matters |
|---|---|---|
| **Power budget** | 900 mAh LiPo. ESP32-S3 with BLE active draws roughly 80–150 mA average (bursts higher during transmission); MAX30102 with LEDs on draws ~10–30 mA; MPU6050 draws ~4 mA; an OLED draws ~10–20 mA. | At ~150 mA average draw, a 900 mAh battery gives roughly 5–6 hours, not the >8h target in your plan. You will likely need duty-cycling (sleep between reads) or a bigger battery — flag this now, don't discover it at the end. |
| **I2C bus sharing** | MAX30102, MPU6050, and an I2C OLED can all sit on the *same two wires* (SDA/SCL) because I2C is a shared bus with per-device addresses. | Saves pins and wiring, but means all three devices must have **different addresses** (see [Section 11](#11-communication-architecture)) and you must handle bus errors from one device without freezing the others. |
| **Sample-rate consistency** | Per your project's own design principle: HR and respiration signal processing assume a fixed, unbroken sample rate. | The wearable's firmware must poll MAX30102 at a **fixed interval**, not "whenever the loop happens to get to it." Use a hardware timer or consistent `delay()`-based loop, not variable-length work in between reads. |
| **Physical size** | Everything (ESP32-S3 dev board, battery, sensors, wiring) has to fit inside a neoprene band without being bulky or uncomfortable. | Drives the choice of a *small* OLED (or none, if space is too tight — see [Section 4](#4-bill-of-materials)) and how tightly things are packed in [Phase 4](#10-phase-4--final-band-assembly). |
| **Skin contact requirement** | MAX30102 needs to sit flush against skin with minimal ambient light leakage to get a usable PPG signal. | Enclosure design (already covered in the CAD discussion) needs a snug, light-sealed window over just this sensor. |
| **Budget** | You've stated a Ā$200 total project budget. | Every BOM item below is chosen to be inexpensive; alternatives are only suggested when the cheap option has a real downside. |
| **Your skill level** | Zero hardware background. | Every phase below over-explains rather than under-explains. Skip what you already know. |

---

## 3. System Architecture

```
                         WEARABLE BAND
        ┌───────────────────────────────────────────────┐
        │                                                │
        │   MAX30102 ──┐                                 │
        │  (I2C 0x57)  │                                 │
        │              ├── I2C bus (SDA/SCL) ── ESP32-S3 │
        │   MPU6050  ──┤                          │      │
        │  (I2C 0x68)  │                          │      │
        │              │                          │      │
        │  OLED (opt.)─┘                     BLE radio   │
        │  (I2C 0x3C)                              │     │
        │                                          │      │
        │   LiPo Battery ── TP4056 charger ── ESP32-S3    │
        │                                                │
        └───────────────────────────────────────────────┘
                                    │
                                   BLE
                                    │
                                    ▼
                         RASPBERRY PI / LAPTOP
                    (bleak library receives packets)
                                    │
                                    ▼
                       Fusion Engine (camera + wearable)
                                    │
                                    ▼
                    Dashboard / Database / (future) Screen
```

Key design decision: **the on-band screen and the BLE link are independent outputs from the same firmware loop** — the ESP32 reads sensors once per cycle, then both draws to the OLED *and* sends a BLE packet. Neither depends on the other; if BLE disconnects, the screen keeps working, and vice versa.

---

## 4. Bill of Materials

### 4.1 Prototype Phase (breadboard, disposable-budget)

| Item | Approx. Price | Notes |
|---|---|---|
| ESP32 DevKitC (WROOM-32, **not** S3) | $4–6 | Same toolchain/peripherals as the S3; cheap enough to not worry about frying it |
| GY-MAX30102 breakout | $2–4 | Get 2 if budget allows, in case of a bad unit |
| GY-521 MPU6050 breakout | $2–3 | Same — get 2 if possible |
| SSD1306 128×64 I2C OLED | $2–4 | Small, cheap, well-documented, natively supported in the simulator (Section 6) |
| Full-size breadboard ×2 | $3–5 each | One for the microcontroller, one for sensors, or combine on one large breadboard |
| Jumper wire kit (M-M, M-F, F-F) | $3–5 | Buy a full assorted pack |
| USB cable matching your ESP32 board (Micro-USB or USB-C) | $3–5 | For programming and power |
| Small multimeter | $8–15 | Non-negotiable — you will need to check continuity and voltages |

**Prototype subtotal: ~$25–40**

### 4.2 Final Build

| Item | Approx. Price | Notes |
|---|---|---|
| ESP32-S3 DevKit-C-1 | $8–12 | The real target chip |
| GY-MAX30102 breakout (fresh unit) | $2–4 | Use a known-good one from prototyping, or a new one |
| GY-521 MPU6050 breakout (fresh unit) | $2–3 | Same |
| SSD1306 128×64 I2C OLED | $2–4 | Same unit type as prototype |
| 3.7V 900mAh LiPo battery (flat cell) | $6–10 | As per your existing plan |
| TP4056 LiPo charger module (with protection circuit) | $2 | **Get the protection-circuit version**, not bare TP4056 — see [Section 15](#15-safety-reference-sheet) |
| SPDT slide switch | $1 | Power on/off |
| Perma-proto board (small, solderable prototype board) | $2–4 | For a semi-permanent soldered assembly before final enclosure fit |
| JST-PH 2-pin connector pairs ×3 | $3 | Battery connector + 2 detachable joints |
| Thin silicone-insulated wire, 26–30 AWG, several colors | $5–8 | Flexes without cracking, unlike solid breadboard wire |
| Heat shrink tubing assortment | $3 | Insulate every solder joint that will flex |
| Elastic neoprene band | $5 | Physical housing, per your existing plan |

**Final-build subtotal: ~$45–65** (on top of what you already listed for the Pi/webcam/IR sensor side)

### 4.3 Tools & Consumables (one-time, shared across prototype + final)

| Item | Approx. Price | Notes |
|---|---|---|
| Soldering iron (temperature-adjustable, basic) | $15–25 | A $15 unit is fine for this scale of work |
| Leaded solder (60/40) + flux pen | $8 | Leaded solder is easier to work with as a beginner (lower melting point, flows better) — wash hands after handling |
| Solder wick / desoldering braid | $3 | For fixing mistakes |
| Small wire strippers | $6 | For the 26–30 AWG wire |
| Helping-hands / PCB holder (optional but recommended) | $8–12 | Holds boards steady while soldering — genuinely useful as a beginner |
| Anti-static wrist strap (optional) | $5 | Cheap insurance against ESD killing a chip — see Section 15 |
| Small fireproof LiPo charging bag/box | $6–10 | **Not optional if you're charging LiPo cells** — see Section 15 |

---

## 5. Absolute Basics — Read This Before Touching Anything

If you already know Ohm's law, breadboards, and I2C, skip to [Section 6](#6-simulate-first-wokwi).

### 5.1 Voltage, current, ground
- **Voltage (V)** is electrical "pressure" — how hard electrons are being pushed. Your components run on **3.3V** (ESP32, all three I2C sensors). The battery is nominally **3.7V**, close enough that it can power the 3.3V logic through the board's onboard regulator, but never wire the raw battery straight to a sensor's VCC pin if the sensor is 3.3V-only — check each breakout's datasheet/silkscreen for its accepted voltage range (most GY-breakouts have their own onboard regulator and accept 3.3–5V, but confirm before connecting).
- **Current (A or mA)** is how much charge flows. This is what determines battery life and what can overheat/damage a component if too much flows.
- **Ground (GND)** is the shared reference "zero" that every component's negative terminal connects back to. **Every single component in this project must share a common ground** — this is the single most common beginner wiring mistake (a component that "does nothing" is very often just missing its GND connection).

### 5.2 The breadboard
A breadboard is a reusable board with rows of metal clips **hidden under the plastic**, letting you connect components without soldering.
- The two outer rows (marked with **red +** and **blue −** lines) run the *full length* of the board — these are your power rails. Connect your 3.3V and GND here once, then tap into them anywhere along the board.
- The inner rows are grouped in short vertical strips of 5 holes (**a–e** and **f–j** typically, split by the center gap) — each strip of 5 is one electrically connected node. Plugging two component legs into the same strip connects them; plugging into different strips does not.
- The center gap is where you straddle IC/breakout modules so their two rows of pins don't short together.

### 5.3 I2C in one paragraph
I2C is a 2-wire communication protocol: **SDA** (data) and **SCL** (clock). Multiple devices share the same two wires; the microcontroller ("master") talks to one device at a time by its unique **address** (a small number like `0x57` or `0x68`). This is why MAX30102, MPU6050, and the OLED can all connect to the *same* SDA/SCL pins on the ESP32 — no separate wires needed per sensor, just shared SDA, shared SCL, shared GND, and each gets its own VCC.

### 5.4 Reading a breakout module's silkscreen
Every breakout board (GY-MAX30102, GY-521, SSD1306) has pin labels printed directly on the board ("silkscreen"). Before wiring anything, photograph or note down what's printed next to each pin — labels are almost always some combination of `VCC/VIN`, `GND`, `SDA`, `SCL`, and sometimes extra pins (`INT`, `AD0`, `XDA`, `XCL` on the MPU6050 — you can ignore these extras for this project).

### 5.5 ESD (electrostatic discharge) — the invisible killer
A static shock too small for you to feel can permanently damage a chip. Before handling any bare board (especially the ESP32), touch a grounded metal object (like a laptop's metal chassis while it's plugged in, or a radiator) to discharge yourself. If you bought the anti-static wrist strap, wear it and clip it to something grounded.

---

## 6. Simulate First: Wokwi

You asked for a way to test wiring/logic before real hardware. **Wokwi** (wokwi.com, free, browser-based, no install) is the right tool here — it's purpose-built for exactly this microcontroller ecosystem, unlike Tinkercad Circuits which has a much smaller, more generic parts library and no ESP32-S3 support.

### 6.1 What Wokwi can and can't simulate for you

| Component | Wokwi support | What this means for you |
|---|---|---|
| ESP32-S3 DevKitC-1 | ✅ Native, first-class support | Full firmware simulation, exactly matches real board behavior |
| MPU6050 | ✅ Native part (`wokwi-mpu6050`) | Realistic simulated accelerometer/gyro values, works with the same Adafruit_MPU6050 library you'll use on real hardware |
| SSD1306 OLED | ✅ Native part | Exact pixel-for-pixel simulation of what you'll see on the real screen |
| MAX30102 | ⚠️ Community-made "custom chip," not official | Good enough to test I2C wiring and firmware logic (does your code correctly initialize it, read registers, structure data), but **do not expect realistic PPG waveform/BPM values** out of the simulation — validate actual heart-rate accuracy only on real hardware with a real finger/wrist |

This distinction matters: use Wokwi to catch wiring mistakes and firmware bugs cheaply, but budget real bench-testing time for the MAX30102's actual signal quality — the simulator won't tell you that.

### 6.2 Setting up your first Wokwi project
1. Go to wokwi.com, sign up (free), click "New Project" → "ESP32-S3".
2. You'll get a `sketch.ino` (your Arduino code) and a `diagram.json` (the virtual wiring) side by side, with a "Start Simulation" button.
3. Search the parts library (left sidebar) for `mpu6050` and `ssd1306` and drag them onto the canvas.
4. Wire them in the diagram exactly as described in [Section 7.2](#72-step-by-step-wiring) below — Wokwi lets you click-and-drag wires between pins visually.
5. Paste in the test firmware from [Section 7.3–7.6](#73-firmware-stage-1--i2c-bus-scanner) and hit "Start Simulation." A virtual serial monitor shows your `Serial.print()` output live, and the virtual OLED updates in real time.

Do this for every firmware stage below **before** wiring the real breadboard — it will catch the majority of your mistakes for free.

---

## 7. Phase 1 — Breadboard Prototype (Cheap ESP32)

### 7.1 Safety basics first
- Never wire anything with the USB cable plugged in — always disconnect power before changing wiring.
- Double-check every connection against the pin map below *before* powering on — a shorted VCC-to-GND is the single fastest way to damage a board.
- Keep water/drinks away from your workspace.

### 7.2 Step-by-step wiring

All three I2C devices share the same 4 signal types. On a classic ESP32 DevKit, the **default** I2C pins are **GPIO21 (SDA)** and **GPIO22 (SCL)** — use these unless you have a specific reason not to.

| From (ESP32 pin) | To | Wire color (suggested) |
|---|---|---|
| 3V3 | Breadboard **+** rail | Red |
| GND | Breadboard **−** rail | Black |
| GPIO21 | SDA on MAX30102, MPU6050, **and** OLED (all three tied together) | Yellow |
| GPIO22 | SCL on MAX30102, MPU6050, **and** OLED (all three tied together) | Green |

For each of MAX30102 / MPU6050 / OLED individually:

| Breakout pin | Connects to |
|---|---|
| VCC / VIN | Breadboard **+** rail (3.3V) |
| GND | Breadboard **−** rail |
| SDA | ESP32 GPIO21 (shared bus) |
| SCL | ESP32 GPIO22 (shared bus) |

**Do this now, on the breadboard, with power off:**
1. Place the ESP32 DevKit straddling the center gap of one breadboard.
2. Run a red wire from ESP32 `3V3` to the `+` rail, a black wire from `GND` to the `−` rail.
3. Place MAX30102, MPU6050, and OLED breakouts on the breadboard (or a second one, connected rail-to-rail with red/black wires).
4. Wire each breakout's VCC to `+`, GND to `−`.
5. Wire each breakout's SDA to ESP32 GPIO21, SCL to GPIO22 (you'll have 3 yellow wires all landing on GPIO21's row, and 3 green wires on GPIO22's row — this is correct, it's how a shared bus looks).
6. **Before plugging in USB:** visually trace every wire back to its start and confirm nothing crosses rails (no red touching black anywhere).

### 7.3 Firmware Stage 1 — I2C bus scanner

**Goal:** confirm all three devices are electrically present and talking, before writing any sensor-specific code. This single sketch will save you hours of confused debugging later.

```cpp
#include <Wire.h>

void setup() {
  Wire.begin();          // uses default SDA=21, SCL=22 on classic ESP32
  Serial.begin(115200);
  delay(1000);
  Serial.println("I2C Scanner starting...");
}

void loop() {
  byte count = 0;
  Serial.println("Scanning...");
  for (byte address = 1; address < 127; address++) {
    Wire.beginTransmission(address);
    if (Wire.endTransmission() == 0) {
      Serial.print("Found device at 0x");
      Serial.println(address, HEX);
      count++;
    }
  }
  if (count == 0) {
    Serial.println("No I2C devices found — check wiring!");
  }
  delay(3000);
}
```

**Expected output** once wiring is correct: three lines, showing addresses around `0x3C` (OLED), `0x57` (MAX30102), and `0x68` (MPU6050). If you see fewer than three, see [Section 14](#14-troubleshooting-guide) before proceeding — do not move on until this passes.

### 7.4 Firmware Stage 2 — Read the MPU6050

Install via Arduino IDE Library Manager: **Adafruit MPU6050** and **Adafruit Unified Sensor**.

```cpp
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  if (!mpu.begin()) {
    Serial.println("MPU6050 not found!");
    while (1) delay(10);
  }
  Serial.println("MPU6050 ready.");
}

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  Serial.print("Accel X: "); Serial.print(a.acceleration.x);
  Serial.print(" Y: "); Serial.print(a.acceleration.y);
  Serial.print(" Z: "); Serial.println(a.acceleration.z);
  delay(200);
}
```

Confirm values change sensibly when you tilt/move the breadboard (Z should read roughly 9.8 m/s² when flat and still — that's gravity).

### 7.5 Firmware Stage 3 — Read the MAX30102

Install via Library Manager: **SparkFun MAX3010x Pulse and Proximity Sensor Library**.

```cpp
#include <Wire.h>
#include "MAX30105.h"

MAX30105 particleSensor;

void setup() {
  Serial.begin(115200);
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found!");
    while (1) delay(10);
  }
  particleSensor.setup(); // default settings: works for initial testing
  Serial.println("MAX30102 ready.");
}

void loop() {
  long irValue = particleSensor.getIR();
  Serial.print("IR: "); Serial.println(irValue);
  delay(20); // ~50 samples/sec — fixed interval, per your sample-rate constraint
}
```

Put a finger gently on the sensor — the IR value should jump from near-zero (nothing in front of it) to a large number (tens of thousands) and show a small periodic wobble as your pulse passes underneath. That wobble *is* your raw PPG signal — actual BPM extraction from it comes later in your existing `signal_processing.py`-equivalent firmware logic or by doing it host-side after BLE transmission (recommended: keep the ESP32 firmware simple and do POS/FFT-style processing on the Pi, matching your existing rPPG pipeline's philosophy of pure math functions separate from stateful I/O).

### 7.6 Firmware Stage 4 — Add the OLED

Install via Library Manager: **Adafruit SSD1306** and **Adafruit GFX**.

```cpp
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

Adafruit_SSD1306 display(128, 64, &Wire, -1);

void setup() {
  Serial.begin(115200);
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED not found!");
    while (1) delay(10);
  }
  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
}

void loop() {
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("HR: -- bpm");
  display.println("SpO2: -- %");
  display.display();
  delay(500);
}
```

Once this shows text, you've proven the full I2C bus (all three devices) works simultaneously — this is the point where you combine Stages 2–4 into one sketch that reads both sensors and updates the screen in the same loop, on a fixed timing interval.

### 7.7 Testing checklist before moving to Phase 2

- [ ] I2C scanner finds all 3 devices at their expected addresses
- [ ] MPU6050 values respond correctly to tilting the board
- [ ] MAX30102 IR value jumps when a finger is placed on it, and shows a periodic wobble
- [ ] OLED displays static text correctly
- [ ] Combined sketch reads both sensors **and** updates the OLED in one fixed-interval loop without crashing or freezing after several minutes running

---

## 8. Phase 2 — Move to ESP32-S3

The ESP32-S3 is *not* pin-identical to the classic ESP32 you prototyped with — most importantly, **it has no fixed default I2C pins**; you must specify them explicitly in code.

1. Repeat the exact same breadboard wiring as Phase 1, but on the ESP32-S3 DevKitC-1 board.
2. Check your specific S3 board's printed pinout diagram (varies slightly between vendors) for two free GPIOs. **GPIO8 (SDA) and GPIO9 (SCL)** are commonly free on most ESP32-S3 DevKitC-1 boards — confirm against your board's silkscreen before wiring.
3. Change one line in every sketch above:
   ```cpp
   Wire.begin(8, 9);  // SDA=8, SCL=9 — replace with your board's chosen pins
   ```
4. Re-run the full checklist from [Section 7.7](#77-testing-checklist-before-moving-to-phase-2) on the S3 board specifically — don't assume it'll "just work" because it worked on the classic ESP32.
5. Once confirmed working, add **BLE** — this is the point where you implement the GATT service described in [Section 11](#11-communication-architecture). Get comfortable with the sensor+display loop working reliably first; add BLE last, since it's the most likely source of new bugs.

---

## 9. Phase 3 — Soldering & Permanent Assembly

Only proceed here once Phase 2 is fully working and stable on the breadboard for an extended test run (leave it running for 30+ minutes without freezing or crashing).

1. **Practice first.** Before soldering anything that matters, practice on a scrap piece of wire and a spare header pin — get comfortable with heating the joint (not the solder directly), feeding solder in, and pulling away cleanly. A good joint looks shiny and cone-shaped, not dull/grainy (a "cold joint," which will fail).
2. **Transfer the wiring to a perma-proto board** in the same layout as your breadboard — this is a "practice run" for the final wearable assembly with a much smaller failure cost than soldering straight into the enclosure.
3. **Use short lengths of your 26–30 AWG silicone wire**, not the stiff breadboard jumpers, for every connection that will need to flex once mounted in the neoprene band.
4. **Heat-shrink every joint** that isn't rigidly supported by the board itself — this prevents shorts if wires shift during wear.
5. Re-run the full [Section 7.7 checklist](#77-testing-checklist-before-moving-to-phase-2) again on this soldered assembly before moving to final enclosure integration.

---

## 10. Phase 4 — Final Band Assembly

1. Fit-check the soldered assembly (board + sensors + OLED + battery) inside your CAD-designed enclosure *before* final wire routing — adjust the CAD model if anything doesn't clear.
2. Mount the MAX30102 directly behind the enclosure's optical window, flush against where skin will contact it, with minimal gap (light leakage between sensor and skin degrades signal quality).
3. Route wires with a small amount of slack (not taut) so wrist movement doesn't stress solder joints — this is why flexible silicone wire matters.
4. Connect the LiPo battery via its JST connector to the TP4056 module's output, and the TP4056's output to the ESP32-S3's battery input (check your specific S3 board's battery connector — many DevKitC-1 boards don't have one and you'll need to wire directly to a 3.3V/GND pin pair instead of a battery JST; confirm this on your specific board before assuming a JST battery port exists).
5. Install the SPDT power switch in-line between battery and the rest of the circuit.
6. Close the enclosure, do a final full-checklist test **before** sealing/gluing anything permanently.
7. Wear-test for comfort and signal quality over at least a 10-minute session before considering the build "done."

---

## 11. Communication Architecture

### 11.1 I2C bus (inside the band)

| Device | I2C Address | Bus speed |
|---|---|---|
| MAX30102 | `0x57` | 400 kHz (Fast Mode) |
| MPU6050 | `0x68` (default) or `0x69` if `AD0` pin is pulled high | 100–400 kHz |
| SSD1306 OLED | `0x3C` (default; some units are `0x3D`) | 100–400 kHz |

No address conflicts exist between these three at default settings — you don't need to reassign anything.

### 11.2 BLE (band → Raspberry Pi)

Design one custom GATT service with a single "notify" characteristic that pushes a structured packet whenever new data is ready — this matches the packet structure your existing project plan already sketched:

```
struct WearablePacket {
  uint32_t timestamp;       // ms since boot
  uint16_t sequence_number; // increments each packet, lets host detect drops
  uint8_t  heart_rate;      // BPM
  uint8_t  spo2;            // %
  int16_t  accel_x, accel_y, accel_z;  // raw or scaled MPU6050 accel
  uint8_t  wearable_sqi;    // 0-100 signal quality index
  uint8_t  battery_pct;     // 0-100
};
```

Send this as a single BLE notification (raw bytes, or JSON if you prefer simplicity over bandwidth efficiency — at this data rate, either works fine). On the Pi side, `bleak` (already in your stack) subscribes to the notify characteristic and parses each packet as it arrives.

---

## 12. End-to-End Data Flow

```
MAX30102/MPU6050 (fixed-interval sampling)
        ↓
ESP32-S3 firmware: read → compute SQI → package
        ↓                              ↓
   OLED display                   BLE notify
   (on-band, immediate)                ↓
                              Raspberry Pi (bleak)
                                        ↓
                         Fusion Engine (+ camera HR/SQI)
                                        ↓
                        Dashboard / Database / Logs
```

---

## 13. Testing & Validation Protocol

1. **Bench sanity check (not clinical validation):** compare the band's displayed HR against a phone pulse-measurement app or a basic consumer pulse oximeter while both are worn simultaneously, over several short sessions. This is a *plausibility* check, not a source of thesis-grade accuracy numbers.
2. **BLE reliability test:** run the band + Pi for 30+ minutes, log every received packet's `sequence_number`, and compute a packet-loss rate — this becomes a real, reportable metric for your thesis's wearable section.
3. **Battery life test:** fully charge, run continuously, log time until shutdown — compare against your >8h target and document the gap honestly (per Section 2, expect this to come in under target initially).
4. **Wear comfort test:** at least one full continuous session of realistic length, noting any pressure points, sensor slippage, or overheating.

---

## 14. Troubleshooting Guide

| Symptom | Most likely cause | Fix |
|---|---|---|
| I2C scanner finds 0 devices | Missing/swapped SDA-SCL, or missing GND connection | Re-check every wire against Section 7.2 — GND is the most commonly forgotten connection |
| I2C scanner finds some but not all devices | One breakout has a bad solder joint or is DOA | Swap in your spare unit; test it alone first |
| MAX30102 IR value stays near-zero even with finger on it | Sensor not seated flush, or ambient light leaking in | Press firmly, or test in a dim room first |
| MPU6050 values are frozen/identical every read | `mpu.getEvent()` not being called, or I2C bus locked by another device's fault | Comment out the other devices temporarily to isolate |
| OLED shows garbage/noise | Wrong I2C address passed to `display.begin()` | Re-run the scanner to confirm actual address (`0x3C` vs `0x3D`) |
| ESP32 keeps resetting/rebooting | Brownout — insufficient current from USB port or battery | Try a different USB port/cable, or a powered USB hub; on battery, check charge level |
| BLE won't connect from Pi | `bleak` scanning for wrong device name/UUID, or ESP32 not advertising | Confirm the GATT UUIDs match exactly on both ends; use a phone BLE scanner app to confirm the ESP32 is advertising at all |
| Values look fine on the bench but junk once worn | Motion artifact, poor skin contact, or band too loose/tight | This is expected and part of why you're building the SQI/confidence system — log it rather than treating it as a hardware failure |

---

## 15. Safety Reference Sheet

### LiPo batteries — read this before your battery arrives
- **Always** use a charger module with a built-in protection circuit (the TP4056 module in the BOM includes one) — never charge a bare LiPo cell with no protection circuit.
- **Never** puncture, crush, bend sharply, or short-circuit a LiPo cell.
- **Charge in a fireproof LiPo bag/box**, on a non-flammable surface, and don't leave it charging unattended, especially overnight.
- **Stop using immediately** if a cell becomes swollen/puffy — this indicates internal damage and the cell should be safely disposed of (many electronics/battery recycling points accept LiPo cells; don't put in regular trash).
- Store at roughly half-charge if not using the project for an extended period, rather than fully charged or fully drained.

### General electrical
- Disconnect power before changing any wiring.
- Keep liquids away from your workspace.
- Double-check polarity (+ / −) before connecting the battery — reversed polarity can instantly destroy the ESP32 board.

### ESD
- Ground yourself before handling bare boards (Section 5.5).
- Store spare ICs/boards in anti-static bags if you have them (the bags breakout modules often ship in are usually anti-static — keep and reuse them).

---

## 16. Timeline

Mapping onto your existing project plan's **Month 3 (hardware setup)** window:

| Week | Activity |
|---|---|
| 1 | Order hardware ([Section 4](#4-bill-of-materials)); while waiting, complete all Wokwi simulation stages ([Section 6](#6-simulate-first-wokwi)) |
| 2 | Phase 1: breadboard prototype on cheap ESP32, through full checklist ([Section 7](#7-phase-1--breadboard-prototype-cheap-esp32)) |
| 3 | Phase 2: repeat on ESP32-S3, add BLE ([Section 8](#8-phase-2--move-to-esp32-s3)) |
| 4 | Phase 3 + 4: solder, assemble final band, run full validation protocol ([Sections 9–13](#9-phase-3--soldering--permanent-assembly)) |

---

## 17. Glossary

- **Breadboard**: reusable board for wiring components without soldering.
- **GPIO**: General Purpose Input/Output — a microcontroller pin you can control in software.
- **I2C**: a 2-wire protocol (SDA + SCL) letting multiple devices share one communication bus via unique addresses.
- **GATT**: the data structure BLE devices use to organize "services" and "characteristics" — think of it as BLE's version of an API.
- **PPG**: photoplethysmography — the light-absorption-based technique MAX30102 uses to detect blood volume changes (your pulse).
- **SQI**: Signal Quality Index — a confidence score for how trustworthy a given reading is.
- **Silkscreen**: the printed labels/markings on a circuit board.
- **Brownout**: when a device resets because it isn't getting enough stable power.
- **ESD**: Electrostatic Discharge — static electricity that can damage electronics.

---

## 18. Appendix: Pin Maps & Checklists

### Full pin map (Phase 1, classic ESP32 prototype)

| ESP32 Pin | Connects to |
|---|---|
| 3V3 | Breadboard `+` rail |
| GND | Breadboard `−` rail |
| GPIO21 (SDA) | MAX30102 SDA, MPU6050 SDA, OLED SDA |
| GPIO22 (SCL) | MAX30102 SCL, MPU6050 SCL, OLED SCL |

### Full pin map (Phase 2, ESP32-S3 final)

| ESP32-S3 Pin | Connects to |
|---|---|
| 3V3 | Breadboard `+` rail |
| GND | Breadboard `−` rail |
| GPIO8 (SDA)* | MAX30102 SDA, MPU6050 SDA, OLED SDA |
| GPIO9 (SCL)* | MAX30102 SCL, MPU6050 SCL, OLED SCL |

\* Confirm against your specific board's pinout diagram before wiring — S3 dev boards vary by vendor.

### Pre-power-on checklist (every single time you change wiring)
- [ ] No wire crosses from `+` rail directly to `−` rail
- [ ] Every breakout has both VCC and GND connected (missing GND is the #1 mistake)
- [ ] SDA/SCL not swapped on any device
- [ ] Battery polarity correct, if connected
- [ ] USB cable disconnected while making changes, reconnected only once you're done