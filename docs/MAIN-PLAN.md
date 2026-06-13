# Vital Signs Monitoring System — Complete Project Plan
**Final Year Graduation Project**
Last Updated: June 2026 | Deadline: December 2026 (Writeup) → April 2027 (Implementation)

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Current Status Snapshot](#2-current-status-snapshot)
3. [Hardware Procurement](#3-hardware-procurement)
4. [Phase-by-Phase Plan](#4-phase-by-phase-plan)
5. [Module-by-Module Technical Breakdown](#5-module-by-module-technical-breakdown)
6. [Thesis Writing Plan](#6-thesis-writing-plan)
7. [Volunteer Testing Protocol](#7-volunteer-testing-protocol)
8. [Risk Register](#8-risk-register)
9. [Accuracy & Performance Targets](#9-accuracy--performance-targets)
10. [Key Decisions Log](#10-key-decisions-log)

---

## 1. Project Summary

A hybrid contactless + wearable system for real-time vital signs monitoring in clinical settings. Combines computer vision (webcam + IR sensor) with a wearable elastic band (ESP32 + sensors) to monitor:

- Heart rate (rPPG + MAX30102)
- Body temperature (MLX90614 IR sensor)
- Respiratory rate (chest motion detection + IMU)
- Age estimation (facial ML model)
- Pupil dilation (MediaPipe Iris)
- Blood pressure (non-invasive estimation, stretch goal)

**Target:** 90–95% aggregate accuracy, <10 second latency per scan, tested on 30 volunteers.

**Stack:** Python 3.9.1, PyTorch, MediaPipe, FastAPI, MySQL, ESP32 (BLE), Raspberry Pi 4B.

---

## 2. Current Status Snapshot

### Completed

| Module | Status | Notes |
|--------|--------|-------|
| Project structure & venv | ✅ Done | Python 3.9.1 |
| Face detection (MediaPipe) | ✅ Done | 478 landmarks, webcam overlay, 30fps |
| UTKFace dataset | ✅ Done | 66,918 images downloaded & preprocessed |
| LightAgeNet (from scratch) | ✅ Done | MAE ~8.5 yrs — baseline only |
| MobileNetV3 (transfer) | ✅ Done | Improved accuracy |
| EfficientNet-B0 (transfer) | ✅ Done | Best model, current MAE ~5–6 yrs |
| Age-balanced sampling | ✅ Done | WeightedRandomSampler |
| Huber loss training | ✅ Done | More robust to outliers |
| Webcam real-time age demo | ✅ Done | `python -m src.main --mode age` |
| MySQL schema + DB module | ✅ Done | patients, readings, alerts, volunteers tables |

### In Progress / Not Started

| Module | Status | Priority |
|--------|--------|----------|
| rPPG heart rate | ⏳ Not started | HIGH — start next |
| Respiratory rate | ⏳ Not started | HIGH |
| Pupil detection | ⏳ Not started | MEDIUM |
| FastAPI dashboard | ⏳ Not started | MEDIUM |
| Hardware (Pi, ESP32, sensors) | ⏳ Not ordered | ORDER NOW |
| BLE communication | ⏳ Not started | Must-have |
| Kalman filter fusion | ⏳ Not started | Phase 3 |
| Volunteer testing | ⏳ Not started | Phase 4 |
| UBFC-rPPG dataset | ⏳ Not acquired | Kaggle mirror found |

### Age Model Decision

**Current MAE: ~5–6 years. Accept this and move on.**

The target is ±4 years. You are close. Face alignment preprocessing (`scripts/preprocessing/preprocess_aligned.py`) exists and can be applied later if needed. Don't get stuck here — rPPG is the harder problem and needs more time.

---

## 3. Hardware Procurement

**Order immediately — AliExpress shipping to Turkey takes 3–4 weeks.**

### Core System

| Component | Spec | Purpose | Est. Price |
|-----------|------|---------|------------|
| Raspberry Pi 4B | **4GB RAM** (not 2GB) | Central hub + ML inference | $55 |
| Official Pi 4 Power Supply | 5.1V / 3A USB-C | Stable power — cheap ones crash Pi | $8 |
| MicroSD Card | 32GB, Class 10 / A2 rated | OS + model storage | $10 |
| Logitech C920 Webcam | 1080p, USB | Primary camera | $60 |
| MLX90614 IR Sensor | I2C module version | Contactless temperature | $12 |

> **Note:** Get the MLX90614 as a breakout **module** (with I2C header pins), not the bare chip.

### Wearable Band

| Component | Spec | Purpose | Est. Price |
|-----------|------|---------|------------|
| ESP32-S3 DevKit-C | BLE + WiFi, 240MHz | Band microcontroller | $10 |
| MAX30102 Module | I2C, 3.3V | Pulse oximeter / HR backup | $5 |
| MPU6050 Module | I2C, ±2g accel | IMU for respiration backup | $3 |
| 3.7V 900mAh LiPo | Flat cell | Band power source | $8 |
| TP4056 Module | With overcharge protection | LiPo charging | $2 |
| Elastic Neoprene Band | ~5cm wide | Physical housing for sensors | $5 |

### Ground Truth Equipment (For Volunteer Testing)

| Component | Purpose | Est. Price |
|-----------|---------|------------|
| Finger-clip pulse oximeter | Heart rate ground truth | $15 |
| Digital thermometer (oral/forehead) | Temperature ground truth | $10 |
| Stopwatch / phone timer | Manual breath counting | $0 |

### Accessories

| Item | Why | Est. Price |
|------|-----|------------|
| Pi heatsink + small fan | Pi throttles under ML load without cooling | $8 |
| Jumper wires (M-M, M-F, F-F assorted) | Sensor connections | $5 |
| Small breadboard | Prototyping before any soldering | $3 |
| USB-C to USB-A cable | Programming the ESP32 | $3 |

### Total Estimate: ~$165–185 (within $200 budget)

### Where to Buy

- **Raspberry Pi:** Check [rpilocator.com](https://rpilocator.com) for stock. In Turkey: Robotistan or Robolink TR for local sourcing.
- **Sensors + ESP32:** AliExpress (cheapest). Amazon Turkey if you need faster delivery.
- **Webcam:** MediaMarkt or Vatan Bilgisayar locally saves shipping time.

### What NOT to Buy

- Touchscreen display — dashboard runs in browser
- CSI ribbon camera module — USB webcam is simpler and compatible
- Extra LiPo batteries — one is sufficient for a prototype

---

## 4. Phase-by-Phase Plan

### Timeline Overview

```
Month 1 (Jun–Jul 2026):   rPPG + Respiratory Rate
Month 2 (Jul–Aug 2026):   Pupil Detection + Pipeline Unification
Month 3 (Aug–Sep 2026):   Hardware arrives + Raspberry Pi setup + BLE prototype
Month 4 (Sep–Oct 2026):   Dashboard + Database Integration + Sensor Fusion
Month 5 (Oct–Nov 2026):   Volunteer Testing (30 people)
Month 6 (Nov–Dec 2026):   Optimization + Writeup finalization
Month 7–9 (Jan–Apr 2027): Full implementation polish + final presentation prep
```

---

### Phase 1 — Complete CV Modules (Month 1–2)

**Goal:** All contactless vital sign modules working on laptop webcam before any hardware arrives.

#### Month 1 Focus: rPPG + Respiratory Rate

**rPPG Heart Rate** is the priority. It is the hardest CV module — signal is weak, noise-sensitive, and accuracy degrades with motion, lighting changes, and darker skin tones. Start it first to give maximum iteration time.

Key approach:
- Extract forehead ROI using MediaPipe landmark indices (landmarks 10, 338, 297, 332, 284 for forehead region)
- Visualize raw RGB signal first before adding any filtering — if the waveform looks like noise, the ROI extraction is wrong
- Build 10-second sliding window (300 frames at 30fps)
- Apply POS (Plane-Orthogonal-to-Skin) algorithm
- Bandpass filter: 0.7–4 Hz (42–240 BPM)
- FFT to extract dominant frequency → convert to BPM
- Confidence scoring based on signal quality index

**Respiratory Rate** after rPPG is stable:
- Define chest ROI (below face bounding box, upper torso)
- Optical flow for vertical motion tracking
- Bandpass filter: 0.1–0.5 Hz (6–30 BPM)
- Peak detection → breaths per minute
- Alternative: MediaPipe Pose shoulder landmarks if chest ROI is unreliable

**UBFC-rPPG Dataset:**
- The Kaggle mirror at [malekdinarito/ubfc-rppg-dataset](https://www.kaggle.com/datasets/malekdinarito/ubfc-rppg-dataset) can be used to start
- Verify all 42 subjects have ground truth `.txt` files (PPG signal + BPM per frame)
- If any are missing, request official access at sites.google.com/view/yaboromance/ubfc-rppg (takes 1–7 days)
- **Do this now — don't wait until you need the dataset to request it**

#### Month 2 Focus: Pupil Detection + Pipeline

**Pupil Detection** is the quickest win — MediaPipe Iris is mostly plug-and-play:
- 5 iris landmarks per eye
- Calculate iris diameter in pixels
- Convert to mm using eye width as reference (~24mm average adult eye width)
- Blink detection to filter invalid frames
- Lighting compensation for dilation accuracy

**Pipeline Unification:**
- Build `VitalsPipeline` class wrapping all modules
- Single `process_frame(frame)` call returns JSON with all vitals
- Error handling for module failures (face lost, low confidence, etc.)
- Exponential moving average for smooth real-time readings
- This is what the dashboard and database will plug into — make it clean

---

### Phase 2 — Hardware Setup (Month 3, parallel)

**Order hardware by end of Month 1** so it arrives during Month 2/3.

When Pi arrives:
1. Flash Raspbian OS, install Python 3.9.1 venv, clone project
2. Run existing CV modules on Pi — benchmark performance
3. If inference > 10 seconds, apply ONNX quantization earlier than planned
4. Confirm webcam works at 30fps on Pi

When ESP32 arrives:
1. Wire MAX30102 + MPU6050 on breadboard
2. Program ESP32 to read heart rate + accelerometer data
3. Implement BLE advertisement and notify characteristic
4. Connect from laptop/Pi using Python `bleak` library
5. Confirm heart rate data flows: ESP32 → BLE → Python

BLE note: Test on your actual OS. BLE with `bleak` has known quirks on Windows (adapter compatibility). If issues arise, test on Linux first (Raspberry Pi is Linux and will be cleaner).

---

### Phase 3 — Dashboard + Integration (Month 4)

Build in this order — do not build frontend before backend is stable:

1. FastAPI REST endpoints (`/api/vitals/latest`, `/api/patients`, `/api/statistics`, `/api/alerts`)
2. Auto-save pipeline readings to MySQL database
3. WebSocket endpoint (`/ws/vitals`) for live updates
4. HTML dashboard frontend with Chart.js graphs
5. Kalman filter fusion (rPPG heart rate + MAX30102 heart rate)
6. Alert system with configurable thresholds

Dashboard pages needed:
- Live vitals view (real-time metrics + camera feed)
- Patient management
- Session history + graphs
- ML statistics / model comparison
- Active alerts

---

### Phase 4 — Volunteer Testing (Month 5)

**Recruit 40+ people, aim for 30 completed sessions** (some will cancel).

Standardized testing environment:
- Consistent seating position, 50–70cm from camera
- Three lighting conditions tested: bright overhead, normal room, dim (evening)
- Record volunteer demographics: age, gender, skin tone (Fitzpatrick scale I–VI)
- Ground truth: finger pulse oximeter for HR, thermometer for temp, manual breath count for respiration rate

Per session procedure:
1. Register volunteer in database (anonymous ID: V001, V002...)
2. Collect demographics + consent
3. 30-second capture in each lighting condition
4. Record ground truth simultaneously
5. System outputs vitals → compare to ground truth → log session

Vary conditions deliberately:
- Different skin tones if possible (important for rPPG robustness claims in thesis)
- Some sessions with slight motion (natural fidgeting)
- Some with glasses, beards, makeup

Start recruiting in Month 4 — do not wait until testing starts.

---

### Phase 5 — Optimization + Writeup (Month 6)

Model optimization:
- Export age + HR models to ONNX
- Apply INT8 quantization
- Benchmark FP32 vs INT8 inference time on Pi
- Verify accuracy drop < 1%
- Apply face alignment preprocessing if age MAE still > 5 years

Thesis writing:
- Chapter 4 (Methodology) and Chapter 5 (Results) should be drafted during testing
- Final pass on all chapters + formatting in Month 6
- Target: complete draft by December 2026

---

### Phase 6 — Implementation Polish (Month 7–9, Jan–Apr 2027)

- Final integration of all modules on Raspberry Pi
- Elastic band fabrication (mount sensors on neoprene, tidy up wiring)
- End-to-end system demo: patient walks in, sits down, all vitals measured in <10s
- Presentation slides (15–20 slides)
- Live demo preparation + rehearsal
- Q&A preparation

---

## 5. Module-by-Module Technical Breakdown

### 5.1 Age Estimation (Current)

| Item | Detail |
|------|--------|
| Dataset | UTKFace (66,918 images) |
| Best model | EfficientNet-B0 (5.3M params, transfer learning) |
| Current MAE | ~5–6 years |
| Target MAE | ±4 years |
| Status | Accept current result. Apply face alignment if revisiting. |

Training command: `python training/age_detection/train.py --model efficientnet`

### 5.2 Heart Rate via rPPG

| Item | Detail |
|------|--------|
| Dataset | UBFC-rPPG (42 subjects, Kaggle mirror) |
| Method | POS algorithm, FFT-based BPM |
| Signal window | 10 seconds (300 frames @ 30fps) |
| Bandpass | 0.7–4 Hz |
| Target MAE | ±5 BPM |
| Risk | High — sensitive to lighting, motion, skin tone |

Files to create:
- `src/contactless/heart_rate/rppg.py`
- `src/contactless/heart_rate/signal_processing.py`

### 5.3 Respiratory Rate

| Item | Detail |
|------|--------|
| Method | Optical flow on chest ROI |
| Bandpass | 0.1–0.5 Hz (6–30 BPM) |
| Target MAE | ±2 BPM |
| Fallback | MediaPipe Pose shoulder landmarks |
| Risk | Low — simpler signal than rPPG |

Files to create:
- `src/contactless/respiration/detect.py`
- `src/contactless/respiration/motion_analysis.py`

### 5.4 Pupil Detection

| Item | Detail |
|------|--------|
| Method | MediaPipe Iris (5 landmarks per eye) |
| Output | Diameter in mm (2–8mm normal range) |
| Calibration | Eye width reference (~24mm) |
| Risk | Low — library handles most complexity |

Files to create:
- `src/contactless/pupil_detection/detect.py`
- `src/contactless/pupil_detection/iris_tracker.py`

### 5.5 Temperature

| Item | Detail |
|------|--------|
| Sensor | MLX90614 (I2C, ±0.5°C) |
| Interface | Raspberry Pi GPIO → I2C |
| Target accuracy | ±1°C |
| Dependency | Hardware arrival |

### 5.6 BLE Communication (Wearable Band)

| Item | Detail |
|------|--------|
| MCU | ESP32-S3 |
| Sensors | MAX30102 (HR/SpO2), MPU6050 (IMU) |
| Protocol | BLE GATT notify characteristic |
| Python library | `bleak` (cross-platform BLE) |
| Risk | Medium — BLE platform quirks on Windows |

### 5.7 Sensor Fusion

| Item | Detail |
|------|--------|
| Method | Kalman filter |
| Inputs | rPPG heart rate + MAX30102 heart rate |
| Weighting | Confidence score based |
| File | `src/central/fusion/kalman.py` |

### 5.8 FastAPI Dashboard

| Item | Detail |
|------|--------|
| Backend | FastAPI (sync endpoints + WebSocket) |
| Frontend | HTML + Jinja2 + Chart.js |
| Database | MySQL (auto-save all readings) |
| Live updates | WebSocket `/ws/vitals` |
| Alert system | Threshold-based, logged to DB |

---

## 6. Thesis Writing Plan

Write chapters progressively — do not leave everything to Month 6.

| Chapter | When to Write | Status |
|---------|--------------|--------|
| Ch. 1: Introduction | Already drafted in thesis.txt | Needs expansion |
| Ch. 2: Literature Review | Mostly drafted | Needs citation formatting |
| Ch. 3: System Design | Write during Phase 1–2 | Not started |
| Ch. 4: Methodology | Write during implementation (Month 1–4) | Not started |
| Ch. 5: Results & Analysis | Write during/after testing (Month 5) | Not started |
| Ch. 6: Discussion | Write Month 6 | Not started |
| Ch. 7: Conclusion | Write last | Not started |
| Appendices | Accumulate throughout | Ongoing |

### Chapter 4 (Methodology) — Key Sections to Cover

- Data acquisition: UTKFace, UBFC-rPPG, VIPL-HR (if obtained)
- Model development: from-scratch vs transfer learning comparison with metrics table
- rPPG algorithm: POS method, signal processing pipeline, confidence scoring
- Data fusion: Kalman filter implementation and rationale
- System architecture: contactless + wearable + central unit diagram
- Optimization: ONNX quantization, inference latency measurements

### Chapter 5 (Results) — Required Content

- Accuracy tables for each vital sign vs ground truth
- From-scratch vs pretrained model comparison
- Performance by lighting condition (bright / normal / dim)
- Performance by skin tone (Fitzpatrick scale)
- Latency measurements (startup, per-frame, end-to-end)
- Battery life test results (wearable band)
- Error distribution plots (use `src/db_connect/statistics.py`)

---

## 7. Volunteer Testing Protocol

### Pre-Testing Setup

- [ ] Prepare anonymous consent form (name not required — just ID, age, gender, skin tone)
- [ ] Set up fixed testing chair at consistent distance from camera (60cm)
- [ ] Mark floor position for chair
- [ ] Prepare three lighting setups: bright overhead on, one side lamp only, lights off with monitor ambient
- [ ] Calibrate ground truth devices (reset oximeter, zero thermometer)
- [ ] Test full system end-to-end before first volunteer

### Per-Session Checklist (30 volunteers)

1. Register volunteer: `INSERT INTO volunteers (identifier, age, gender, skin_tone, consent_given)`
2. Note Fitzpatrick skin tone scale (I–VI)
3. Record glasses / facial hair / makeup in session notes
4. Run three 30-second captures (bright / normal / dim lighting)
5. Record ground truth simultaneously for each:
   - Heart rate: finger oximeter (read at 15s and 30s, average)
   - Temperature: oral/forehead thermometer
   - Respiratory rate: manual count for 30 seconds × 2
6. Save session: `INSERT INTO test_sessions` with ground truth fields populated
7. Let volunteer see their results (good for recruitment word-of-mouth)

### Diversity Targets

- Aim for variation in age (18–60+ if possible)
- At least some coverage across Fitzpatrick I–IV skin tones
- Include people with glasses (tests pupil module robustness)
- Include at least a few sessions with deliberate slight motion

### Data Analysis After Testing

Run `src/db_connect/statistics.py`:

```
stats = MLStatistics()
stats.print_report()             # Full accuracy summary
stats.get_readings_by_condition()  # Performance by lighting/skin tone
stats.plot_error_distribution()    # Visual for thesis Chapter 5
```

---

## 8. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| rPPG accuracy below ±5 BPM in real conditions | High | High | Rely more on MAX30102 wearable reading; be transparent in thesis about hybrid approach |
| BLE connection instability on Windows | Medium | Medium | Develop BLE on Raspberry Pi (Linux) — cleaner BLE stack |
| Raspberry Pi CPU too slow for real-time pipeline | Medium | High | Benchmark immediately on arrival; apply ONNX quantization early if needed |
| Hardware shipping delays (3–4 weeks AliExpress) | Medium | Medium | Order immediately; use laptop for all CV module dev |
| UBFC-rPPG Kaggle mirror incomplete | Low-Medium | Medium | Verify subject count + ground truth files before building evaluation pipeline |
| Fewer than 30 volunteers available | Low | Medium | Recruit 40+; start outreach in Month 4 not Month 5 |
| Age MAE remains above 5 years | Low | Low | Apply face alignment preprocessing; or accept ±5 and justify in thesis |
| ESP32 firmware instability | Low | Medium | Use stable ESP-IDF examples as base; test BLE thoroughly before integration |

---

## 9. Accuracy & Performance Targets

### Accuracy by Module

| Vital Sign | Target MAE | Target % Within Range | Current Status |
|------------|-----------|----------------------|----------------|
| Age | ±4 years | 95% within range | ~5–6 yrs (close) |
| Heart Rate (rPPG) | ±5 BPM | 93% within range | Not started |
| Heart Rate (fused) | ±3 BPM | 95% within range | Not started |
| Temperature | ±1°C | 92% within range | Pending hardware |
| Respiratory Rate | ±2 BPM | 90% within range | Not started |
| Pupil Dilation | ±0.5mm | — | Not started |

### System Performance Targets

| Metric | Target |
|--------|--------|
| End-to-end scan latency | < 10 seconds |
| Real-time frame processing | 15+ fps |
| Model inference (age) | < 100ms per frame |
| Wearable battery life | > 8 hours |
| Aggregate accuracy (30 volunteers) | ≥ 90% |

### Model Comparison (For Thesis)

Both from-scratch and pretrained results must be logged to `model_performance` table and reported in Chapter 5. This is a core academic requirement of the project — not optional.

---

## 10. Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| Jan 2026 | Use Python 3.9.1 | MediaPipe compatibility |
| Jan 2026 | MySQL + Supabase backup | Local dev + cloud reliability |
| Jan 2026 | EfficientNet-B0 as best age model | 5.3M params, best MAE of three architectures |
| Jan 2026 | Huber loss over L1/MSE | Robust to age outliers in UTKFace |
| Jan 2026 | FastAPI over Flask | Better async support for WebSocket |
| Jun 2026 | Accept ~5–6 yr age MAE | Close to target; diminishing returns; rPPG is higher priority |
| Jun 2026 | Order hardware now | 3–4 week shipping to Turkey; must not block Phase 3 |

---

## Immediate Actions (This Week)

1. **Accept age model result** — freeze EfficientNet as the production age model
2. **Check UBFC-rPPG Kaggle dataset** — verify all 42 subjects + ground truth files present
3. **Place hardware order** — prioritize Pi 4B (4GB), ESP32-S3, MAX30102, MLX90614, ground truth oximeter
4. **Begin rPPG design** — map out exact forehead landmark indices from MediaPipe 478-point schema; sketch signal processing pipeline on paper before writing code
5. **Start volunteer list** — write down 40 people you can contact; share your project with classmates now

---

*This document should be updated at the end of each phase with actual results, decisions made, and any changes to the plan.*
