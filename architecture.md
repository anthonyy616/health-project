# Architecture Documentation

> Private reference for the Vital Signs Monitoring System codebase.
> Last updated: 2026-08-20

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Module Breakdown](#module-breakdown)
3. [Signal Processing Pipelines](#signal-processing-pipelines)
4. [Model Architectures](#model-architectures)
5. [Datasets and Preprocessing](#datasets-and-preprocessing)
6. [Performance Analysis](#performance-analysis)
7. [Testing Strategy](#testing-strategy)
8. [Configuration](#configuration)
9. [Future: Wearable Band Sync](#future-wearable-band-sync)
10. [Future: Backend Streaming](#future-backend-streaming)

---

## System Overview

The system is a contactless vital signs monitor that reads heart rate, respiratory rate, pupil dilation, and apparent age from a standard webcam feed. It is designed for clinical settings where non-contact measurement is preferred.

### Data Flow

```
Webcam (30fps)
    |
    v
FaceDetector.detect(frame)  -->  FaceResult
    |
    +---> HeartRateDetector.add_frame(face_result)        -->  HeartRateResult
    +---> RespirationDetector.add_frame(frame, face_result) -->  RespirationResult
    +---> PupilDetector.detect(frame, face_result)          -->  PupilResult
    +---> AgeEstimator.estimate(face_roi)                   -->  AgeEstimationResult
    |
    v
VitalsReading (unified output)
```

### Entry Points

| Script | Purpose |
|--------|---------|
| `scripts/run_webcam_vitals.py` | Live webcam demo with OpenCV overlay |
| `python -m pytest tests/ -v` | Run full test suite |
| `training/age_detection/train.py` | Train age estimation model |

---

## Module Breakdown

### 1. Face Detection (`src/contactless/face_detection/`)

**File:** `detect.py` — `FaceDetector` class

**Method:** MediaPipe Face Landmarker (Tasks API, 0.10.x+)

**What it does:**
- Detects a single face per frame
- Extracts 478 3D landmarks (x, y, z) in pixel coordinates
- Computes bounding box from face oval landmarks (indices 10, 338, 297, ...)
- Extracts ROIs for downstream modules:
  - `face_roi`: 224x224 crop (for age estimation)
  - `forehead_roi`: 64x32 crop (for rPPG)
  - `left_eye_roi` / `right_eye_roi`: 64x32 crops (for pupil detection)

**Performance:** ~15-25ms per frame on CPU

**Key constants:**
- `FOREHEAD_LANDMARKS = [10, 67, 69, 104, 108, 151, 299, 337, 338, 297, 332, 284]`
- `LEFT_EYE_LANDMARKS = [33, 133, 160, 159, 158, 144, 145, 153, 154, 155, 157, 163]`
- `RIGHT_EYE_LANDMARKS = [362, 263, 387, 386, 385, 373, 374, 380, 381, 382, 384, 390]`

**Dependencies:** `mediapipe>=0.10.8`, `opencv-python>=4.8.0`

---

### 2. Heart Rate Detection (`src/contactless/heart_rate/`)

**Files:** `rppg.py` (stateful class), `signal_processing.py` (pure functions)

**Method:** Remote Photoplethysmography (rPPG) using the POS algorithm

**How it works:**
1. Each frame, the mean RGB values of the forehead ROI are appended to a rolling buffer (deque, max 300 frames = 10 seconds at 30fps)
2. When the buffer has >= 150 frames (5 seconds), signal processing begins:
   - `normalize_rgb_trace()`: Remove DC offset per channel
   - `apply_pos_algorithm()`: Project RGB to 2D plane using POS matrix `[[0,1,-1],[-2,1,1]]`
   - `bandpass_filter()`: 4th-order Butterworth, 0.7-4.0 Hz (42-240 BPM)
   - `extract_bpm_from_fft()`: FFT peak detection in the valid frequency band
3. Confidence is the ratio of peak power to total power in the band

**Critical constraint:** Frames must NOT be skipped — the Butterworth filter cutoffs and FFT timing math assume the actual sample rate matches the `fps` parameter. Skipping frames would alias the BPM.

**Performance:** ~1-2ms per frame (signal processing only, after buffer fills)

**Buffer parameters:**
- `fps=30`, `window_seconds=10` → buffer_size=300
- `min_seconds=5.0` → min_buffer_size=150
- `quality_threshold=0.1` → minimum green-channel CV to attempt BPM

---

### 3. Respiration Detection (`src/contactless/respiration/`)

**Files:** `detect.py` (stateful class), `motion_analysis.py` (pure functions)

**Method:** Dense optical flow on chest ROI

**How it works:**
1. Derives a chest bounding box below the face bbox (width=1.2x face, height=1.5x face, gap=0.15x face)
2. Crops and converts to grayscale, resizes to 128x128
3. Computes vertical component of Farneback optical flow between consecutive frames
4. Appends mean vertical flow value to a rolling buffer (deque, max 600 frames = 20 seconds at 30fps)
5. When buffer has >= 300 frames (10 seconds):
   - `detrend_signal()`: Remove linear trend (posture drift)
   - `bandpass_filter_respiration()`: 3rd-order Butterworth, 0.1-0.5 Hz (6-30 BPM)
   - `detect_breathing_peaks()`: Peak detection with prominence threshold
   - `compute_bpm_from_peaks()`: Regularity-based confidence from peak spacing

**Critical constraint:** Same as heart rate — frames must not be skipped.

**Performance:** ~5-8ms per frame (optical flow is the bottleneck)

**Key constants:**
- `RESP_MIN_HZ = 0.1` (6 BPM), `RESP_MAX_HZ = 0.5` (30 BPM)
- `FARNEBACK_PARAMS = (0.5, 3, 15, 3, 5, 1.2, 0)`
- `MIN_MOTION_STD = 0.001`, `MAX_MOTION_STD = 0.5`

---

### 4. Pupil Detection (`src/contactless/pupil_detection/`)

**Files:** `detect.py` (stateful class), `iris_tracker.py` (pure functions)

**Method:** Iris-calibrated pupil segmentation + EAR blink detection

**How it works:**
1. **Blink detection:** Computes average Eye Aspect Ratio (EAR) across both eyes. If EAR < 0.2, returns cached result (marked as blinking) to avoid display flashing.
2. **Iris ring measurement:** Uses MediaPipe iris landmarks (468-477) to compute iris diameter in pixels, then converts to px/mm scale using the known anatomical iris diameter (11.7mm average).
3. **Pupil segmentation:** Crops a square region around the iris center, applies CLAHE lighting compensation, Otsu thresholding, and contour detection to find the pupil (darkest circular region).
4. **Dilation conversion:** `pupil_diameter_mm = (pupil_radius_px * 2) / px_per_mm`
5. **EMA smoothing:** Per-eye exponential moving average (alpha=0.3) to smooth frame-to-frame jitter
6. **Baseline tracking:** First 30 valid readings establish a baseline; subsequent readings report signed delta from baseline

**Left/Right swap note:** MediaPipe uses subject-perspective naming (RIGHT_IRIS = 468-472 = image-left eye). The code accounts for this in `PupilDetector.detect()`.

**Performance:** ~3-5ms per frame

**Key constants:**
- `AVERAGE_IRIS_DIAMETER_MM = 11.7`
- `MIN_PLAUSIBLE_PUPIL_RATIO = 0.2`, `MAX_PLAUSIBLE_PUPIL_RATIO = 0.9`
- `CIRCULARITY_THRESHOLD = 0.6`

---

### 5. Age Estimation (`src/contactless/age_estimation/`)

**Files:** `estimator.py` (inference wrapper), `processed_dataset.py` (dataset loader)

**Method:** MobileNetV3-Small transfer learning on UTKFace

**AgeEstimator class:**
- Default `model_type="mobilenet"` (matches `best_model.pt` on disk)
- Preprocesses face ROI: BGR→RGB, resize 224x224, normalize to [0,1], HWC→CHW
- Forward pass through MobileNetV3Age, clamp to [0, 100], round to int
- Confidence estimation: 0.85 base, reduced for extreme ages or unloaded model

**ProcessedUTKFaceDataset:**
- Loads from `data/processed/utkface/` with manifest JSON files
- Supports train/val/test splits, augmentation (flip, brightness, contrast, rotation)
- Returns `(image_tensor, age_int)` tuples

**Performance:** ~20-30ms per face (single inference)

---

### 6. Unified Pipeline (`src/contactless/pipeline.py`)

**Class:** `VitalsPipeline`

**What it does:**
- Wraps all 5 modules behind a single `process_frame(frame)` call
- Handles frame skipping (age every 5th frame, pupil every frame)
- HR and respiration always run (critical: skipping would alias their math)
- Caches last good age/pupil results on skip frames
- Optional ThreadPoolExecutor for parallel module execution
- Module errors are collected, never crash the pipeline

**Key method:** `_build_reading()` assembles a `VitalsReading` dataclass from whichever sub-results succeeded.

---

## Signal Processing Pipelines

### rPPG (Heart Rate)

```
Forehead ROI (64x32 BGR)
    → cv2.mean() → (R, G, B) per frame
    → Rolling buffer (300 frames)
    → normalize_rgb_trace()     [DC removal + channel scaling]
    → POS algorithm             [RGB → 2D projection → pulse signal]
    → bandpass_filter()         [4th-order Butterworth, 0.7-4.0 Hz]
    → extract_bpm_from_fft()    [FFT peak detection → BPM + confidence]
```

### Respiration

```
Chest ROI (128x128 grayscale)
    → Farneback optical flow (consecutive frames)
    → mean vertical flow → scalar per frame
    → Rolling buffer (600 frames)
    → detrend_signal()                    [linear trend removal]
    → bandpass_filter_respiration()       [3rd-order Butterworth, 0.1-0.5 Hz]
    → detect_breathing_peaks()            [peak detection with prominence]
    → compute_bpm_from_peaks()            [interval → BPM + regularity confidence]
```

### Pupil Dilation

```
Full frame + FaceResult (478 landmarks)
    → EAR (Eye Aspect Ratio) from eye contours
    → Blink detection (EAR < 0.2)
    → Iris ring diameter (landmarks 468-477) → px/mm scale
    → Square crop around iris center (64x64)
    → CLAHE → Otsu threshold → contour detection
    → Pupil radius (pixels) → diameter (mm)
    → EMA smoothing (alpha=0.3) → baseline tracking
```

---

## Model Architectures

### Production Model: MobileNetV3-Small (`mobilenet_age.py`)

| Property | Value |
|----------|-------|
| Architecture | MobileNetV3-Small (torchvision) |
| Backbone | Pretrained on ImageNet |
| Classifier head | Linear(576→256) → Hardswish → Dropout(0.2) → Linear(256→64) → ReLU → Dropout(0.1) → Linear(64→1) |
| Parameters | ~2.5M |
| Input | (B, 3, 224, 224) float [0, 1] |
| Output | (B,) float (predicted age) |
| Weight file | `models/weights/age_detection/best_model.pt` |

### Alternative: EfficientNet-B0 (`efficientnet_age.py`)

| Property | Value |
|----------|-------|
| Architecture | EfficientNet-B0 (torchvision) |
| Classifier head | Dropout(0.3) → Linear(1280→512) → ReLU → Dropout(0.15) → Linear(512→128) → ReLU → Linear(128→1) |
| Parameters | ~5.3M |
| Weight file | `age_model_fp32.onnx`, `age_model_int8.onnx` |

**Note:** The on-disk `best_model.pt` contains MobileNetV3 weights, not EfficientNet. The `AgeEstimator` default is `model_type="mobilenet"` to match.

### Lightweight: LightAgeNet (`light_age_net.py`)

Custom 5-block CNN (~500K params). Used for experiments, not production.

---

## Weight Files

| File | Description |
|------|-------------|
| `best_model.pt` | Production weights (MobileNetV3, ~2.5M params) |
| `best_checkpoint.pt` | Full checkpoint with optimizer state |
| `checkpoint_epoch_10.pt` | Epoch 10 checkpoint |
| `checkpoint_epoch_20.pt` | Epoch 20 checkpoint |
| `age_model_fp32.onnx` | ONNX FP32 export (EfficientNet) |
| `age_model_int8.onnx` | ONNX INT8 quantized (slower on non-VNNI CPUs) |
| `age_model_20260103_*.pt` | Timestamped training runs |

---

## Datasets and Preprocessing

### UTKFace (Age Estimation)

- **Source:** ~23K face images, labeled with age (0-116), gender, race
- **Preprocessing:** `scripts/preprocessing/preprocess_utkface.py`
  - Resizes to 224x224
  - Splits into train (46K), val (10K), test (10K)
  - Creates manifest JSON files per split
  - Expected at `data/processed/utkface/`
- **Augmentation:** Horizontal flip, brightness/contrast jitter, rotation (±10°)
- **Balanced sampling:** `WeightedRandomSampler` with inverse-frequency age bin weights

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=0.001, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Loss | HuberLoss (delta=5.0) |
| Early stopping | Patience=10 epochs |
| Batch size | 32 |
| Device | CPU only |

---

## Performance Analysis

### Per-Frame Latency (approximate, on CPU)

| Module | Latency | Bottleneck |
|--------|---------|------------|
| Face Detection | 15-25ms | MediaPipe inference |
| Heart Rate | 1-2ms | FFT on 300 samples |
| Respiration | 5-8ms | Farneback optical flow |
| Pupil Detection | 3-5ms | Contour detection + CLAHE |
| Age Estimation | 20-30ms | MobileNetV3 forward pass |
| **Total (sequential)** | **~50-70ms** | Face detection + age |
| **Total (threaded)** | **~30-40ms** | Parallel module execution |

### Known Performance Issues

1. **HR/Resp confidence is low** because:
   - No dedicated rPPG training dataset (UBFC-rPPG not yet integrated)
   - Signal quality depends heavily on lighting and forehead visibility
   - The POS algorithm is a general-purpose approach; training-specific weights would improve accuracy

2. **INT8 quantization is 6.7x slower** on CPUs without VNNI instructions (this machine). FP32 is the production path.

3. **Age estimation confidence** is 0.85 base with adjustments for extreme ages. No per-sample calibration.

### Confidence Gating

All modules return `None` for their primary reading (BPM, diameter, age) when confidence is below threshold:
- HR: `signal_quality < 0.1` → returns `bpm=None`
- Respiration: `signal_quality < 0.1` → returns `bpm=None`
- Pupil: `is_blinking=True` → returns cached result, or `average_mm=None`
- Age: always returns a value (random weights when model not loaded)

---

## Testing Strategy

### Test Files

| File | Tests | Coverage |
|------|-------|----------|
| `test_age_dataset.py` | 7 | Dataset loading, statistics, splits, augmentation config |
| `test_age_model.py` | 8 | Model architecture, estimator init/preprocess/inference |
| `test_age_onnx.py` | 3 | ONNX export match, INT8 shape, INT8 speed |
| `test_pipeline.py` | 5 | Pipeline construction, frame processing, error isolation, reset |
| `test_rppg.py` | 8 | POS algorithm, bandpass, BPM extraction, detector lifecycle |
| `test_respiration.py` | 11 | Chest bbox, optical flow, detrend, bandpass, peaks, detector |
| `test_pupil_detection.py` | 9 | Iris diameter, EAR, blink, pupil segmentation, dilation mm |

**Total:** 57 tests, all passing as of 2026-08-20

### Test Logging

`conftest.py` at project root hooks into pytest to write JSON logs to `test_logs/`:
- Filename: `test_results_YYYY-MM-DD_HH-MM-SS.json`
- Includes features tested, per-test outcomes, durations, failure info
- Format: JSON (cheap for AI agents to parse)

---

## Configuration

**File:** `config.yaml`

Key settings:
- `camera.fps: 30` — Must match the `fps` parameter passed to `VitalsPipeline`
- `vitals.heart_rate.buffer_seconds: 10` — Rolling window for HR
- `vitals.respiration.buffer_seconds: 20` — Rolling window for respiration
- `training.batch_size: 16` — Smaller for CPU training
- `wearable.enabled: false` — Wearable band not yet implemented
- `database.type: mysql` — For future data persistence

---

## Future: Wearable Band Sync

### Planned Hardware

- **MCU:** ESP32-S3 (dual-core, BLE 5.0, Wi-Fi)
- **Sensors:**
  - MAX30102 pulse oximeter (SpO2 + heart rate)
  - MPU6050 IMU (6-axis accelerometer + gyroscope for respiration via chest motion)
- **Communication:** BLE GATT service with custom UUIDs

### BLE Service Architecture

```
Wearable Band (ESP32-S3)
    │
    │  BLE GATT
    │  ├── Service: Vitals Service (UUID: 12345678-1234-5678-1234-56789abcdef0)
    │  │   ├── Characteristic: Heart Rate (UUID: ...0001) - Notify
    │  │   ├── Characteristic: SpO2 (UUID: ...0002) - Notify
    │  │   ├── Characteristic: Respiration (UUID: ...0003) - Notify
    │  │   └── Characteristic: IMU Raw (UUID: ...0004) - Notify
    │  └── CCCD: Client enable/disable notifications
    │
    v
Raspberry Pi Central Hub
    │
    ├── BLE Scanner (bleak library)
    │   └── Connects to band, subscribes to notifications
    │
    ├── Data Fusion
    │   ├── CV vitals (from webcam pipeline)
    │   ├── Wearable vitals (from BLE notifications)
    │   └── Kalman filter or weighted average for final reading
    │
    v
Dashboard / Database
```

### Sync Protocol (Proposed)

1. **Band-side:** ESP32 reads sensors at 25Hz, packages into BLE notification packets (8 bytes each: 2-byte heart rate, 1-byte SpO2, 2-byte respiration, 3-byte IMU)
2. **Hub-side:** Python BLE client (`bleak`) receives packets, unpacks, timestamps
3. **Fusion:** When both CV and wearable readings are available for the same vital, use a weighted average based on confidence scores
4. **Display:** Real-time overlay on the webcam feed shows fused readings

---

## Future: Backend Streaming

### Planned Architecture

```
VitalsPipeline
    │
    ├── WebSocket (FastAPI)
    │   ├── /ws/vitals  →  Real-time JSON stream to dashboard
    │   └── /ws/wearable  →  BLE packet stream from hub
    │
    ├── REST API (FastAPI)
    │   ├── POST /api/readings  →  Save reading to database
    │   ├── GET /api/readings  →  Query historical readings
    │   └── GET /api/patient/{id}  →  Patient vitals history
    │
    └── Database (MySQL)
        ├── readings table: timestamp, patient_id, hr, rr, pupil, age
        └── sessions table: start_time, end_time, device_info
```

### Data Format for Streaming

```json
{
  "timestamp": "2026-08-20T14:30:00Z",
  "source": "cv" | "wearable" | "fused",
  "heart_rate_bpm": 72.5,
  "hr_confidence": 0.85,
  "respiratory_rate_bpm": 16.0,
  "resp_confidence": 0.70,
  "pupil_diameter_mm": 3.42,
  "pupil_confidence": 0.90,
  "age": 25,
  "age_confidence": 0.85,
  "spo2": 98,
  "session_id": "uuid"
}
```

### Wearable Band Integration Points

1. **Config:** `wearable.enabled: true` in `config.yaml` enables BLE scanning
2. **BLE client:** New module `src/wearable/ble_client.py` using `bleak`
3. **Data fusion:** New module `src/wearable/fusion.py` for CV + wearable reading combination
4. **Backend:** FastAPI app in `src/central/api.py` (currently placeholder)

---

*This document is a living reference. Update as new modules are added or existing ones change.*
