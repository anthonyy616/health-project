# Vital Signs Monitoring System using Computer Vision and NN's trained on Embedded Raspberry PI

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hybrid non-contact and wearable system for real-time vital signs monitoring in clinical settings. This final year project uses computer vision and machine learning to monitor temperature, blood pressure (estimated), pupil dilation, heart rate, respiratory rate, and patient age.

## Current Status

| Module | Status | Method |
|--------|--------|--------|
| Face Detection | Working | MediaPipe Face Landmarker (478 landmarks) |
| Age Estimation | Working | MobileNetV3-Small transfer learning (UTKFace) |
| Heart Rate (rPPG) | Working | POS algorithm + FFT on forehead ROI |
| Respiration Rate | Working | Chest ROI optical flow (Farneback) |
| Pupil Detection | Working | Iris-calibrated segmentation + EAR blink detection |
| Temperature | Pending | MLX90614 IR sensor (hardware required) |
| Wearable Band | Pending | ESP32-S3 + MAX30102 + MPU6050 (hardware required) |

**Test results:** 57 passed, 0 failed (as of 2026-08-20)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Tests

```bash
python -m pytest tests/ -v
```

### 3. Run Webcam Vitals Detection

```bash
# Basic usage (all modules)
python scripts/run_webcam_vitals.py

# Skip age estimation for faster FPS
python scripts/run_webcam_vitals.py --no-age

# Use thread pool for parallel module execution
python scripts/run_webcam_vitals.py --threading

# Save readings to auto-generated log file
python scripts/run_webcam_vitals.py --save-log

# Save readings to specific file
python scripts/run_webcam_vitals.py --save-log output/my_readings.json

# Use external camera
python scripts/run_webcam_vitals.py --camera 1
```

**Controls:**
- `q` -- Quit
- `r` -- Reset all detectors (clears accumulated buffers)

**Notes:**
- Heart rate and respiration need ~10-15 seconds of staring at the camera to fill their buffers before giving stable readings
- Ensure you are well-lit and facing the camera directly for best face detection
- The `--no-age` flag skips age estimation which is the heaviest module

## Project Structure

```
final_year_project/
├── src/contactless/              # Core detection modules
│   ├── face_detection/           # MediaPipe face detection + landmark extraction
│   ├── age_estimation/           # Age inference wrapper + dataset loader
│   ├── heart_rate/               # rPPG heart rate (POS algorithm + FFT)
│   ├── respiration/              # Chest ROI optical flow respiration
│   ├── pupil_detection/          # Iris-calibrated pupil dilation
│   ├── temperature/              # IR sensor integration (placeholder)
│   └── pipeline.py               # Unified vitals pipeline (VitalsPipeline)
├── models/
│   ├── age_detection/            # Model architectures (EfficientNet, MobileNet, LightAgeNet)
│   └── weights/age_detection/    # Trained weights (.pt, .onnx)
├── training/age_detection/       # Training script with balanced sampling
├── scripts/
│   ├── run_webcam_vitals.py      # Webcam vitals demo
│   ├── preprocessing/            # Dataset preprocessing
│   └── optimization/             # ONNX export + INT8 quantization
├── tests/                        # Unit tests (pytest)
├── test_logs/                    # Automated test result logs (JSON)
├── config.yaml                   # Project configuration
└── architecture.md               # Detailed architecture documentation
```

## Datasets

| Vital Sign | Dataset | Source |
|------------|---------|--------|
| Age | UTKFace | [Link](https://susanqq.github.io/UTKFace/) |
| Heart Rate | UBFC-rPPG | [Link](https://sites.google.com/view/yaboromance/ubfc-rppg) |
| Respiration | Custom | Collected during testing |
| Temperature | FLIR Thermal | [Link](https://www.flir.com/oem/adas/adas-dataset-form/) |

## Training the Age Model

```bash
# Quick test run (5 epochs)
python training/age_detection/train.py --quick

# Full training with EfficientNet-B0
python training/age_detection/train.py --model efficientnet --epochs 50

# Train with MobileNetV3 (currently in production)
python training/age_detection/train.py --model mobilenet --epochs 20

# Train without balanced sampling
python training/age_detection/train.py --no-balanced
```

**Note:** Requires preprocessed UTKFace dataset at `data/processed/utkface/`. Run preprocessing first:
```bash
python scripts/preprocessing/preprocess_utkface.py
```

## ONNX Optimization

```bash
# Export to ONNX FP32
python scripts/optimization/export_age_onnx.py

# Quantize to INT8
python scripts/optimization/quantize_age_model.py

# Benchmark inference speed
python scripts/optimization/benchmark_age_inference.py
```

**Finding:** Dynamic INT8 quantization is ~6.7x slower than FP32 on CPUs without VNNI instructions. The production path uses PyTorch FP32.

## Architecture

See [architecture.md](architecture.md) for detailed documentation of the entire codebase, including:
- Module-by-module breakdown with class/function descriptions
- Performance characteristics and bottleneck analysis
- Signal processing pipelines
- Model architectures and weight files
- Dataset details and preprocessing
- Testing strategy
- Future: wearable band BLE sync and backend streaming

## Target Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Age Estimation | +/-4 years | Working |
| Heart Rate | +/-5 BPM | Working (low confidence until dedicated rPPG training) |
| Respiration | +/-2 BPM | Working |
| Pupil Dilation | +/-0.1mm | Working |
| Temperature | +/-1 C | Pending hardware |
| Latency | <100ms | ~30-50ms per frame |
| Battery (Band) | >8 hours | Pending hardware |

## Author

[Anthony Ogbuah N.](https://anthonyy616.vercel.app/)

Final Year Project -- Solo Development

## License

This project is for educational purposes. Not certified for clinical use.
