# Vital Signs Monitoring System using Computer Vision and NN's trained on Embedded Raspberry PI

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hybrid non-contact and wearable system for real-time vital signs monitoring in clinical settings. This final year project uses computer vision and machine learning to monitor temperature, blood pressure (estimated), pupil dilation, heart rate, respiratory rate, and patient age.

## 🎯 Features

### Contactless Station (Computer Vision)
- **Face Detection** - MediaPipe-based face detection and landmarks
- **Age Estimation** - Deep learning model for apparent age prediction
- **Heart Rate (rPPG)** - Remote photoplethysmography from facial color changes
- **Respiratory Rate** - Chest movement detection
- **Pupil Detection** - Eye tracking and dilation measurement
- **Temperature** - Infrared sensor integration (MLX90614)

### Wearable Band (Future)
- Pulse oximeter (MAX30102) for continuous heart rate
- IMU (MPU6050) for respiration detection
- ESP32-S3 with BLE communication
- Edge ML inference with TensorFlow Lite

## 📁 Project Structure

```
final_year_project/
├── docs/                       # Documentation
├── src/                        # Source code
│   ├── contactless/            # CV-based vital detection
│   ├── wearable/               # ESP32 band code
│   ├── central/                # Raspberry Pi hub & dashboard
│   └── utils/                  # Shared utilities
├── models/                     # ML model weights
├── data/                       # Datasets (gitignored)
├── notebooks/                  # Jupyter experiments
├── tests/                      # Unit tests
├── hardware/                   # Schematics
└── scripts/                    # Utility scripts
```



### Prerequisites
- Python 3.11+
- Webcam (laptop built-in or USB)
- MySQL Server (optional, for data storage)


## 📊 Datasets

| Vital Sign | Dataset | Source |
|------------|---------|--------|
| Age | UTKFace | [Link](https://susanqq.github.io/UTKFace/) |
| Heart Rate | UBFC-rPPG | [Link](https://sites.google.com/view/yaboromance/ubfc-rppg) |
| Respiration | Custom | Collected during testing |
| Temperature | FLIR Thermal | [Link](https://www.flir.com/oem/adas/adas-dataset-form/) |



## 📈 Target Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Age Estimation | ±4 years | 🔄 In Progress |
| Heart Rate | ±5 BPM | 🔄 In Progress |
| Temperature | ±1°C | ⏳ Pending Hardware |
| Latency | <10 seconds | 🔄 In Progress |
| Battery (Band) | >8 hours | ⏳ Pending Hardware |

## 🗓️ Timeline (8-10 Weeks)

- **Weeks 1-2**: Face detection, age estimation
- **Weeks 3-4**: rPPG heart rate, respiratory rate
- **Weeks 5-6**: Pupil detection, model optimization
- **Weeks 7-8**: Dashboard, data fusion
- **Weeks 9-10**: Testing with volunteers, documentation

## 👤 Author[Link:[anthony](https://anthonyy616.vercel.app/)]

Final Year Project - Solo Development

## 📄 License

This project is for educational purposes. Not certified for clinical use.
