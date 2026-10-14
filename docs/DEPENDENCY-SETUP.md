# Dependency Setup

## Environment

This project uses a local virtual environment at `.venv`.

- Python: 3.11.9
- Install packages into the project venv, not the system Python:
  - `.venv/bin/pip install -r requirements.txt`
  - Or, for one package: `.venv/bin/pip install <package>`
- Verify the active environment with:
  - `python -c "import sys; print(sys.executable)"` should print a path inside `.venv`

Do not run `pip install` against the system interpreter. If a command is picking up a different Python/pip, fix the PATH/venv activation instead of reinstalling into system Python.

## Core dependencies that must be present locally

Before running the pipeline or tests, these must be importable from the project venv:

- numpy
- scipy
- opencv-python / opencv-python-headless
- mediapipe
- torch
- torchvision
- onnxruntime
- fastapi

A quick smoke check:

```bash
python -c "import numpy, scipy, cv2, mediapipe, torch, torchvision, onnxruntime, fastapi; print('deps OK')"
```

## MediaPipe note

MediaPipe is required by the face detection module (`src/contactless/face_detection/detect.py`).

- Confirmed working version in this project: `mediapipe 1.1.0`
- Install: `.venv/bin/pip install mediapipe`
- First run of `FaceDetector` will download the face landmarker model asset automatically to `src/contactless/face_detection/face_landmarker.task` if it is missing.
- If face detection fails on first use, check:
  - MediaPipe is installed in the same venv used to run the code
  - The machine has network access for the initial model download
  - `src/contactless/face_detection/face_landmarker.task` exists after first run

If MediaPipe is missing, face detection and anything downstream of it (age, rPPG HR, respiration, pupil, full pipeline mode) cannot work. Other modules that do not depend on MediaPipe may still be usable.

## Model artifacts

See `docs/MODEL-ARTIFACT-STATUS.md` for the current state of trained weights, checkpoints, and ONNX exports.

Quick summary:

- `models/weights/age_detection/best_model.pt` is the only currently valid age-inference artifact.
- Several other files in that directory are Git LFS pointer stubs or otherwise invalid for direct loading. Do not pass them to `torch.load(..., weights_only=True)` or `onnxruntime` expecting real weights.

## Dataset dependencies

Some tests and evaluation workflows require data that is not in the repository:

- Processed UTKFace data under `data/processed/utkface/` (train/val/test manifests + images)
- Raw UBFC-rPPG data under `data/ubfc-rppg/` for HR method comparison

Tests and evaluation scripts that depend on these must skip explicitly when the data is absent. They should not fabricate ground truth or pretend a run succeeded.

## First-time verification

After installing dependencies, run:

```bash
pytest tests/ -q
```

Expect:

- Core pipeline, HR, respiration, pupil, and evaluation-metric tests to pass
- Some tests to skip or fail only when the required dataset artifacts or model artifacts are missing

If everything imports but the suite still fails on missing manifests/weights, that is expected until the corresponding data/artifact dependencies are satisfied.
