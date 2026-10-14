import numpy as np
from pathlib import Path
import sys
from importlib import import_module

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import matplotlib
matplotlib.use("Agg")

from src.contactless.heart_rate.signal_processing import (
    normalize_rgb_trace,
    apply_pos_algorithm,
    bandpass_filter,
    extract_bpm_from_fft,
)

f_hz = 1.0
amp_g = 0.04
amp_r = 0.02
amp_b = 0.01
n = 250
fps = 25.0
t = np.arange(n) / fps
g = np.sin(2 * np.pi * f_hz * t) * amp_g
r = np.sin(2 * np.pi * f_hz * t) * amp_r
b = np.sin(2 * np.pi * f_hz * t) * amp_b
trace = np.stack([r, g, b], axis=1)

normalized = normalize_rgb_trace(trace)
pulse = apply_pos_algorithm(normalized)
filtered = bandpass_filter(pulse, fps)
bpm, conf = extract_bpm_from_fft(filtered, fps)
print("POS balanced RFU-style trace")
print("bpm", round(float(bpm), 3))
print("conf", round(float(conf), 4))
print("pulse_std", float(np.std(pulse)))
print("normalized is finite:", bool(np.all(np.isfinite(normalized))))
print("pulse is finite:", bool(np.all(np.isfinite(pulse))))
print("filtered is finite:", bool(np.all(np.isfinite(filtered))))
