import numpy as np
from scipy.signal import butter, filtfilt
from pathlib import Path
import sys
root = Path(".").resolve()
sys.path.insert(0, str(root))

# Exact DC normalization used by the evaluation CHROM/GREEN extractors
def dc_normalize(segment):
    means = np.mean(segment, axis=0)
    out = np.zeros_like(segment, dtype=np.float64)
    for c in range(3):
        if abs(means[c]) < 1e-6:
            out[:, c] = segment[:, c]
        else:
            out[:, c] = (segment[:, c] - means[c]) / means[c]
    return out

fps = 25.0
n = 250
f_hz = 1.0
amp_g = 0.04
amp_r = 0.02
amp_b = 0.01
t = np.arange(n) / fps
g = np.sin(2*np.pi*f_hz*t)*amp_g
r = np.sin(2*np.pi*f_hz*t)*amp_r
b = np.sin(2*np.pi*f_hz*t)*amp_b
trace = np.stack([r, g, b], axis=1)

# GREEN
green = dc_normalize(trace)[:, 1]
if len(green) < 15:
    filtered_g = green
else:
    nyq = fps/2.0
    bcoef, acoef = butter(4, [0.7/nyq, 4.0/nyq], btype="band")
    filtered_g = filtfilt(bcoef, acoef, green)

fft_g = np.abs(np.fft.rfft(filtered_g))
freqs_g = np.fft.rfftfreq(len(filtered_g), d=1.0/fps)
mask_g = (freqs_g >= 0.7) & (freqs_g <= 4.0)
vp_g = fft_g[mask_g]
vf_g = freqs_g[mask_g]
pk_g = int(np.argmax(vp_g))
bpm_g = vf_g[pk_g] * 60.0
conf_g = vp_g[pk_g] / np.sum(vp_g) if np.sum(vp_g) > 0 else 0.0

# CHROM
Xs = np.array([[1.0, -1.0, 0.0]])
Xd = np.array([[0.5, 0.5, -1.0]])
norm = dc_normalize(trace)
s = (norm @ Xs.T)[:, 0]
d = (norm @ Xd.T)[:, 0]
std_s = np.std(s)
std_d = np.std(d)
alpha = std_s / std_d if std_d > 1e-6 else 0.0
pulse_c = s - alpha * d
ps_c = np.std(pulse_c)
if ps_c > 1e-6:
    pulse_c = (pulse_c - np.mean(pulse_c)) / ps_c

if len(pulse_c) < 15:
    filtered_c = pulse_c
else:
    nyq = fps/2.0
    bcoef, acoef = butter(4, [0.7/nyq, 4.0/nyq], btype="band")
    filtered_c = filtfilt(bcoef, acoef, pulse_c)

fft_c = np.abs(np.fft.rfft(filtered_c))
freqs_c = np.fft.rfftfreq(len(filtered_c), d=1.0/fps)
mask_c = (freqs_c >= 0.7) & (freqs_c <= 4.0)
vp_c = fft_c[mask_c]
vf_c = freqs_c[mask_c]
pk_c = int(np.argmax(vp_c))
bpm_c = vf_c[pk_c] * 60.0
conf_c = vp_c[pk_c] / np.sum(vp_c) if np.sum(vp_c) > 0 else 0.0

print("GREEN balanced RFU-style trace")
print("bpm", round(float(bpm_g), 3))
print("conf", round(float(conf_g), 4))
print("CHROM balanced RFU-style trace")
print("bpm", round(float(bpm_c), 3))
print("conf", round(float(conf_c), 4))
print("GREEN finite:", bool(np.all(np.isfinite(filtered_g))))
print("CHROM finite:", bool(np.all(np.isfinite(filtered_c))))
print("GREEN pulse_std", float(np.std(green)))
print("CHROM pulse_std", float(np.std(pulse_c)))
