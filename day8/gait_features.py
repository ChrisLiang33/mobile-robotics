"""
Shared by the laptop (record/train/live) and the UNO Q (runs the trained
models with plain numpy -- no scikit-learn on the board).

A window is WINDOW_S seconds of IMU samples: columns ax, ay, az (g) and
gx, gy, gz (deg/s). Two models:

  model 1  ONE number: the peak acceleration along one axis in the window
           (PEAK_AXIS, default az -- the vertical axis on an ankle). Standing
           is a flat ~1 g, walking peaks a little above it, running a lot,
           biking is in between but smooth. The trained model is just two or
           three thresholds on that number (a depth-limited decision tree).
  model 2  all six IMU outputs: for each axis mean, std, min, max, and for
           the acceleration magnitude its std and dominant frequency, plus
           gyro energy -- 29 numbers -> a decision tree.
"""

import numpy as np

SAMPLE_HZ = 50
WINDOW_S = 2.0
STRIDE_S = 0.5
AXES = ["ax", "ay", "az", "gx", "gy", "gz"]
PEAK_AXIS = "az"
CLASSES = ["standing", "walking", "running", "biking"]


def peak_feature(win, axis=PEAK_AXIS):
    """Model 1's single number: the largest |acceleration| on one axis."""
    col = AXES.index(axis)
    return float(np.max(np.abs(win[:, col])))


def full_features(win, hz=SAMPLE_HZ):
    """Model 2's feature vector from all six channels."""
    f = []
    for c in range(6):
        x = win[:, c]
        f += [float(x.mean()), float(x.std()), float(x.min()), float(x.max())]
    mag = np.linalg.norm(win[:, :3], axis=1)               # |accel|
    f += [float(mag.std()), float(mag.max() - mag.min())]
    spec = np.abs(np.fft.rfft(mag - mag.mean()))
    freqs = np.fft.rfftfreq(len(mag), 1.0 / hz)
    band = (freqs >= 0.5) & (freqs <= 6.0)                 # human movement lives here
    dom = float(freqs[band][np.argmax(spec[band])]) if band.any() and spec[band].max() > 0 else 0.0
    f += [dom, float(spec[band].max() / (len(mag) / 2)) if band.any() else 0.0]
    gyro = np.linalg.norm(win[:, 3:], axis=1)
    f += [float(gyro.mean())]
    return np.array(f)


FULL_NAMES = [f"{a}_{s}" for a in AXES for s in ("mean", "std", "min", "max")] + \
    ["accel_mag_std", "accel_mag_range", "dominant_freq_hz", "dominant_amp", "gyro_mean"]


def windows(samples, hz=SAMPLE_HZ, window_s=WINDOW_S, stride_s=STRIDE_S):
    """Slice an (N, 6) array into overlapping windows."""
    n, step = int(window_s * hz), int(stride_s * hz)
    return [samples[i:i + n] for i in range(0, len(samples) - n + 1, step)]


# ---- a decision tree exported from scikit-learn, evaluated with no sklearn ----
def tree_predict(node, x):
    while "class" not in node:
        node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
    return node["class"]


def export_tree(clf, class_names):
    """sklearn DecisionTreeClassifier -> nested dict of thresholds."""
    t = clf.tree_

    def rec(i):
        if t.children_left[i] == -1:
            return {"class": class_names[int(np.argmax(t.value[i]))], "samples": int(t.n_node_samples[i])}
        return {"feature": int(t.feature[i]), "threshold": float(t.threshold[i]),
                "left": rec(t.children_left[i]), "right": rec(t.children_right[i])}
    return rec(0)


class Models:
    """The two trained models, loaded from models.json."""

    def __init__(self, d):
        self.d = d
        self.peak_axis = d["model1"]["axis"]

    def predict1(self, win):
        return tree_predict(self.d["model1"]["tree"], [peak_feature(win, self.peak_axis)])

    def predict2(self, win):
        return tree_predict(self.d["model2"]["tree"], full_features(win))
