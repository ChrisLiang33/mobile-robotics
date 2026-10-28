"""
Shared by the laptop (record/train/live) and the UNO Q (runs the trained
models with plain numpy -- no scikit-learn on the board).

A window is WINDOW_S seconds of IMU samples: columns ax, ay, az (g) and
gx, gy, gz (deg/s). Two models:

  model 1  ONE number: the peak acceleration along one axis in the window
           (PEAK_AXIS, default ax -- best of the three accel axes across sessions). Standing
           is a flat ~1 g, walking peaks a little above it, running a lot,
           biking is in between but smooth. The trained model is just two or
           three thresholds on that number (a depth-limited decision tree).
  model 2  all six IMU outputs, combined into the accel and gyro
           magnitudes (mean, std, range of each) plus the step frequency
           and its strength -- 8 numbers -> a decision tree.
"""

import numpy as np

SAMPLE_HZ = 50
WINDOW_S = 2.0
STRIDE_S = 0.5
AXES = ["ax", "ay", "az", "gx", "gy", "gz"]
PEAK_AXIS = "ax"
CLASSES = ["standing", "walking", "running", "biking"]


def peak_feature(win, axis=PEAK_AXIS):
    """Model 1's single number: the largest acceleration on one axis,
    measured from the window's mean so gravity (which depends on how the
    strap sits) drops out -- only the motion's own peak is left."""
    x = win[:, AXES.index(axis)]
    return float(np.max(np.abs(x - x.mean())))


def full_features(win, hz=SAMPLE_HZ):
    """Model 2's feature vector from all six channels.

    Built on the MAGNITUDES of the accel and gyro vectors, not the single
    axes, so it doesn't care how the strap sits on the ankle -- per-axis
    features scored 98% within one recording but 70% on a new one,
    because the board was turned a bit differently each time.
    """
    f = []
    for c in (slice(0, 3), slice(3, 6)):                   # |accel|, |gyro|
        m = np.linalg.norm(win[:, c], axis=1)
        f += [float(m.mean()), float(m.std()), float(m.max() - m.min())]
    mag = np.linalg.norm(win[:, :3], axis=1)
    spec = np.abs(np.fft.rfft(mag - mag.mean()))
    freqs = np.fft.rfftfreq(len(mag), 1.0 / hz)
    band = (freqs >= 0.5) & (freqs <= 6.0)                 # human movement lives here
    dom = float(freqs[band][np.argmax(spec[band])]) if band.any() and spec[band].max() > 0 else 0.0
    f += [dom, float(spec[band].max() / (len(mag) / 2)) if band.any() else 0.0]
    return np.array(f)


FULL_NAMES = ["accel_mag_mean", "accel_mag_std", "accel_mag_range",
              "gyro_mag_mean", "gyro_mag_std", "gyro_mag_range",
              "step_freq_hz", "step_amp"]


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
