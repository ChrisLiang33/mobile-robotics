"""
K-nearest-neighbors classifier for the floor sensor (a second algorithm from
the week's list). Each sample is the Color Sensor's reading turned into a
feature vector; the label is "floor" or "tape". To classify a new reading,
find the K stored samples closest to it and take a vote.
"""

import json
import os

import numpy as np


def features(raw_rgb, reflection):
    """Normalized color (independent of brightness) plus reflection."""
    r, g, b = (float(v) for v in raw_rgb)
    total = r + g + b + 1e-9
    return np.array([r / total, g / total, b / total, float(reflection) / 100.0])


class KNN:
    def __init__(self, k=5, path=None):
        self.k, self.path = k, path
        self.X, self.y = [], []
        if path and os.path.exists(path):
            d = json.load(open(path))
            self.X, self.y = [np.array(x) for x in d["X"]], list(d["y"])

    def add(self, x, label):
        self.X.append(np.asarray(x, dtype=float))
        self.y.append(label)
        if self.path:
            json.dump({"X": [list(map(float, x)) for x in self.X], "y": self.y}, open(self.path, "w"))

    def counts(self):
        return {lab: self.y.count(lab) for lab in set(self.y)}

    def ready(self):
        return len(set(self.y)) >= 2 and len(self.y) >= self.k

    def predict(self, x):
        """Returns (label, confidence) -- confidence = share of the K votes."""
        if not self.ready():
            return None, 0.0
        X = np.stack(self.X)
        d = np.linalg.norm(X - np.asarray(x, dtype=float), axis=1)
        idx = np.argsort(d)[: self.k]
        votes = {}
        for i in idx:
            votes[self.y[i]] = votes.get(self.y[i], 0) + 1
        label = max(votes, key=votes.get)
        return label, votes[label] / self.k
