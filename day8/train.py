"""
Step 2 -- train the two gait models from data/gait.csv and export them.

    python train.py                    # -> models.json (+ copy into the UNO Q app), plots/

Model 1: the peak |acceleration| on ONE axis per 2 s window -> a tiny decision
         tree (depth 2 = up to three thresholds). One number in, class out.
Model 2: 29 features from all six IMU channels -> a decision tree (depth 4).

Both are tested with leave-one-session-out cross-validation (train on the
other recordings, test on each one in turn), which is the honest number
for "will it work on someone else". No neural nets anywhere.
"""

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree

from gait_features import (AXES, CLASSES, FULL_NAMES, PEAK_AXIS, SAMPLE_HZ, export_tree,
                           full_features, peak_feature, windows)

HERE = os.path.dirname(os.path.abspath(__file__))


def build_windows(df):
    X1, X2, y, groups = [], [], [], []
    for (session, label), g in df.groupby(["session", "label"], sort=False):
        arr = g[AXES].to_numpy(dtype=float)
        for win in windows(arr):
            X1.append([peak_feature(win)])
            X2.append(full_features(win))
            y.append(label)
            groups.append(session)
    return np.array(X1), np.array(X2), np.array(y), np.array(groups)


def loso(clf_factory, X, y, groups):
    """Leave-one-session-out accuracy and a pooled confusion matrix."""
    preds = np.empty_like(y)
    for s in np.unique(groups):
        tr, te = groups != s, groups == s
        if tr.sum() == 0:
            preds[te] = "?"
            continue
        clf = clf_factory().fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])
    acc = float((preds == y).mean())
    return acc, confusion_matrix(y, preds, labels=CLASSES), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(HERE, "data", "gait.csv"))
    ap.add_argument("--axis", default=PEAK_AXIS, choices=AXES, help="model 1's axis")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    print(f"{len(df)} samples, sessions: {sorted(df.session.unique())}")
    print("seconds per class:", {k: round(v / SAMPLE_HZ) for k, v in df.label.value_counts().items()})
    X1, X2, y, groups = build_windows(df)
    if args.axis != PEAK_AXIS:
        X1 = np.array([[peak_feature(w, args.axis)] for s, g in df.groupby(["session", "label"], sort=False)
                       for w in windows(g[AXES].to_numpy(dtype=float))])
    print(f"{len(y)} windows of 2 s")

    m1 = lambda: DecisionTreeClassifier(max_depth=2, min_samples_leaf=5, random_state=0)
    m2 = lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=0)
    multi = len(np.unique(groups)) > 1
    if multi:
        acc1, cm1, _ = loso(m1, X1, y, groups)
        acc2, cm2, _ = loso(m2, X2, y, groups)
        how = "leave-one-session-out"
    else:
        from sklearn.model_selection import cross_val_predict, StratifiedKFold
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        p1 = cross_val_predict(m1(), X1, y, cv=cv); p2 = cross_val_predict(m2(), X2, y, cv=cv)
        acc1, acc2 = float((p1 == y).mean()), float((p2 == y).mean())
        cm1, cm2 = confusion_matrix(y, p1, labels=CLASSES), confusion_matrix(y, p2, labels=CLASSES)
        how = "5-fold (only one session so far; record a second one for an honest test)"
    print(f"\n=== accuracy ({how})")
    print(f"model 1 (peak |{args.axis}| only): {acc1:.1%}")
    print(f"model 2 (all IMU features):   {acc2:.1%}")
    for name, cm in (("model 1", cm1), ("model 2", cm2)):
        print(f"\n{name} confusion (rows = truth, cols = predicted; order {CLASSES}):")
        print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

    # final models on everything
    clf1, clf2 = m1().fit(X1, y), m2().fit(X2, y)
    models = {"model1": {"axis": args.axis, "tree": export_tree(clf1, list(clf1.classes_)), "accuracy": acc1},
              "model2": {"features": FULL_NAMES, "tree": export_tree(clf2, list(clf2.classes_)), "accuracy": acc2},
              "classes": CLASSES, "window_s": 2.0, "sample_hz": SAMPLE_HZ, "cv": how}
    out = os.path.join(HERE, "models.json")
    json.dump(models, open(out, "w"), indent=1)
    shutil.copy(out, os.path.join(HERE, "unoq", "gait-wearable", "python", "models.json"))
    print(f"\nwrote {out} and a copy into the UNO Q app")

    # plots for the Notion page
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(X1.min(), X1.max(), 40)
    for c in CLASSES:
        ax.hist(X1[y == c, 0], bins=bins, alpha=0.6, label=c)
    def thresholds(node, acc):
        if "threshold" in node:
            acc.append(node["threshold"]); thresholds(node["left"], acc); thresholds(node["right"], acc)
        return acc
    for t in thresholds(models["model1"]["tree"], []):
        ax.axvline(t, color="k", ls="--")
    ax.set_xlabel(f"peak |{args.axis}| in a 2 s window (g)"); ax.set_ylabel("windows")
    ax.set_title(f"Model 1: one number and {len(thresholds(models['model1']['tree'], []))} threshold(s)  -  accuracy {acc1:.0%}")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(HERE, "plots", "model1_thresholds.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(clf2, feature_names=FULL_NAMES, class_names=list(clf2.classes_), filled=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title(f"Model 2: decision tree on all IMU features  -  accuracy {acc2:.0%}")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "plots", "model2_tree.png"), dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for a, (name, cm, acc) in zip(axes, (("Model 1", cm1, acc1), ("Model 2", cm2, acc2))):
        a.imshow(cm, cmap="Blues"); a.set_xticks(range(4)); a.set_yticks(range(4))
        a.set_xticklabels(CLASSES, rotation=30); a.set_yticklabels(CLASSES)
        for i in range(4):
            for j in range(4):
                a.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        a.set_title(f"{name}: {acc:.0%}"); a.set_xlabel("predicted"); a.set_ylabel("truth")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "plots", "confusion.png"), dpi=150)
    print("plots -> plots/model1_thresholds.png, plots/model2_tree.png, plots/confusion.png")


if __name__ == "__main__":
    main()
