"""
Step 1 -- record labeled IMU data from the wearable (over MQTT) into a CSV.

    python record.py                 # appends to data/gait.csv
    python record.py --session chris_ankle_1

A window shows the live signal. Press the key for what you are doing RIGHT
NOW and keep doing it; every sample is saved with that label until you press
another key.  s standing   w walking   r running   b biking   SPACE unlabeled
(discarded)   q quit.  Aim for ~60 s of each activity, in a couple of sessions.
"""

import argparse
import csv
import json
import os
import time
from collections import deque

import cv2
import numpy as np

from mqttlib import MQTTClient

PLAYER = "ChrisLiang"
RAW_TOPIC = f"ME193/{PLAYER}/gait/imu"
HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = {"s": "standing", "w": "walking", "r": "running", "b": "biking"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=time.strftime("session_%m%d_%H%M"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "gait.csv"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    new = not os.path.exists(args.out)
    f = open(args.out, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["session", "label", "t", "ax", "ay", "az", "gx", "gy", "gz"])

    label = [None]
    trace = deque(maxlen=400)
    counts = {k: 0 for k in KEYS.values()}
    last_seen = [0.0]

    def on_raw(topic, payload):
        try:
            d = json.loads(payload)
        except ValueError:
            return
        last_seen[0] = time.monotonic()
        for s in d.get("s", []):
            trace.append(s[1:7])
            if label[0]:
                w.writerow([args.session, label[0]] + s)
                counts[label[0]] += 1

    c = MQTTClient()
    c.connect()
    c.subscribe(RAW_TOPIC, on_raw)
    print(f"listening on {RAW_TOPIC}; writing {args.out}")
    try:
        while True:
            p = np.full((420, 820, 3), 24, np.uint8)
            online = time.monotonic() - last_seen[0] < 1.5
            cv2.putText(p, f"wearable: {'ONLINE' if online else 'no data'}   labeling: {label[0] or '-- (press s/w/r/b)'}",
                        (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 0) if online else (0, 0, 230), 2, cv2.LINE_AA)
            cv2.putText(p, "   ".join(f"{k}: {v / 50:.0f} s" for k, v in counts.items()), (16, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            if len(trace) > 2:
                arr = np.array(trace)
                for i, (name, col) in enumerate(zip(["ax", "ay", "az"], [(0, 0, 255), (0, 255, 0), (255, 128, 0)])):
                    ys = 230 - arr[:, i] * 40
                    pts = np.stack([np.linspace(16, 800, len(ys)), ys], 1).astype(np.int32)
                    cv2.polylines(p, [pts], False, col, 1)
                    cv2.putText(p, name, (16 + 40 * i, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
                for i, (name, col) in enumerate(zip(["gx", "gy", "gz"], [(0, 0, 255), (0, 255, 0), (255, 128, 0)])):
                    ys = 360 - arr[:, 3 + i] * 0.15
                    pts = np.stack([np.linspace(16, 800, len(ys)), ys], 1).astype(np.int32)
                    cv2.polylines(p, [pts], False, col, 1)
                cv2.putText(p, "accel (g)", (700, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
                cv2.putText(p, "gyro (deg/s)", (680, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
            cv2.putText(p, "s standing  w walking  r running  b biking  SPACE pause  q quit", (16, 405),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
            cv2.imshow("record gait", p)
            k = cv2.waitKey(50) & 0xFF
            if k in (27, ord("q")):
                break
            if chr(k) in KEYS if k != 255 else False:
                label[0] = KEYS[chr(k)]
                print("now labeling:", label[0])
            if k == ord(" "):
                label[0] = None
    finally:
        f.close()
        c.disconnect()
        cv2.destroyAllWindows()
        print("saved:", {k: f"{v / 50:.0f} s" for k, v in counts.items()})


if __name__ == "__main__":
    main()
