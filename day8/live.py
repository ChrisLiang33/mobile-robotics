"""
Step 3 -- live dashboard on the laptop: shows what the wearable is streaming
and what BOTH models say about it (the board also classifies and shows
model 2 on its LED matrix; this window is for the audience).

    python live.py
"""

import json
import os
import time
from collections import deque

import cv2
import numpy as np

from gait_features import Models, SAMPLE_HZ, WINDOW_S
from mqttlib import MQTTClient

PLAYER = "ChrisLiang"
RAW_TOPIC = f"ME193/{PLAYER}/gait/imu"
LABEL_TOPIC = f"ME193/{PLAYER}/gait/label"
HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = {"standing": (200, 200, 200), "walking": (0, 200, 0), "running": (0, 80, 255), "biking": (255, 160, 0), "-": (90, 90, 90)}


def main():
    models = Models(json.load(open(os.path.join(HERE, "models.json"))))
    buf = deque(maxlen=int(WINDOW_S * SAMPLE_HZ))
    trace = deque(maxlen=400)
    board = {"m1": "-", "m2": "-"}
    seen = [0.0]

    def on_raw(topic, payload):
        try:
            d = json.loads(payload)
        except ValueError:
            return
        seen[0] = time.monotonic()
        for s in d.get("s", []):
            buf.append(s[1:7]); trace.append(s[1:7])

    def on_label(topic, payload):
        try:
            board.update(json.loads(payload))
        except ValueError:
            pass

    c = MQTTClient(); c.connect()
    c.subscribe(RAW_TOPIC, on_raw); c.subscribe(LABEL_TOPIC, on_label)
    local = {"m1": "-", "m2": "-"}; last = 0.0
    try:
        while True:
            now = time.monotonic()
            if len(buf) == buf.maxlen and now - last >= 0.5:
                win = np.array(buf); local["m1"], local["m2"] = models.predict1(win), models.predict2(win); last = now
            p = np.full((460, 900, 3), 24, np.uint8)
            online = now - seen[0] < 1.5
            cv2.putText(p, f"wearable {'ONLINE' if online else 'no data'}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0) if online else (0, 0, 230), 2, cv2.LINE_AA)
            cv2.putText(p, "model 1 (peak accel, one axis)", (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)
            cv2.putText(p, local["m1"].upper(), (16, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.6, COLORS.get(local["m1"], COLORS["-"]), 3, cv2.LINE_AA)
            cv2.putText(p, "model 2 (all IMU outputs)", (460, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)
            cv2.putText(p, local["m2"].upper(), (460, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.6, COLORS.get(local["m2"], COLORS["-"]), 3, cv2.LINE_AA)
            cv2.putText(p, f"the board itself says: m1 {board['m1']}   m2 {board['m2']}", (16, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
            if len(trace) > 2:
                arr = np.array(trace)
                for i, col in enumerate([(0, 0, 255), (0, 255, 0), (255, 128, 0)]):
                    ys = 300 - arr[:, i] * 40
                    cv2.polylines(p, [np.stack([np.linspace(16, 880, len(ys)), ys], 1).astype(np.int32)], False, col, 1)
                    ys = 410 - arr[:, 3 + i] * 0.12
                    cv2.polylines(p, [np.stack([np.linspace(16, 880, len(ys)), ys], 1).astype(np.int32)], False, col, 1)
                cv2.putText(p, "accel", (16, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
                cv2.putText(p, "gyro", (16, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
            cv2.imshow("gait live", p)
            if cv2.waitKey(50) & 0xFF in (27, ord("q")):
                break
    finally:
        c.disconnect(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
