"""
UNO Q wearable: the sketch streams MPU-6050 samples at 50 Hz with
Bridge.notify("imu", ax, ay, az, gx, gy, gz). This side

  * publishes the raw samples over MQTT in 100 ms batches  (for record.py / live.py)
  * if models.json is present (trained on the laptop, deployed with the app),
    classifies the last 2 s with BOTH models every 0.5 s, publishes the labels,
    and shows model 2's answer as an icon on the LED matrix
"""

import json
import os
import threading
import time
from collections import deque

import numpy as np
import paho.mqtt.client as mqtt

from arduino.app_utils import App, Bridge
from gait_features import Models, windows, SAMPLE_HZ, WINDOW_S

try:
    from arduino.app_utils import Frame

    def to_board_bytes(a):
        return Frame(a).to_board_bytes()
except ImportError:
    def to_board_bytes(a):
        return a.astype(np.uint8).tobytes()

PLAYER = os.environ.get("GAIT_PLAYER", "ChrisLiang")
RAW_TOPIC = f"ME193/{PLAYER}/gait/imu"
LABEL_TOPIC = f"ME193/{PLAYER}/gait/label"
BROKER, PORT = "test.mosquitto.org", 1883
HERE = os.path.dirname(os.path.abspath(__file__))

_lock = threading.Lock()
_buf = deque(maxlen=int(WINDOW_S * SAMPLE_HZ))       # last 2 s
_batch = []
_status = None
_models = None
_label = {"m1": "-", "m2": "-"}
if os.path.exists(os.path.join(HERE, "models.json")):
    _models = Models(json.load(open(os.path.join(HERE, "models.json"))))
    print("[gait] models loaded")
else:
    print("[gait] no models.json yet: streaming raw data only")


def imu_cb(ax, ay, az, gx, gy, gz):
    s = [float(v) for v in (ax, ay, az, gx, gy, gz)]
    with _lock:
        _buf.append(s)
        _batch.append([round(time.time(), 3)] + [round(v, 4) for v in s])


def status_cb(ok):
    global _status
    _status = int(ok)
    print("[gait] MPU-6050", "OK" if _status else "NOT FOUND - check SDA/SCL/VCC/GND")


Bridge.provide("imu", imu_cb)
Bridge.provide("status", status_cb)

try:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
except AttributeError:
    client = mqtt.Client()
client.on_connect = lambda c, u, f, rc: print(f"[mqtt] connected rc={rc}; raw -> {RAW_TOPIC}")
client.reconnect_delay_set(min_delay=1, max_delay=30)
client.connect_async(BROKER, PORT, keepalive=60)
client.loop_start()

# 13x8 icons
ICONS = {
    "standing": ["......X......", "......X......", ".....XXX.....", "......X......", "......X......", "......X......", ".....X.X.....", ".....X.X....."],
    "walking":  ["......X......", ".....XXX.....", "....X.X.X....", "......X......", ".....X.X.....", "....X...X....", "...X.....X...", "............."],
    "running":  ["........X....", ".......XXX...", "....X.X.X.X..", "......X......", ".....X.X.....", "...X....X....", "..X......X...", "............."],
    "biking":   ["......X......", ".....XXX.....", "......X.X....", "....X.X.X....", "...X...X.....", "..XXX.XXX....", ".X...X...X...", "..XXX.XXX...."],
    "none":     [".............", ".............", ".............", "......X......", "......X......", ".............", ".............", "............."],
}


def icon(name):
    a = np.zeros((8, 13), dtype=np.uint8)
    for r, row in enumerate(ICONS.get(name, ICONS["none"])):
        for c, ch in enumerate(row[:13]):
            if ch == "X":
                a[r, c] = 7
    return a


_last_pub = _last_cls = 0.0


def loop():
    global _last_pub, _last_cls, _batch
    now = time.monotonic()
    if now - _last_pub >= 0.1:
        with _lock:
            batch, _batch = _batch, []
        if batch:
            client.publish(RAW_TOPIC, json.dumps({"hz": SAMPLE_HZ, "s": batch}))
        _last_pub = now
    if _models is not None and now - _last_cls >= 0.5:
        with _lock:
            full = len(_buf) == _buf.maxlen
            win = np.array(_buf) if full else None
        if full:
            _label["m1"], _label["m2"] = _models.predict1(win), _models.predict2(win)
            client.publish(LABEL_TOPIC, json.dumps(_label))
        _last_cls = now
    try:
        shown = _label["m2"] if _models is not None else ("none" if _status else "none")
        a = icon(shown)
        if _status == 0:
            a[7, :] = 3                                   # bottom row lit: sensor missing
        Bridge.call("draw", to_board_bytes(a))
    except Exception as e:
        print(f"[bridge] draw failed: {e}")
    time.sleep(0.05)


App.run(user_loop=loop)
