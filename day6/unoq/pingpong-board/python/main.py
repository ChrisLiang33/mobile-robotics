"""
UNO Q scoreboard for Day 6 ping-pong: draws the live court on the 8x13 LED
matrix from the JSON that pingpong.py publishes on ME193/Rogers/<name>/live.

Rows 0-6: the court -- a bright pixel for the ball, a 3-pixel bar at the
right edge for the paddle. Row 7: the current streak as a bar (one pixel per
hit, up to 13). Before the game starts the center pixel blinks.
Works on old and new board images (same compatibility shims as Day 5).
"""

import json
import os
import threading
import time

import numpy as np
import paho.mqtt.client as mqtt

from arduino.app_utils import App, Bridge

try:
    from arduino.app_utils import Frame

    def to_board_bytes(a):
        return Frame(a).to_board_bytes()
except ImportError:
    def to_board_bytes(a):
        return a.astype(np.uint8).tobytes()

MQTT_BROKER, MQTT_PORT = "test.mosquitto.org", 1883
LIVE_TOPIC = os.environ.get("PINGPONG_TOPIC", "ME193/Rogers/ChrisLiang/live")
ROWS, COLS = 8, 13
COURT_ROWS = 7
REFRESH = 0.05
STALE = 2.0

_lock = threading.Lock()
_live = None
_seen = 0.0


def on_connect(client, userdata, flags, rc):
    print(f"[mqtt] connected rc={rc}, subscribing to {LIVE_TOPIC}")
    client.subscribe(LIVE_TOPIC)


def on_message(client, userdata, msg):
    global _live, _seen
    try:
        d = json.loads(msg.payload.decode("utf-8", errors="replace"))
    except ValueError:
        return
    if isinstance(d, dict) and "bx" in d:
        with _lock:
            _live, _seen = d, time.monotonic()


def build_frame():
    a = np.zeros((ROWS, COLS), dtype=np.uint8)
    with _lock:
        d, seen = _live, _seen
    now = time.monotonic()
    if d is None or now - seen > STALE or d.get("st") == "WAI":
        if int(now * 2) % 2:                       # blink: waiting for the game
            a[ROWS // 2, COLS // 2] = 3
        return a
    bx = max(0, min(COLS - 2, int(d["bx"] * (COLS - 1))))
    by = max(0, min(COURT_ROWS - 1, int(d["by"] * (COURT_ROWS - 1))))
    a[by, bx] = 7
    if d.get("py") is not None:
        py = max(1, min(COURT_ROWS - 2, int(d["py"] * (COURT_ROWS - 1))))
        a[py - 1:py + 2, COLS - 1] = 4
    bar = min(COLS, int(d.get("s", 0)))
    if bar:
        a[ROWS - 1, :bar] = 2
    return a


try:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
except AttributeError:
    client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.reconnect_delay_set(min_delay=1, max_delay=30)
client.connect_async(MQTT_BROKER, MQTT_PORT, keepalive=60)
client.loop_start()


def loop():
    try:
        Bridge.call("draw", to_board_bytes(build_frame()))
    except Exception as e:
        print(f"[bridge] draw failed: {e}")
    time.sleep(REFRESH)


App.run(user_loop=loop)
