"""
UNO Q sensor node for the sumo wrestler: the sketch measures the HC-SR04
range ~16x/s and hands it over with Bridge.notify("range", cm); this side
publishes it over MQTT for the laptop brain and shows it on the LED matrix.

    range   -> ME193/ChrisLiang/sumo/range   {"cm": 37.2}      (10 Hz, -1 = nothing)
    state   <- ME193/ChrisLiang/sumo/state   (to show OUT / CHARGE on the matrix)

Matrix: a range bar on the top rows (closer = longer), a sad face when the
brain says we're out, a full-bright flash while charging.
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

PLAYER = os.environ.get("SUMO_PLAYER", "ChrisLiang")
RANGE_TOPIC = f"ME193/{PLAYER}/sumo/range"
STATE_TOPIC = f"ME193/{PLAYER}/sumo/state"
BROKER, PORT = "test.mosquitto.org", 1883
PUBLISH_HZ = 10
MAX_CM = 150.0

_lock = threading.Lock()
_cm = -1.0
_cm_t = 0.0
_state = {}
_state_t = 0.0


def range_cb(cm):                       # called from the sketch via Bridge.notify
    global _cm, _cm_t
    with _lock:
        _cm, _cm_t = float(cm), time.monotonic()


Bridge.provide("range", range_cb)


def diag_cb(code, echo_level):          # once a second from the sketch
    names = {0: "echo never went high", 1: "echo stuck HIGH", 2: "echo too long (nothing in range)", 3: "ok"}
    print(f"[ultrasonic] {names.get(int(code), code)}; echo pin idle level = {int(echo_level)}")


Bridge.provide("diag", diag_cb)


def on_connect(client, userdata, flags, rc):
    print(f"[mqtt] connected rc={rc}; range -> {RANGE_TOPIC}")
    client.subscribe(STATE_TOPIC)


def on_message(client, userdata, msg):
    global _state, _state_t
    try:
        d = json.loads(msg.payload.decode("utf-8", errors="replace"))
    except ValueError:
        return
    if isinstance(d, dict):
        with _lock:
            _state, _state_t = d, time.monotonic()


try:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
except AttributeError:
    client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.reconnect_delay_set(min_delay=1, max_delay=30)
client.connect_async(BROKER, PORT, keepalive=60)
client.loop_start()

SAD = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,7,0,0,0,0,0,7,0,0,0],
    [0,0,0,7,0,0,0,0,0,7,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,7,7,7,0,0,0,0,0],
    [0,0,0,0,7,0,0,0,7,0,0,0,0],
    [0,0,0,7,0,0,0,0,0,7,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0]], dtype=np.uint8)


def build_frame():
    with _lock:
        cm, cm_t, st, st_t = _cm, _cm_t, dict(_state), _state_t
    now = time.monotonic()
    if st.get("out") and now - st_t < 5.0:
        return SAD
    a = np.zeros((8, 13), dtype=np.uint8)
    if st.get("doing") == "charge" and now - st_t < 1.0 and int(now * 6) % 2:
        a[:] = 7                                        # flash while charging
        return a
    if 0 < cm and now - cm_t < 1.0:
        n = int(round(13 * (1 - min(cm, MAX_CM) / MAX_CM)))   # closer = longer bar
        a[0:3, :max(1, n)] = 5
    else:
        a[0, 0] = 2
    if st.get("edge") and now - st_t < 1.0:
        a[7, :] = 3                                     # tape seen: bottom row
    return a


_last_pub = 0.0


def loop():
    global _last_pub
    now = time.monotonic()
    if now - _last_pub >= 1.0 / PUBLISH_HZ:
        with _lock:
            cm, cm_t = _cm, _cm_t
        client.publish(RANGE_TOPIC, json.dumps({"cm": round(cm, 1) if now - cm_t < 0.5 else -1}))
        _last_pub = now
    try:
        Bridge.call("draw", to_board_bytes(build_frame()))
    except Exception as e:
        print(f"[bridge] draw failed: {e}")
    time.sleep(0.05)


App.run(user_loop=loop)
