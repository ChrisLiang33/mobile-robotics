"""
UNO Q side of Day 5 (runs on the board's Linux side under App Lab).

Part 1: subscribe to the laptop's ME193/minifig feed and show the minifig's
        position as a blue marker at the scaled spot on the 8x13 LED matrix.
Part 2: drive the DC motors forward/backward so the minifig ends up in the
        middle of the laptop camera's view.

The display half is Prof. Rogers' "MQTT Minifig Monitor" App Lab project;
the drive policy is ours.

Policy (evaluated every REFRESH_INTERVAL):
    err   = (x - w/2) / (w/2)          # -1 = left edge, 0 = center, +1 = right edge
    speed = KP * err                    # proportional: far = fast, close = slow
    |err| < DEADBAND         -> 0       # parked in the middle
    0 < |speed| < MIN_SPEED  -> MIN     # beat the motors' static friction
    clamp to +-MAX_SPEED, * DIRECTION   # DIRECTION = -1 if it drives away
No minifig: an explicit {"found": false} stops the motors at once; if the
laptop goes silent for STALE_TIMEOUT the motors stop anyway (and the LED
marker shrinks to 1 pixel at the last known position).
"""

import json
import os
import threading
import time

import numpy as np
import paho.mqtt.client as mqtt

from arduino.app_utils import App, Bridge

try:                                    # newer board images ship a Frame helper
    from arduino.app_utils import Frame

    def to_board_bytes(array):
        return Frame(array).to_board_bytes()
except ImportError:                     # older images (app-bricks 0.5): by hand --
    def to_board_bytes(array):          # 104 brightness bytes, row-major, values 0-7
        return array.astype(np.uint8).tobytes()

# --- MQTT ------------------------------------------------------------------
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
# Our own topic, not the shared class default "ME193/minifig": with several
# teams in one room, a shared topic means everyone's detections drive
# everyone's car. Must match MQTT_TOPIC in the laptop's common.py.
MQTT_TOPIC = os.environ.get("MINIFIG_TOPIC", "ME193/minifig/chris")
DRIVE_TOPIC = MQTT_TOPIC + "/drive"       # we echo our decisions here (debugging)
TEST_TOPIC = MQTT_TOPIC + "/test"         # publish {"speed": 50} here to spin the
TEST_HOLD_S = 2.0                         # motors for 2 s -- wiring test, no camera needed
MQTT_HEARTBEAT_TOPIC = "ME193/heartbeat"
HEARTBEAT_INTERVAL = 60
DEVICE_ID = "ChrisLiang-UNOQ"              # shown in the heartbeat; set to your board's name

# --- Display ---------------------------------------------------------------
FRAME_ROWS, FRAME_COLS = 8, 13
PIXEL_BRIGHTNESS = 7
REFRESH_INTERVAL = 0.05
FRESH_MARKER_SIZE, STALE_MARKER_SIZE = 3, 1
STALE_TIMEOUT = 0.75

# --- Drive policy ----------------------------------------------------------
KP = 70.0            # percent of full speed per unit of normalized error
DEADBAND = 0.08      # +-8% of half-width counts as centered
MIN_SPEED = 30       # smallest command that actually moves the car
MAX_SPEED = 80
DIRECTION = +1       # flip to -1 if the car drives away from center
DRIVE_STALE = 0.5    # s without a position -> stop
DRIVE_KEEPALIVE = 0.3  # resend an unchanged speed this often: the sketch's
                       # watchdog stops the motors after 1 s of silence
DRIVE_ECHO_HZ = 5

_state_lock = threading.Lock()
_last_col = _last_row = None
_last_seen = 0.0
_pos_x = _pos_w = None      # latest position (pixels) and frame width
_pos_time = 0.0
_lost = True
_test_speed = 0            # manual motor test (from TEST_TOPIC)
_test_until = 0.0


def on_connect(client, userdata, flags, rc):
    print(f"[mqtt] connected (rc={rc}), subscribing to {MQTT_TOPIC!r}")
    client.subscribe(MQTT_TOPIC)
    client.subscribe(TEST_TOPIC)
    send_heartbeat(force=True)


def on_disconnect(client, userdata, rc):
    print(f"[mqtt] disconnected (rc={rc})")


def on_message(client, userdata, msg):
    global _last_col, _last_row, _last_seen, _pos_x, _pos_w, _pos_time, _lost
    global _test_speed, _test_until
    try:
        data = json.loads(msg.payload.decode("utf-8", errors="replace"))
    except (ValueError, json.JSONDecodeError):
        return
    now = time.monotonic()
    if msg.topic == TEST_TOPIC:             # wiring test: {"speed": 50} or just 50
        try:
            spd = int(data["speed"] if isinstance(data, dict) else data)
        except (TypeError, KeyError, ValueError):
            return
        with _state_lock:
            _test_speed, _test_until = max(-100, min(100, spd)), now + TEST_HOLD_S
        print(f"[test] motors at {spd} for {TEST_HOLD_S}s")
        return
    if not isinstance(data, dict):
        return
    if data.get("found") is False:          # laptop says: nothing in view
        with _state_lock:
            _lost = True
            _pos_time = now
        return
    try:
        x, y, w, h = data["x"], data["y"], data["w"], data["h"]
    except (TypeError, KeyError):
        return
    if not w or not h:
        return
    col = max(0, min(FRAME_COLS - 1, int(x / w * FRAME_COLS)))
    row = max(0, min(FRAME_ROWS - 1, int(y / h * FRAME_ROWS)))
    with _state_lock:
        _last_col, _last_row, _last_seen = col, row, now
        _pos_x, _pos_w, _pos_time, _lost = float(x), float(w), now, False


def build_frame():
    array = np.zeros((FRAME_ROWS, FRAME_COLS), dtype=np.uint8)
    with _state_lock:
        col, row, last_seen = _last_col, _last_row, _last_seen
    if col is None:
        return array
    fresh = (time.monotonic() - last_seen) <= STALE_TIMEOUT
    half = (FRESH_MARKER_SIZE if fresh else STALE_MARKER_SIZE) // 2
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            r, c = row + dr, col + dc
            if 0 <= r < FRAME_ROWS and 0 <= c < FRAME_COLS:
                array[r, c] = PIXEL_BRIGHTNESS
    return array


def compute_speed():
    """The policy: normalized horizontal error -> signed motor speed."""
    with _state_lock:
        lost, x, w, t = _lost, _pos_x, _pos_w, _pos_time
        test_speed, test_until = _test_speed, _test_until
    if time.monotonic() < test_until:       # manual wiring test overrides the policy
        return test_speed, None
    if lost or x is None or time.monotonic() - t > DRIVE_STALE:
        return 0, None
    err = (x - w / 2) / (w / 2)
    if abs(err) < DEADBAND:
        return 0, err
    speed = KP * err
    if abs(speed) < MIN_SPEED:
        speed = MIN_SPEED if speed > 0 else -MIN_SPEED
    speed = max(-MAX_SPEED, min(MAX_SPEED, speed)) * DIRECTION
    return int(round(speed)), err


try:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)   # paho-mqtt 2.x
except AttributeError:
    client = mqtt.Client()                                    # paho-mqtt 1.x
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message
client.reconnect_delay_set(min_delay=1, max_delay=30)
# connect_async + loop_start retries in the background, so the app survives
# being started at boot before WiFi is up, and WiFi dropouts afterwards.
client.connect_async(MQTT_BROKER, MQTT_PORT, keepalive=60)
client.loop_start()

_last_heartbeat = time.monotonic()
_last_cmd = None
_last_drive_sent = 0.0
_last_echo = 0.0
_last_bridge_error = 0.0


def bridge_call(name, *args):
    """Bridge.call that reports a failure instead of killing the app."""
    global _last_bridge_error
    try:
        Bridge.call(name, *args)
        return True
    except Exception as e:                      # keep the control loop alive
        now = time.monotonic()
        if now - _last_bridge_error > 2.0:      # don't flood the log at 20 Hz
            print(f"[bridge] {name} failed: {e}")
            _last_bridge_error = now
        return False


def send_heartbeat(force=False):
    global _last_heartbeat
    now = time.monotonic()
    if not force and now - _last_heartbeat < HEARTBEAT_INTERVAL:
        return
    _last_heartbeat = now
    client.publish(MQTT_HEARTBEAT_TOPIC, json.dumps({"device": DEVICE_ID, "ts": time.time()}))


def loop():
    global _last_cmd, _last_drive_sent, _last_echo
    send_heartbeat()
    bridge_call("draw", to_board_bytes(build_frame()))

    speed, err = compute_speed()
    now = time.monotonic()
    changed = speed != _last_cmd
    # Send on change, and keep resending a non-zero speed so the sketch's
    # watchdog knows we're still here.
    if changed or (speed != 0 and now - _last_drive_sent >= DRIVE_KEEPALIVE):
        if bridge_call("drive", speed):
            _last_cmd, _last_drive_sent = speed, now
            if changed:
                print(f"[drive] err={err if err is None else round(err, 3)} -> speed {speed}")
    if now - _last_echo >= 1.0 / DRIVE_ECHO_HZ:
        client.publish(DRIVE_TOPIC, json.dumps({"err": None if err is None else round(err, 3),
                                                "speed": speed}))
        _last_echo = now
    time.sleep(REFRESH_INTERVAL)


App.run(user_loop=loop)
