"""
Laptop side of the sumo wrestler when the brain (sumo.py --headless) runs on
the UNO Q: shows the robot's state as it arrives over MQTT, sends your key
presses to the robot over MQTT, and plays the sad song when the robot
announces that it lost.

    python sumo_remote.py

Keys (same as sumo.py): W/S forward/back  A/D turn  SPACE stop  f/t teach floor/tape
                        x referee: we're out  r reset  c fake contact  q quit
"""

import json
import time

import cv2
import numpy as np

import sounds
from mqttlib import MQTTClient
from sumo import PLAYER, STATE_TOPIC, TELEOP_TOPIC, RESULT_TOPIC

state = {}
seen = [0.0]
results = []


def on_state(topic, payload):
    try:
        state.update(json.loads(payload))
        seen[0] = time.monotonic()
    except ValueError:
        pass


def on_result(topic, payload):
    results.append(payload)
    print("***", payload)
    if payload.startswith(f"{PLAYER} lost"):
        sounds.play("sad")


def draw():
    p = np.full((420, 720, 3), 24, np.uint8)
    W, Y, G, Rd, Gy = (255, 255, 255), (0, 220, 255), (0, 200, 0), (0, 0, 230), (130, 130, 130)
    now = time.monotonic()
    online = now - seen[0] <= 2.0
    cv2.putText(p, f"SUMO remote  -  robot {'ONLINE' if online else 'OFFLINE (no state for 2 s)'}", (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, G if online else Rd, 2, cv2.LINE_AA)
    y = 70
    if state:
        rows = [
            (f"doing: {state.get('doing')}", Rd if state.get("out") else G),
            (f"OUT - {state.get('out_reason')}" if state.get("out") else "", Rd),
            (f"floor: {state.get('floor')} (knn {state.get('knn_conf', 0):.0%}, samples {state.get('knn_counts')})   sensor says {state.get('sensor_color')}   refl {state.get('reflection')}",
             Rd if state.get("edge") else W),
            (f"contact: {'YES' if state.get('contact') else 'no'}   hub jerk {state.get('jerk')} mg", Rd if state.get("contact") else W),
            (f"human: {state.get('human') or 'no input'}" + ("   VETOED: tape ahead" if state.get("veto") else ""), Rd if state.get("veto") else W),
            (f"wheels: {state.get('cmd')}", W),
        ]
        for text, col in rows:
            if text:
                cv2.putText(p, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
            y += 28
    for r in results[-3:]:
        cv2.putText(p, r, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, Y, 1, cv2.LINE_AA); y += 26
    cv2.putText(p, "W/S fwd/back  A/D turn  SPACE stop  f/t teach floor/tape  c fake contact", (16, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.45, Gy, 1, cv2.LINE_AA)
    cv2.putText(p, "x referee: OUT   r reset   q quit        keys go to the robot over MQTT", (16, 394), cv2.FONT_HERSHEY_SIMPLEX, 0.45, Gy, 1, cv2.LINE_AA)
    return p


def main():
    c = MQTTClient()
    c.connect()
    c.subscribe(STATE_TOPIC, on_state)
    c.subscribe(RESULT_TOPIC, on_result)
    print(f"watching {STATE_TOPIC}, sending keys to {TELEOP_TOPIC}")
    arrows = {82: "w", 84: "s", 81: "a", 83: "d"}
    try:
        while True:
            cv2.imshow("sumo remote", draw())
            key = cv2.waitKey(50) & 0xFF
            if key in (27, ord("q")):
                break
            ch = arrows.get(key, chr(key) if key != 255 else "")
            if ch in ("w", "a", "s", "d", " ", "x", "r", "c", "f", "t", "e"):
                c.publish(TELEOP_TOPIC, json.dumps({"key": ch}))
    finally:
        c.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
