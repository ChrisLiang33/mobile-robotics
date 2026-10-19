"""
Day 7 -- Sumo wrestler: LEGO Double Motor base + Color Sensor + UNO Q ultrasonic,
decided by a BEHAVIOR TREE, with the human in the loop over the keyboard.

    python sumo.py                 # hub + color sensor over Bluetooth, UNO Q range over MQTT
    python sumo.py --sim           # no hardware: e = edge, o = opponent in range, c = contact
    python sumo.py --any-motor     # don't pin the hub to my card
    python sumo.py --headless      # no window: keys arrive over MQTT from sumo_remote.py

Sensors feeding the decision (the "sensor fusion"):
    autonomous  UNO Q + HC-SR04 ultrasonic on the plow, over MQTT -> opponent ahead, how far
    autonomous  Color Sensor at the back, pointing at the floor    -> edge of the ring (blue tape)
    autonomous  hub IMU (accelerometer + gesture)                  -> contact: we're touching them
    teleop      keyboard (W/A/S/D)                                 -> where the human wants to push

The tree, highest priority first (bt.py):
    1. OUT?        tape under the rear sensor for OUT_TAPE_S -> we've been pushed to
                   the edge: stop, play the sad song, publish "lost"
    2. OPPONENT?   ultrasonic sees them within CHARGE_CM -> charge at 100%
                   (the auto power-up; A/D can angle it, the human can't cancel it)
    3. HUMAN?      fresh keyboard command -> drive it
    4. otherwise   sit still (or search, if SEARCH_WHEN_IDLE)

Tape is the Color Sensor reporting "Blue" (its own classifier, reliable
within ~1 cm of clean tape), OR a confident vote from the K-NEAREST-NEIGHBORS
model (knn.py) if you taught it: press f on floor and t on tape a few times
each. The KNN is a fallback for tape the sensor won't call Blue.

Starts in MANUAL mode: the motors only follow the keyboard while the HUD
shows what the tree would do -- test one sensor at a time. Press m to hand
control to the tree (autonomy ON) for the match.

Keys: m autonomy on/off  W/S forward/back  A/D turn  SPACE stop  f/t record floor/tape
      x referee: declare us out  r reset  c fake contact  e fake edge (--sim)  q quit
"""

import argparse
import json
import os
import random
import threading
import time

import numpy as np

try:
    import cv2
except ImportError:            # headless board: no window, keys come over MQTT
    cv2 = None

import bt
import knn
import sounds

PLAYER = "ChrisLiang"
RANGE_TOPIC = f"ME193/{PLAYER}/sumo/range"    # UNO Q ultrasonic -> us: {"cm": 37.2}  (~10 Hz)
BUMP_TOPIC = f"ME193/{PLAYER}/sumo/bump"      # optional UNO Q bumper -> us: {"bump": true/false}
STATE_TOPIC = f"ME193/{PLAYER}/sumo/state"    # us -> the laptop remote + anyone watching (5 Hz)
TELEOP_TOPIC = f"ME193/{PLAYER}/sumo/teleop"  # laptop remote -> us: {"key": "w"}
RESULT_TOPIC = f"ME193/{PLAYER}/sumo/result"  # us -> everyone: who lost

# --- the ring edge sensor ----------------------------------------------------
SENSOR_AT = "back"        # "back": tape behind us -> escape FORWARD; "front": back up, then turn

# --- the opponent, by ultrasonic ---------------------------------------------
CHARGE_CM = 20.0          # opponent closer than this, straight ahead -> charge at full power
RANGE_STALE = 0.5         # s without a range message -> "don't know"
RANGE_MAX_CM = 200.0      # HC-SR04 readings beyond this are "nothing there"

# --- contact from the hub's IMU --------------------------------------------
ACCEL_HIT = 350           # milli-g away from resting 1 g that counts as a collision
CONTACT_HOLD = 0.6        # s a collision keeps "contact" true (the shove lasts at least this)

# --- my LEGO devices' connection cards ---------------------------------------
CARD_COLOR_NAME, CARD_SERIAL = "BLUE", 1392          # Double Motor (shares the card with the sensor)
COLOR_CARD_NAME, COLOR_CARD_SERIAL = "BLUE", 1392    # Color Sensor
LEFT_SIGN, RIGHT_SIGN = +1, -1                # motors are mounted mirror-image
DIRECTION = +1                                # flip if "forward" drives the plow backwards

# --- speeds (percent) -------------------------------------------------------
SHOVE_SPEED = 100
HUMAN_MAX = 80
SEARCH_SPIN = 40
SEARCH_CREEP = 45
RETREAT_BACK = (-75, 0.55)                    # front sensor: speed, seconds backing up
RETREAT_TURN = (60, 0.65)                     # front sensor: then turning
ESCAPE_FWD = (90, 0.6)                        # back sensor: drive forward, away from the tape

# --- timing -----------------------------------------------------------------
TICK_HZ = 20
KEY_HOLD = 0.45           # a key press counts as "held" this long (key repeat refreshes it)
CONTACT_STALE = CONTACT_HOLD
UNOQ_STALE = 3.0          # s without any UNO Q message -> show it as offline (optional sensor)
OUT_TAPE_S = 0.3          # tape under the rear sensor this long -> pushed to the edge, out
SEARCH_WHEN_IDLE = False  # True: spin/creep looking for the opponent when nobody's driving
USE_IMU_CONTACT = False   # True: an IMU jolt also triggers a full-power shove
EDGE_AGREE = 3            # consecutive sensor readings (~150 ms) that must agree before the
                          # edge flag flips -- a single dip of the car doesn't count as tape

HERE = os.path.dirname(os.path.abspath(__file__))
KNN_PATH = os.path.join(HERE, "knn_samples.json")


# ============================================================================
# Blackboard + sensors
# ============================================================================
def new_blackboard():
    return {
        "edge": False, "tape_since": None, "last_tape": -1e9, "floor_class": "?", "knn_conf": 0.0,
        "rgb": (0, 0, 0), "reflection": 0, "sensor_color": "-",
        "contact": False, "last_bump": -1e9, "unoq_seen": -1e9, "jerk": 0.0, "gesture": -1,
        "range_cm": None, "range_t": -1e9, "opponent": False,
        "human": None, "human_t": -1e9,                     # (speed, turn)
        "out": False, "out_reason": "", "retreats": [],
        "retreat": None,                                    # ("back"|"turn", until)
        "search": None,                                     # ("spin"|"creep", until, dir)
        "active": "-", "trace": [], "cmd": (0, 0), "now": 0.0,
        "mourned": False, "manual": True,
    }


def sensor_loop(cs, classifier, bb, running):
    """Color Sensor at ~20 Hz: raw color -> KNN -> edge flag."""
    while running[0]:
        try:
            rgb = cs.raw_rgb()
            refl = cs.reflection()
            name = cs.detect_color()
        except Exception as e:
            time.sleep(0.2)
            continue
        x = knn.features(rgb, refl)
        label, conf = classifier.predict(x)
        if name == "Blue":                        # the sensor's own verdict is tape, always
            label, conf = "tape", 1.0
        elif label is None or conf < 0.8:         # untrained or unsure KNN: not tape
            label, conf = "floor", (conf if label else 0.5)
        now = time.monotonic()
        bb["rgb"], bb["reflection"], bb["sensor_color"] = rgb, refl, name
        bb["floor_class"], bb["knn_conf"], bb["_last_x"] = label, conf, x
        # debounce: the flag only flips after EDGE_AGREE readings in a row agree
        votes = bb.setdefault("_votes", [])
        votes.append(label == "tape")
        del votes[:-EDGE_AGREE]
        if len(votes) == EDGE_AGREE and all(v == votes[0] for v in votes):
            set_edge(bb, votes[0], now)
        time.sleep(0.05)


def imu_loop(dm, le, bb, running):
    """Hub IMU at ~50 Hz: an acceleration spike or the hub's own COLLISION
    gesture means we just hit something -- the opponent."""
    import math
    while running[0]:
        try:
            d = dm.imu_device
            mag = math.sqrt(d.accelerometerX ** 2 + d.accelerometerY ** 2 + d.accelerometerZ ** 2)
            jerk = abs(mag - 1000.0)                       # resting = 1 g = 1000 mg
            gesture = dm.imu_gesture.gesture
        except Exception:
            time.sleep(0.1)
            continue
        if not math.isnan(jerk):
            bb["jerk"] = jerk
            bb["gesture"] = gesture
            hit = jerk >= ACCEL_HIT or gesture == getattr(le, "MOTION_GESTURE_COLLISION", 2)
            if hit and not bb["out"]:
                bb["last_bump"] = time.monotonic()
        time.sleep(0.02)


def set_edge(bb, on_tape, now):
    bb["edge"] = on_tape
    if on_tape:
        bb["tape_since"] = bb["tape_since"] or now
        bb["last_tape"] = now
    else:
        bb["tape_since"] = None


# ============================================================================
# Behavior tree
# ============================================================================
def build_tree(drive):
    def is_out(bb):
        now = bb["now"]
        if bb["tape_since"] and now - bb["tape_since"] >= OUT_TAPE_S and not bb["out"]:
            bb["out"], bb["out_reason"] = True, "pushed onto the tape"
        return bb["out"]

    def mourn(bb):
        drive(0, 0)
        if not bb["mourned"]:
            bb["mourned"] = True
            sounds.play("sad")
        return bt.SUCCESS

    def edge_or_retreating(bb):
        return bb["edge"] or bb["retreat"] is not None

    def retreat(bb):
        now = bb["now"]
        if bb["retreat"] is None:                      # start: remember the event
            bb["retreats"].append(now)
            first = "fwd" if SENSOR_AT == "back" else "back"
            bb["retreat"] = (first, now + (ESCAPE_FWD[1] if first == "fwd" else RETREAT_BACK[1]))
            bb["turn_dir"] = -1 if (bb["human"] and bb["human"][1] < 0) else (1 if bb["human"] and bb["human"][1] > 0 else random.choice((-1, 1)))
        phase, until = bb["retreat"]
        if phase == "fwd":                             # rear sensor saw tape: get away from it
            if now >= until:
                bb["retreat"] = None
                return bt.FAILURE
            drive(ESCAPE_FWD[0], ESCAPE_FWD[0])
            return bt.RUNNING
        if phase == "back" and now >= until:            # backing done -> start turning now
            bb["retreat"] = ("turn", now + RETREAT_TURN[1])
            phase, until = bb["retreat"]
        if phase == "turn" and now >= until:            # retreat finished: step aside so the
            bb["retreat"] = None                        # selector can pick a lower branch this tick
            return bt.FAILURE
        if phase == "back":
            drive(RETREAT_BACK[0], RETREAT_BACK[0])
        else:
            s = RETREAT_TURN[0] * bb["turn_dir"]
            drive(s, -s)
        return bt.RUNNING

    def opponent_ahead(bb):
        return bb["opponent"]

    def charge(bb):
        bias = 0
        if bb["human"] and bb["now"] - bb["human_t"] <= KEY_HOLD:
            bias = int(20 * np.sign(bb["human"][1]))
        drive(100 - max(0, -bias), 100 - max(0, bias))
        return bt.SUCCESS

    def in_contact(bb):
        return bb["contact"]

    def shove(bb):
        bias = 0
        if bb["human"] and bb["now"] - bb["human_t"] <= KEY_HOLD:
            bias = int(25 * np.sign(bb["human"][1]))     # the human can angle the push
        drive(SHOVE_SPEED - max(0, -bias) , SHOVE_SPEED - max(0, bias))
        return bt.SUCCESS

    def human_driving(bb):
        return bb["human"] is not None and bb["now"] - bb["human_t"] <= KEY_HOLD

    def follow_human(bb):
        speed, turn = bb["human"]
        speed = int(np.clip(speed, -HUMAN_MAX, HUMAN_MAX))
        drive(speed + turn, speed - turn)
        return bt.SUCCESS

    def idle(bb):
        drive(0, 0)
        return bt.SUCCESS

    def search(bb):
        now = bb["now"]
        if bb["search"] is None or now >= bb["search"][1]:
            if bb["search"] and bb["search"][0] == "spin":
                bb["search"] = ("creep", now + 0.7, bb["search"][2])
            else:
                bb["search"] = ("spin", now + 0.9, random.choice((-1, 1)))
        phase, _, d = bb["search"]
        if phase == "spin":
            drive(SEARCH_SPIN * d, -SEARCH_SPIN * d)
        else:
            drive(SEARCH_CREEP, SEARCH_CREEP)
        return bt.SUCCESS

    branches = [
        bt.Sequence("out", bt.Condition("tape under sensor?", is_out), bt.Action("stop + sad song", mourn)),
        bt.Sequence("charge", bt.Condition("opponent in range?", opponent_ahead), bt.Action("charge", charge)),
    ]
    if USE_IMU_CONTACT:
        branches.append(bt.Sequence("attack", bt.Condition("contact (IMU)?", in_contact), bt.Action("shove", shove)))
    branches.append(bt.Sequence("teleop", bt.Condition("human command?", human_driving), bt.Action("follow human", follow_human)))
    branches.append(bt.Action("search", search) if SEARCH_WHEN_IDLE else bt.Action("idle", idle))
    return bt.Selector("sumo", *branches)


# ============================================================================
# Motors, MQTT, display
# ============================================================================
class Motors:
    def __init__(self, dm, le):
        self.dm, self.le, self.last = dm, le, None

    def drive(self, left, right):
        l = int(np.clip(left * DIRECTION, -100, 100))
        r = int(np.clip(right * DIRECTION, -100, 100))
        if (l, r) == self.last:
            return
        self.last = (l, r)
        if self.dm is None:
            return
        try:
            le = self.le
            if l == 0:
                self.dm.motor_stop(motor=le.MOTOR_LEFT)
            else:
                self.dm.motor_run(motor=le.MOTOR_LEFT, speed=LEFT_SIGN * l, blocking=False)
            if r == 0:
                self.dm.motor_stop(motor=le.MOTOR_RIGHT)
            else:
                self.dm.motor_run(motor=le.MOTOR_RIGHT, speed=RIGHT_SIGN * r, blocking=False)
        except Exception as e:
            print(f"[hub] motor command failed: {str(e).splitlines()[0][:80]}")
            self.last = None


FACE = {True: (0, 200, 0), False: (0, 0, 220)}


def classifier_counts(bb):
    c = bb.get("_knn_counts") or {}
    return f"floor {c.get('floor', 0)} / tape {c.get('tape', 0)}"


def draw_hud(bb, tree_lines, sim):
    p = np.full((560, 760, 3), 24, np.uint8)
    now = bb["now"]
    W, Y = (255, 255, 255), (0, 220, 255)
    G, Rd, Gy = (0, 200, 0), (0, 0, 230), (130, 130, 130)
    cv2.putText(p, "SUMO  -  behavior tree", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, Y, 2, cv2.LINE_AA)
    if bb["manual"]:
        cv2.putText(p, "MANUAL  (keys only; m = autonomy ON)", (330, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, Y, 2, cv2.LINE_AA)
        cv2.putText(p, f"tree would: {bb['active']}", (16, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.8, Gy, 2, cv2.LINE_AA)
    else:
        cv2.putText(p, "AUTONOMY ON  (m = manual)", (400, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, G, 2, cv2.LINE_AA)
        cv2.putText(p, f"doing: {bb['active']}", (16, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.8, G if not bb["out"] else Rd, 2, cv2.LINE_AA)
    if bb["out"]:
        cv2.putText(p, f"OUT - {bb['out_reason']}   (r to reset)", (16, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.65, Rd, 2, cv2.LINE_AA)
    # sensors
    y = 130
    cv2.putText(p, "sensors", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, Gy, 1, cv2.LINE_AA); y += 26
    edge_col = Rd if bb["edge"] else G
    cv2.putText(p, f"floor: {bb['floor_class']}  (knn {bb['knn_conf']:.0%}, samples {classifier_counts(bb)})   refl {bb['reflection']}  rgb {tuple(int(v) for v in bb['rgb'])}",
                (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, edge_col, 1, cv2.LINE_AA); y += 24
    uno = now - bb["unoq_seen"] <= UNOQ_STALE
    rng = bb["range_cm"]
    fresh = now - bb["range_t"] <= RANGE_STALE
    cv2.putText(p, f"ultrasonic: {f'{rng:.0f} cm' if (fresh and rng is not None) else 'no reading'}   opponent in range: {'YES' if bb['opponent'] else 'no'} (< {CHARGE_CM:.0f} cm)",
                (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, Rd if bb["opponent"] else (W if fresh else Gy), 1, cv2.LINE_AA); y += 24
    cv2.putText(p, f"contact: {'YES' if bb['contact'] else 'no'}   hub IMU jerk {bb['jerk']:.0f} mg (hit >= {ACCEL_HIT})"
                + (f"   gesture {bb['gesture']}" if bb['gesture'] not in (-1, None) else "") + ("   + UNO Q bumper online" if uno else ""),
                (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, Rd if bb["contact"] else W, 1, cv2.LINE_AA); y += 24
    h = bb["human"] if bb["human"] and now - bb["human_t"] <= KEY_HOLD else None
    cv2.putText(p, f"human: {h if h else 'no input'}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, W, 1, cv2.LINE_AA); y += 24
    cv2.putText(p, f"wheels: L {bb['cmd'][0]:+4d}  R {bb['cmd'][1]:+4d}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, W, 1, cv2.LINE_AA); y += 30
    # the tree with the active path highlighted
    cv2.putText(p, "tree (top = highest priority)", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, Gy, 1, cv2.LINE_AA); y += 24
    for line in tree_lines:
        hot = any(t in line for t in bb["trace"]) and line.strip().split(": ")[-1] in bb["trace"]
        cv2.putText(p, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, G if hot else Gy, 1, cv2.LINE_AA); y += 20
    # keys
    cv2.putText(p, "W/S fwd/back  A/D turn  SPACE stop  f/t teach floor/tape  c fake contact" + ("  e fake edge  o fake opponent" if sim else ""),
                (16, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.45, Gy, 1, cv2.LINE_AA)
    cv2.putText(p, "m autonomy on/off   x referee: OUT   r reset   q quit", (16, 544), cv2.FONT_HERSHEY_SIMPLEX, 0.45, Gy, 1, cv2.LINE_AA)
    return p


def handle_key(ch, bb, classifier, sim):
    """One teleop key, from the local window or from the laptop over MQTT."""
    now = time.monotonic()
    if ch in ("w", "s", "a", "d"):
        speed, turn = (bb["human"] or (0, 0)) if now - bb["human_t"] <= KEY_HOLD else (0, 0)
        if ch == "w": speed, turn = HUMAN_MAX, 0
        if ch == "s": speed, turn = -HUMAN_MAX, 0
        if ch == "a": turn = -45
        if ch == "d": turn = 45
        bb["human"], bb["human_t"] = (speed, turn), now
    elif ch == " ":
        bb["human"], bb["human_t"] = (0, 0), now
    elif ch == "c":
        bb["last_bump"] = now
    elif ch == "e" and sim:
        set_edge(bb, not bb["edge"], now)
    elif ch == "o" and sim:                       # fake the ultrasonic: opponent 25 cm ahead for a moment
        bb["range_cm"], bb["range_t"] = (25.0 if not bb["opponent"] else 150.0), now + 1.0
    elif ch == "m":
        bb["manual"] = not bb["manual"]
        print("MANUAL (keys only)" if bb["manual"] else "AUTONOMY ON")
    elif ch == "x":
        bb["out"], bb["out_reason"] = True, "referee"
    elif ch == "r":
        bb.update(out=False, out_reason="", mourned=False, announced=False, retreats=[], retreat=None, tape_since=None)
    elif ch in ("f", "t") and "_last_x" in bb:
        classifier.add(bb["_last_x"], "floor" if ch == "f" else "tape")
        bb["_knn_counts"] = classifier.counts()
        print("knn samples:", classifier.counts(), " reading:", bb["rgb"], "refl", bb["reflection"])


def state_message(bb, classifier):
    now = bb["now"]
    return {"doing": bb["active"], "edge": bb["edge"], "contact": bb["contact"], "out": bb["out"],
            "range_cm": bb["range_cm"] if now - bb["range_t"] <= RANGE_STALE else None, "opponent": bb["opponent"],
            "out_reason": bb["out_reason"], "human": bb["human"] if bb["human"] and now - bb["human_t"] <= KEY_HOLD else None,
            "cmd": bb["cmd"], "floor": bb["floor_class"], "knn_conf": round(bb["knn_conf"], 2),
            "knn_counts": classifier.counts(), "sensor_color": bb["sensor_color"], "reflection": bb["reflection"],
            "jerk": round(bb["jerk"])}


def main():
    ap = argparse.ArgumentParser(description="LEGO sumo wrestler with a behavior tree")
    ap.add_argument("--sim", action="store_true", help="no hardware; keys e/c fake the sensors")
    ap.add_argument("--any-motor", action="store_true")
    ap.add_argument("--no-mqtt", action="store_true")
    ap.add_argument("--headless", action="store_true", help="no window (for the UNO Q); teleop over MQTT")
    ap.add_argument("--frames", type=int, default=0, help="stop after N ticks (testing)")
    args = ap.parse_args()
    headless = args.headless or cv2 is None

    bb = new_blackboard()
    classifier = knn.KNN(k=5, path=KNN_PATH)
    bb["_knn_counts"] = classifier.counts()
    running = [True]
    dm = cs = le = None
    if not args.sim:
        import legoeducation as le
        from lelib import doubleMotor, colorSensor
        dm = doubleMotor()
        print("Connecting to the Double Motor...")
        if args.any_motor:
            dm.connect()
        else:
            dm.connect(card_serial=CARD_SERIAL, card_color=getattr(le, f"LEGO_COLOR_{CARD_COLOR_NAME}"))
        cs = colorSensor()
        print("Connecting to the Color Sensor...")
        if args.any_motor:
            cs.connect(card_serial=None)
        else:
            cs.connect(card_serial=COLOR_CARD_SERIAL, card_color=getattr(le, f"LEGO_COLOR_{COLOR_CARD_NAME}"))
        print("Connected.")
        threading.Thread(target=sensor_loop, args=(cs, classifier, bb, running), daemon=True).start()
        threading.Thread(target=imu_loop, args=(dm, le, bb, running), daemon=True).start()
    motors = Motors(dm, le)

    def drive(l, r):
        if bb["manual"]:
            bb["tree_cmd"] = (int(l), int(r))      # what the tree wanted, for the HUD only
            return
        bb["cmd"] = (int(l), int(r))
        motors.drive(l, r)

    def manual_drive(bb):
        """Manual mode: keys straight to the wheels, nothing else. Out still stops."""
        if bb["out"] or not (bb["human"] and bb["now"] - bb["human_t"] <= KEY_HOLD):
            l = r = 0
        else:
            speed, turn = bb["human"]
            l, r = speed + turn, speed - turn
        bb["cmd"] = (int(l), int(r))
        motors.drive(l, r)

    client = None
    if not args.no_mqtt:
        from mqttlib import MQTTClient
        client = MQTTClient()
        client.connect()

        def on_bump(topic, payload):
            try:
                d = json.loads(payload)
            except ValueError:
                return
            bb["unoq_seen"] = time.monotonic()
            if d.get("bump"):
                bb["last_bump"] = time.monotonic()
        client.subscribe(BUMP_TOPIC, on_bump)

        def on_range(topic, payload):
            try:
                cm = float(json.loads(payload).get("cm"))
            except (ValueError, TypeError, AttributeError):
                return
            bb["range_cm"], bb["range_t"], bb["unoq_seen"] = cm, time.monotonic(), time.monotonic()
        client.subscribe(RANGE_TOPIC, on_range)

        def on_teleop(topic, payload):
            try:
                ch = json.loads(payload).get("key", "")
            except ValueError:
                ch = payload
            if ch:
                handle_key(str(ch)[:1], bb, classifier, args.sim)
        client.subscribe(TELEOP_TOPIC, on_teleop)
        print(f"state -> {STATE_TOPIC}   teleop <- {TELEOP_TOPIC}   result -> {RESULT_TOPIC}")

    tree = build_tree(drive)
    tree_lines = bt.describe(tree)
    print("\n".join(tree_lines))
    last_state_pub = 0.0
    n = 0
    try:
        while True:
            now = time.monotonic()
            bb["now"] = now
            bb["contact"] = now - bb["last_bump"] <= CONTACT_STALE
            fresh = bb["range_cm"] is not None and now - bb["range_t"] <= RANGE_STALE
            bb["opponent"] = bool(fresh and 2.0 < bb["range_cm"] < CHARGE_CM)
            bb["trace"] = []
            tree.tick(bb)
            if bb["manual"]:
                manual_drive(bb)
            n += 1

            if bb["out"] and not bb.get("announced"):
                bb["announced"] = True
                msg = f"{PLAYER} lost: {bb['out_reason']}"
                print("*** " + msg)
                if client is not None:
                    client.publish(RESULT_TOPIC, msg)
            if client is not None and now - last_state_pub >= 0.2:
                client.publish(STATE_TOPIC, json.dumps(state_message(bb, classifier)))
                last_state_pub = now

            if headless:
                if n % TICK_HZ == 0:
                    print(f"{'MANUAL ' if bb['manual'] else ''}doing {bb['active']:14s} floor {bb['floor_class']:5s} contact {bb['contact']!s:5s} "
                          f"human {bb['human'] if bb['human'] and now - bb['human_t'] <= KEY_HOLD else '-'}  wheels {bb['cmd']}")
                time.sleep(1.0 / TICK_HZ)
                if args.frames and n >= args.frames:
                    break
                continue

            cv2.imshow("sumo", draw_hud(bb, tree_lines, args.sim))
            key = cv2.waitKey(int(1000 / TICK_HZ)) & 0xFF
            if key in (27, ord("q")) or (args.frames and n >= args.frames):
                break
            arrows = {82: "w", 84: "s", 81: "a", 83: "d"}
            if key in arrows:
                handle_key(arrows[key], bb, classifier, args.sim)
            elif key != 255:
                handle_key(chr(key), bb, classifier, args.sim)
    finally:
        running[0] = False
        drive(0, 0)
        if client is not None:
            client.disconnect()
        for dev in (dm, cs):
            if dev is not None:
                try:
                    dev.disconnect()
                except Exception:
                    pass
        if cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
