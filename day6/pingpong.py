"""
Day 6 -- Virtual ping-pong, played with the LEGO Double Motor hub as the paddle.

    python pingpong.py                 # camera + hub + MQTT (the real thing)
    python pingpong.py --no-motor      # no hub: SPACE = swing, no haptics
    python pingpong.py --no-mqtt       # don't publish the score
    python pingpong.py --whistle       # serve on a high whistle instead of a timer

How it plays
    A ball flies across a side-view court toward your paddle. Your paddle's
    height is your WRIST, found by YOLO pose in the laptop camera (hold the
    hub in that hand). A hit needs two things at the moment the ball arrives:
      1. position  -- the paddle covers the ball's height (pose), and
      2. a swing   -- the hub's gyro saw a real swing in the last 0.4 s (IMU).
    Right place but no swing = miss ("no swing"). Swing but wrong place =
    miss ("out of reach"). Hits in a row are your streak; the best streak is
    the score, published as a float to ME193/Rogers/<name>.

AprilTags (print tags_sheet.png): tag 0 starts the game; tags 1/2/3 set the
level, i.e. how fast the ball crosses the court.

The opponent learns. Where it aims is chosen by tabular Q-learning (the
class's qlearn.py): state = which third of the court your paddle is in when
the ball leaves the wall, action = which third to aim at, reward = +1 if you
then miss, -1 if you hit. After a few rallies it aims where you are weakest.

Haptics (not done in class): the hub's motors pulse in your hand on a hit and
buzz on a miss, so you feel the ball without looking at the screen. The UNO Q
shows the live court on its LED matrix (unoq/pingpong-board).
"""

import argparse
import json
import math
import queue
import threading
import time

import cv2
import numpy as np

from qlearn import QTable
import sounds

# --- identity / MQTT --------------------------------------------------------
PLAYER = "ChrisLiang"
SCORE_TOPIC = f"ME193/Rogers/{PLAYER}"        # float: record number of continuous hits
LIVE_TOPIC = SCORE_TOPIC + "/live"            # JSON court state for the UNO Q, 10 Hz

# --- camera / tags ----------------------------------------------------------
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
START_TAG = 0
LEVELS = {1: ("EASY", 2.6), 2: ("MEDIUM", 1.7), 3: ("PRO", 1.1)}   # id -> (name, crossing time s)
DEFAULT_LEVEL = 1

# --- court / rules ----------------------------------------------------------
PADDLE_X = 0.92          # where the paddle plane sits (0 = far wall, 1 = right edge)
PADDLE_HALF = 0.15       # half height of the paddle, as a fraction of the court
SWING_GYRO = 300.0       # gyro magnitude (hub raw units) that counts as a swing
SWING_WINDOW = 0.60      # s before the ball arrives in which a swing counts
LATE_GRACE = 0.35        # s the ball waits at the paddle for a late swing (reaction time)
WRIST_TOP, WRIST_BOTTOM = 0.15, 0.80   # the band of the camera frame that maps onto the court
SERVE_DELAY = 1.5        # s after a miss before the next serve
POSE_CONF = 0.3          # wrist confidence needed to count as "hand visible"
POSE_SMOOTH = 0.6        # EMA weight on the newest wrist reading

# --- the hub (same card as Day 4) -------------------------------------------
CARD_COLOR_NAME = "ORANGE"
CARD_SERIAL = 7572

RED, GREEN, BLUE, YELLOW, WHITE, GREY = (0, 0, 255), (0, 200, 0), (255, 120, 0), (0, 220, 255), (255, 255, 255), (120, 120, 120)


# ============================================================================
# Inputs
# ============================================================================
class Swing:
    """IMU thread: watches the hub's gyro for swings."""

    def __init__(self):
        self.lock = threading.Lock()
        self.gyro = 0.0
        self.last_swing_t = -1e9
        self.peak = 0.0
        self.gesture = None

    def note(self, gyro_mag, gesture=None):
        with self.lock:
            self.gyro = gyro_mag
            self.gesture = gesture
            if gyro_mag >= SWING_GYRO:
                self.last_swing_t = time.monotonic()
                self.peak = max(self.peak, gyro_mag)

    def simulate(self):                 # SPACE key with --no-motor
        self.note(SWING_GYRO * 1.5)

    def recent(self, now, window=SWING_WINDOW):
        with self.lock:
            return now - self.last_swing_t <= window

    def level(self):
        with self.lock:
            return self.gyro


def imu_loop(dm, swing, running):
    while running[0]:
        d = dm.imu_device
        try:
            g = math.sqrt(d.gyroscopeX ** 2 + d.gyroscopeY ** 2 + d.gyroscopeZ ** 2)
        except (TypeError, AttributeError):
            g = float("nan")
        if not math.isnan(g):
            swing.note(g, getattr(dm.imu_gesture, "gesture", None))
        time.sleep(0.02)


class Haptics:
    """Pulses the hub's motors so you feel hits and misses. Runs in its own
    thread because BLE commands take a few ms each."""

    PATTERNS = {
        "hit":    [(70, 0.06), (-70, 0.06)],
        "miss":   [(45, 0.09), (-45, 0.09), (45, 0.09), (-45, 0.09)],
        "record": [(80, 0.05), (0, 0.05), (80, 0.05), (0, 0.05), (80, 0.08)],
    }

    def __init__(self, dm, le):
        self.dm, self.le, self.q = dm, le, queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def play(self, name):
        if self.dm is not None:
            self.q.put(name)

    def _run(self):
        le = self.le
        while True:
            name = self.q.get()
            for speed, dur in self.PATTERNS[name]:
                if speed:
                    self.dm.motor_run(motor=le.MOTOR_LEFT, speed=speed, blocking=False)
                    self.dm.motor_run(motor=le.MOTOR_RIGHT, speed=speed, blocking=False)
                time.sleep(dur)
            self.dm.motor_stop(motor=le.MOTOR_LEFT)
            self.dm.motor_stop(motor=le.MOTOR_RIGHT)


class Paddle:
    """Pose input: the wrist holding the hub -> paddle height (0 top .. 1 bottom)."""

    def __init__(self, hand):
        self.hand = hand            # 'left' / 'right' (screen side) / 'auto'
        self.y = None
        self.px = None              # wrist pixel for drawing

    def update(self, result, w, h):
        kp = result.keypoints
        if kp is None or len(kp) == 0 or kp.conf is None:
            self.y, self.px = None, None
            return
        areas = (result.boxes.xywh[:, 2] * result.boxes.xywh[:, 3]).cpu().numpy()
        i = int(np.argmax(areas))                       # the closest person
        xy, conf = kp.xy[i].cpu().numpy(), kp.conf[i].cpu().numpy()
        cands = [(9, conf[9]), (10, conf[10])]           # COCO: 9 left wrist, 10 right wrist
        if self.hand != "auto" and conf[0] > POSE_CONF:   # pick by screen side of the nose
            nose_x = xy[0][0]
            cands = [(k, c) for k, c in cands if (xy[k][0] > nose_x) == (self.hand == "right")] or cands
        k, c = max(cands, key=lambda t: t[1])
        if c < POSE_CONF:
            self.y, self.px = None, None
            return
        y = (xy[k][1] / h - WRIST_TOP) / (WRIST_BOTTOM - WRIST_TOP)   # comfortable hand range = whole court
        y = float(np.clip(y, 0, 1))
        self.y = y if self.y is None else POSE_SMOOTH * y + (1 - POSE_SMOOTH) * self.y
        self.px = (int(xy[k][0]), int(xy[k][1]))

    def zone(self):
        return 1 if self.y is None else min(ZONES - 1, int(self.y * ZONES))


ZONES = 3


# ============================================================================
# The game
# ============================================================================
class Game:
    WAITING, SERVING, FLYING, ARRIVED, RETURNING = "WAITING", "SERVING", "FLYING", "ARRIVED", "RETURNING"

    def __init__(self, rng_seed=None):
        self.state = self.WAITING
        self.level = DEFAULT_LEVEL
        self.streak = 0
        self.record = 0
        self.ball = [0.0, 0.5, 0.0, 0.0]          # x, y, vx, vy
        self.serve_at = 0.0
        self.arrived_at = 0.0
        self.message, self.message_t = "show the START tag", 0.0
        self.q = QTable(np.arange(ZONES), np.arange(ZONES), gamma=0.0, alpha=0.4)
        self.rng = np.random.default_rng(rng_seed)
        self.epsilon = 0.25
        self.last_sa = None                       # (state, action) of the flight in the air
        self.wait_for_whistle = False
        self.hits = self.misses = 0

    # ---- helpers
    def crossing_time(self):
        return LEVELS[self.level][1]

    def say(self, text, now):
        self.message, self.message_t = text, now

    def start(self, now):
        if self.state == self.WAITING:
            self.streak = 0
            self.state = self.SERVING
            self.serve_at = now + 0.8
            self.say("game on!", now)

    def set_level(self, level, now):
        if level in LEVELS and level != self.level:
            self.level = level
            self.say(f"level {level}: {LEVELS[level][0]}", now)

    def reset(self, now):
        self.__init__()
        self.say("reset", now)

    # ---- the opponent: Q-learning picks where to aim
    def serve(self, paddle_zone, now, from_y=None):
        s = paddle_zone
        a = self.q.choose_action(s, self.epsilon)
        self.last_sa = (s, a)
        y0 = 0.5 if from_y is None else from_y
        target = (a + 0.5) / ZONES + self.rng.uniform(-0.07, 0.07)
        target = float(np.clip(target, 0.10, 0.90))
        T = self.crossing_time()
        self.ball = [0.0, y0, PADDLE_X / T, (target - y0) / T]
        self.state = self.FLYING

    def learn(self, player_missed):
        if self.last_sa is None:
            return
        s, a = self.last_sa
        self.q.rewards[s, a] = 1.0 if player_missed else -1.0      # the opponent's reward
        self.q.bellman_update(s, a, s)
        self.last_sa = None

    # ---- outcomes
    def _hit(self, paddle_y, now):
        self.learn(player_missed=False)
        self.streak += 1
        self.hits += 1
        events = ["hit"]
        if self.streak > self.record:
            self.record = self.streak
            events.append("record")
        self.say(f"HIT!  streak {self.streak}", now)
        offset = (self.ball[1] - paddle_y) / PADDLE_HALF            # -1 top edge .. +1 bottom edge
        T = self.crossing_time()
        y_wall = float(np.clip(self.ball[1] + offset * 0.35, 0.10, 0.90))
        self.ball = [PADDLE_X, self.ball[1], -PADDLE_X / T, (y_wall - self.ball[1]) / T]
        self.state = self.RETURNING
        return events

    def _miss(self, why, now):
        self.learn(player_missed=True)
        self.streak = 0
        self.misses += 1
        self.say(f"MISS - {why}", now)
        self.state = self.SERVING
        self.serve_at = now + SERVE_DELAY
        return ["miss"]

    # ---- one step; returns a list of events for sound/haptics/MQTT
    def update(self, dt, paddle_y, swing_recent, now, whistle=False):
        events = []
        if self.state == self.WAITING:
            self.ball[0] = 0.5 + 0.03 * math.sin(now * 2)
            self.ball[1] = 0.5
            return events
        if self.state == self.SERVING:
            ready = now >= self.serve_at and (not self.wait_for_whistle or whistle)
            if ready:
                zone = 1 if paddle_y is None else min(ZONES - 1, int(paddle_y * ZONES))
                self.serve(zone, now)
                events.append("serve")
            return events

        b = self.ball
        b[0] += b[2] * dt
        b[1] += b[3] * dt
        if b[1] < 0.0 or b[1] > 1.0:               # bounce off top/bottom
            b[1] = float(np.clip(b[1], 0.0, 1.0))
            b[3] = -b[3]

        if self.state == self.FLYING and b[0] >= PADDLE_X:
            b[0] = PADDLE_X
            in_reach = paddle_y is not None and abs(b[1] - paddle_y) <= PADDLE_HALF
            if in_reach and swing_recent:
                events += self._hit(paddle_y, now)
            elif in_reach:
                self.state, self.arrived_at = self.ARRIVED, now
            elif paddle_y is None:
                events += self._miss("hand not visible", now)
            else:
                gap = b[1] - paddle_y
                events += self._miss(f"out of reach - paddle {abs(gap) / PADDLE_HALF:.1f} paddles too {'high' if gap > 0 else 'low'}", now)
        elif self.state == self.ARRIVED:
            in_reach = paddle_y is not None and abs(b[1] - paddle_y) <= PADDLE_HALF
            if in_reach and swing_recent:
                events += self._hit(paddle_y, now)
            elif not in_reach:
                events += self._miss("moved away before swinging", now)
            elif now - self.arrived_at > LATE_GRACE:
                events += self._miss(f"no swing (waited {LATE_GRACE:.2f} s)", now)
        elif self.state == self.RETURNING and b[0] <= 0.0:
            events.append("wall")
            zone = 1 if paddle_y is None else min(ZONES - 1, int(paddle_y * ZONES))
            self.serve(zone, now, from_y=b[1])
        return events

    def live(self, paddle_y):
        return {"s": self.streak, "r": self.record, "st": self.state[:3],
                "bx": round(self.ball[0], 3), "by": round(self.ball[1], 3),
                "py": None if paddle_y is None else round(paddle_y, 3), "lv": self.level}


# ============================================================================
# Drawing
# ============================================================================
PANEL_W, PANEL_H = 560, 540
COURT = (40, 70, 520, 400)          # x1, y1, x2, y2 inside the panel


def court_xy(x, y):
    x1, y1, x2, y2 = COURT
    return int(x1 + x * (x2 - x1)), int(y1 + y * (y2 - y1))


def draw_panel(game, paddle, swing, now, motor_on, whistle_level=None):
    p = np.full((PANEL_H, PANEL_W, 3), 25, np.uint8)
    x1, y1, x2, y2 = COURT
    cv2.rectangle(p, (x1, y1), (x2, y2), (60, 60, 60), -1)
    for z in range(1, ZONES):                                   # the three zones
        yy = int(y1 + z * (y2 - y1) / ZONES)
        cv2.line(p, (x1, yy), (x2, yy), (80, 80, 80), 1)
    cv2.line(p, (x1, y1), (x1, y2), WHITE, 4)                    # the wall
    px = court_xy(PADDLE_X, 0)[0]
    if paddle.y is not None:                                     # the paddle
        top, bot = court_xy(PADDLE_X, paddle.y - PADDLE_HALF)[1], court_xy(PADDLE_X, paddle.y + PADDLE_HALF)[1]
        cv2.rectangle(p, (px - 6, top), (px + 6, bot), GREEN if swing.recent(now) else YELLOW, -1)
    else:
        cv2.putText(p, "hand not visible", (px - 150, y2 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 1, cv2.LINE_AA)
    bx, by = court_xy(game.ball[0], game.ball[1])                # the ball
    cv2.circle(p, (bx, by), 9, WHITE, -1)

    # swing meter
    mx, my, mh = x2 + 14, y1, y2 - y1
    frac = min(1.4, swing.level() / SWING_GYRO)
    cv2.rectangle(p, (mx, my), (mx + 14, my + mh), (50, 50, 50), -1)
    cv2.rectangle(p, (mx, my + mh - int(mh * frac / 1.4)), (mx + 14, my + mh), GREEN if frac >= 1 else BLUE, -1)
    thr = my + mh - int(mh * 1 / 1.4)
    cv2.line(p, (mx - 3, thr), (mx + 17, thr), RED, 2)
    cv2.putText(p, "swing", (mx - 8, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)

    # header
    lv = LEVELS[game.level][0]
    cv2.putText(p, f"streak {game.streak}", (40, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
    cv2.putText(p, f"record {game.record}", (250, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, YELLOW, 2, cv2.LINE_AA)
    cv2.putText(p, f"{lv}", (450, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2, cv2.LINE_AA)
    cv2.putText(p, game.state, (40, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1, cv2.LINE_AA)
    age = now - game.message_t
    if age < 2.5 or game.state in (game.WAITING, game.SERVING):
        col = GREEN if game.message.startswith("HIT") else (RED if game.message.startswith("MISS") else WHITE)
        cv2.putText(p, game.message, (40, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)

    # the opponent's Q-table: rows = where your paddle was, cols = where it aimed
    gx, gy, cell = 300, 430, 26
    cv2.putText(p, "opponent's Q-table  (aim ->)", (gx - 120, gy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1, cv2.LINE_AA)
    for s in range(ZONES):
        for a in range(ZONES):
            v = float(game.q.table[s, a])
            t = max(-1.0, min(1.0, v))
            color = (0, int(120 + 100 * t), 0) if t >= 0 else (0, 0, int(120 - 100 * t))
            cv2.rectangle(p, (gx + a * cell, gy + s * cell), (gx + (a + 1) * cell - 2, gy + (s + 1) * cell - 2), color, -1)
            cv2.putText(p, f"{v:+.1f}", (gx + a * cell + 1, gy + s * cell + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.32, WHITE, 1, cv2.LINE_AA)
    for s, name in enumerate(("top", "mid", "low")):
        cv2.putText(p, name, (gx - 34, gy + s * cell + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1, cv2.LINE_AA)
    cv2.putText(p, f"rallies: {game.hits} hits / {game.misses} misses", (gx + 100, gy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1, cv2.LINE_AA)
    cv2.putText(p, "IMU: hub gyro" if motor_on else "IMU off: SPACE = swing", (gx + 100, gy + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1, cv2.LINE_AA)
    if whistle_level is not None:
        cv2.putText(p, f"whistle level {whistle_level:.0f}", (gx + 100, gy + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1, cv2.LINE_AA)
    cv2.putText(p, "tag0 start  tag1/2/3 level   q quit  r reset", (40, PANEL_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)
    return p


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    for _ in range(40):
        ok, f = cap.read()
        if ok and f is not None:
            return cap
        time.sleep(0.05)
    raise RuntimeError("camera produced no frames (check Privacy & Security -> Camera)")


# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="virtual ping-pong with the LEGO hub as paddle")
    ap.add_argument("--no-motor", action="store_true", help="no hub: SPACE swings, no haptics")
    ap.add_argument("--any-motor", action="store_true", help="connect to the first Double Motor found")
    ap.add_argument("--no-mqtt", action="store_true")
    ap.add_argument("--whistle", action="store_true", help="serve on a high whistle")
    ap.add_argument("--hand", choices=["auto", "left", "right"], default="auto",
                    help="which wrist is the paddle (screen side); auto = the more confident one")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (testing)")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pose = YOLO("yolov8n-pose.pt")
    tags = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                   cv2.aruco.DetectorParameters())

    swing = Swing()
    running = [True]
    dm = le = None
    if not args.no_motor:
        import legoeducation as le
        from lelib import doubleMotor
        dm = doubleMotor()
        print("Connecting to the Double Motor (the paddle)...")
        if args.any_motor:
            dm.connect()
        else:
            dm.connect(card_serial=CARD_SERIAL, card_color=getattr(le, f"LEGO_COLOR_{CARD_COLOR_NAME}"))
        print("Connected.")
        threading.Thread(target=imu_loop, args=(dm, swing, running), daemon=True).start()
    haptics = Haptics(dm, le)

    client = None
    if not args.no_mqtt:
        from mqttlib import MQTTClient
        client = MQTTClient()
        client.connect()
        print(f"score -> {SCORE_TOPIC}   live court -> {LIVE_TOPIC}")

    listener = None
    if args.whistle:
        from whistle import WhistleListener
        listener = WhistleListener()

    cam_index = CAMERA_INDEX if args.camera is None else args.camera
    try:                                   # prefer the laptop's own camera over a nearby iPhone
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "day5"))
        import common as day5_common
        cams = day5_common.list_cameras()
        builtin = [i for i, _, b in cams if b]
        if args.camera is None and builtin:
            cam_index = builtin[0]
    except Exception:
        pass
    cap = open_camera(cam_index)

    game = Game()
    game.wait_for_whistle = args.whistle
    paddle = Paddle(args.hand)
    last_t = time.monotonic()
    last_live = last_score = 0.0
    last_published_record = None
    n = 0
    fps, fps_t = 0.0, time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            n += 1
            now = time.monotonic()
            dt, last_t = min(0.1, now - last_t), now

            # --- tags on the raw frame (a mirrored tag doesn't decode), then mirror for display
            corners, ids, _ = tags.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            seen = []
            if ids is not None:
                for c, tid in zip(corners, ids.flatten()):
                    pts = c.reshape(-1, 2).copy()
                    pts[:, 0] = w - 1 - pts[:, 0]
                    seen.append((int(tid), pts))
                    if int(tid) == START_TAG:
                        game.start(now)
                    elif int(tid) in LEVELS:
                        game.set_level(int(tid), now)

            # --- pose -> paddle
            r = pose.predict(frame, imgsz=640, device=device, conf=0.4, verbose=False)[0]
            paddle.update(r, w, h)

            # --- game step
            whistled = listener.pop() if listener else False
            events = game.update(dt, paddle.y, swing.recent(now), now, whistled)
            for e in events:
                if e == "hit":
                    sounds.play("ping"); haptics.play("hit")
                elif e == "record":
                    sounds.play("record"); haptics.play("record")
                elif e == "miss":
                    sounds.play("miss"); haptics.play("miss")
                elif e == "wall":
                    sounds.play("pong")
                elif e == "serve":
                    sounds.play("serve")

            # --- MQTT: the score (record streak) as a float, and the live court
            if client is not None:
                if game.record != last_published_record or now - last_score > 5.0:
                    client.publish(SCORE_TOPIC, f"{float(game.record):.1f}")
                    last_published_record, last_score = game.record, now
                if now - last_live >= 0.1:
                    client.publish(LIVE_TOPIC, json.dumps(game.live(paddle.y)))
                    last_live = now

            # --- draw
            view = r.plot(boxes=False)
            for tid, pts in seen:
                cv2.polylines(view, [pts.astype(int).reshape(-1, 1, 2)], True, RED, 3)
                label = "START" if tid == START_TAG else (f"LEVEL {tid}" if tid in LEVELS else f"tag {tid}")
                cv2.putText(view, label, (int(pts[0][0]), int(pts[0][1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2, cv2.LINE_AA)
            if paddle.px is not None:
                cv2.circle(view, paddle.px, 14, GREEN if swing.recent(now) else YELLOW, 3)
                cv2.line(view, (0, paddle.px[1]), (w, paddle.px[1]), YELLOW, 1)
            if n % 10 == 0:
                fps, fps_t = 10 / max(1e-6, now - fps_t), now
            cv2.putText(view, f"{fps:.0f} fps   paddle = your wrist   q quit", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2, cv2.LINE_AA)
            cam = cv2.resize(view, (960, PANEL_H))
            panel = draw_panel(game, paddle, swing, now, dm is not None, listener.level if listener else None)
            cv2.imshow("ping-pong", np.hstack([cam, panel]))

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")) or (args.frames and n >= args.frames):
                break
            if key == ord(" "):
                swing.simulate()
            if key == ord("s"):
                game.start(now)
            if key in (ord("1"), ord("2"), ord("3")):
                game.set_level(int(chr(key)), now)
            if key == ord("r"):
                game.reset(now)
    finally:
        running[0] = False
        if client is not None:
            client.publish(SCORE_TOPIC, f"{float(game.record):.1f}")
            client.disconnect()
        if listener:
            listener.close()
        if dm is not None:
            try:
                dm.motor_stop(motor=le.MOTOR_LEFT); dm.motor_stop(motor=le.MOTOR_RIGHT)
                dm.disconnect()
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()
        print(f"final record: {game.record} continuous hits")


if __name__ == "__main__":
    main()
