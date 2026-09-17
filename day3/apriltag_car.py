"""
Day 3 -- AprilTag "stop in the middle of the screen" car.

The car (LEGO Double Motor drive base) carries a printed AprilTag on a
tower and drives back and forth parallel to the computer screen. The
webcam finds the tag every frame, and a PD controller on the tag's
horizontal position drives the wheels so the tag settles on the vertical
centerline of the camera image. Because the command is proportional to
the distance from center, the car is automatically fast when it is far
off-center and slow as it gets close.

Live feed shows the tag with a RED outline, its centroid coordinates,
the pixel error, and the commanded speed.

Run:
    python apriltag_car.py              # full demo (connects to the car)
    python apriltag_car.py --no-robot   # vision + controller only, no BLE

Keys (with the video window focused):
    s        toggle SPRING mode (high Kp, zero Kd -> overshoots and
             comes back, ringing like a mass on a spring)
    q / Esc  quit (motors are stopped on the way out)

Controller (the "policy"), evaluated once per camera frame:
    err   = (tag_cx - frame_cx) / (frame_cx)        # normalized to [-1, 1]
    speed = Kp * err + Kd * d(err)/dt               # then clamped to +-MAX_SPEED
The same signed speed goes to both wheel motors (one negated, because the
motors are mounted mirror-image on the chassis). The target (the frame
center) never moves, so differentiating the error is the same as
differentiating the measurement (up to sign) -- there is no derivative
kick here, unlike Day 2's hand-turned dial. Inside a small deadband
around center the command is forced to 0 so the car doesn't buzz over
pixel noise; just outside it, commands are bumped up to MIN_SPEED so
static friction can't leave the car stalled short of center.

If NO tag is detected: the last command is held for LOST_GRACE_S seconds
(so a single dropped frame mid-crossing doesn't stutter the car), after
which the motors are stopped and the HUD shows TAG LOST until the tag is
seen again. Motors are also stopped in a finally-block on quit or crash.
"""

import argparse
import time
import legoeducation as le

import cv2

# --- Bluetooth card info (same pattern as Day 2) ---------------------------
# None = connect to the first advertising Double Motor found.
CARD_COLOR = le.LEGO_COLOR_ORANGE
CARD_SERIAL = 1142

# --- Vision ----------------------------------------------------------------
TAG_DICT = cv2.aruco.DICT_APRILTAG_36h11
CAMERA_INDEX = 0      # 0 = built-in FaceTime camera
FRAME_W, FRAME_H = 1280, 720

# --- Controller gains ------------------------------------------------------
# Damped mode: enough Kd that the car glides into center with little or no
# overshoot. Spring mode: stiff spring, no damper -- deliberately
# underdamped so it shoots past center and oscillates back.
KP_DAMPED, KD_DAMPED = 65.0, 18.0
KP_SPRING, KD_SPRING = 160.0, 0.0

MAX_SPEED = 75        # clamp on the wheel command, percent
MIN_SPEED = 12        # smallest command that actually moves the car
DEADBAND = 0.035      # |err| below this counts as "centered" -> command 0
DERIV_ALPHA = 0.4     # low-pass factor for d(err)/dt (pixel noise is spiky)
LOST_GRACE_S = 0.4    # keep last command this long after losing the tag

# --- Drivetrain signs ------------------------------------------------------
# Flip DIRECTION if the car drives away from center instead of toward it.
# LEFT/RIGHT signs make one positive "speed" move the whole car one way
# even though the two motors are mounted mirror-image.
DIRECTION = +1
LEFT_SIGN = +1
RIGHT_SIGN = -1

RED = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (0, 220, 255)


class Car:
    """Thin wrapper so the vision loop can run with --no-robot for testing."""

    def __init__(self, enabled):
        self.enabled = enabled
        self.dm = None
        self._le = None
        self._last_cmd = 0

    def connect(self):
        if not self.enabled:
            print("--no-robot: skipping BLE, vision + controller only.")
            return
        import legoeducation as le
        from lelib import doubleMotor
        self._le = le
        self.dm = doubleMotor()
        print("Connecting to Double Motor...")
        self.dm.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
        print("Connected.")

    def drive(self, speed):
        """Send one signed speed (percent) to both wheels, non-blocking."""
        s = int(round(speed))
        if not self.enabled:
            self._last_cmd = s
            return
        if s == 0:
            self.stop()
            return
        le = self._le
        self.dm.motor_run(motor=le.MOTOR_LEFT, speed=LEFT_SIGN * s, blocking=False)
        self.dm.motor_run(motor=le.MOTOR_RIGHT, speed=RIGHT_SIGN * s, blocking=False)
        self._last_cmd = s

    def stop(self):
        if self.enabled and self._last_cmd != 0:
            le = self._le
            self.dm.motor_stop(motor=le.MOTOR_LEFT)
            self.dm.motor_stop(motor=le.MOTOR_RIGHT)
        self._last_cmd = 0

    def disconnect(self):
        if self.enabled and self.dm is not None:
            self.stop()
            self.dm.disconnect()


def open_camera():
    """Open the first camera that actually delivers frames.

    "Opened" is not enough on macOS: a Continuity Camera (iPhone) that isn't
    streaming, or a terminal without camera permission, opens fine but every
    read fails -- so each candidate must prove itself with a real frame.
    Tries CAMERA_INDEX first, then the other low indices.
    """
    indices = [CAMERA_INDEX] + [i for i in range(4) if i != CAMERA_INDEX]
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        for _ in range(40):  # up to ~2 s of sensor warm-up
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Using camera {idx}")
                return cap
            time.sleep(0.05)
        print(f"Camera {idx} opened but produced no frames, trying next...")
        cap.release()
    raise RuntimeError(
        "No camera delivered any frames. Check System Settings -> Privacy & "
        "Security -> Camera and allow your terminal app (then restart it), "
        "or set CAMERA_INDEX at the top of this file."
    )


def largest_tag(corners):
    """Return (index, centroid, corner array) of the biggest detected tag,
    so a stray second tag in the background can't steal the controller."""
    best_i, best_area = 0, -1.0
    for i, c in enumerate(corners):
        area = cv2.contourArea(c.reshape(-1, 2))
        if area > best_area:
            best_i, best_area = i, area
    pts = corners[best_i].reshape(-1, 2)
    cx, cy = pts.mean(axis=0)
    return best_i, (float(cx), float(cy)), pts


def main():
    ap = argparse.ArgumentParser(description="AprilTag center-seeking car")
    ap.add_argument("--no-robot", action="store_true",
                    help="run vision + controller without connecting to the car")
    args = ap.parse_args()

    car = Car(enabled=not args.no_robot)
    car.connect()

    dictionary = cv2.aruco.getPredefinedDictionary(TAG_DICT)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    cap = open_camera()

    spring = False
    prev_err = None
    prev_t = None
    derr_f = 0.0
    last_seen_t = -1e9
    last_cmd = 0.0

    try:
        bad_reads = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                bad_reads += 1
                if bad_reads > 30:      # ~1 s of dead air = camera is gone
                    print("Camera stopped delivering frames -- exiting.")
                    break
                car.stop()               # don't keep driving blind
                time.sleep(0.03)
                continue
            bad_reads = 0
            now = time.monotonic()
            h, w = frame.shape[:2]
            half_w = w / 2.0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                ids = ids.flatten()  # OpenCV 4.x gives (N,1), 5.x gives (N,)

            kp, kd = (KP_SPRING, KD_SPRING) if spring else (KP_DAMPED, KD_DAMPED)

            # Center line and "close enough" deadband zone
            band = int(DEADBAND * half_w)
            cv2.line(frame, (w // 2, 0), (w // 2, h), YELLOW, 1)
            cv2.rectangle(frame, (w // 2 - band, 0), (w // 2 + band, h), YELLOW, 1)

            if ids is not None and len(corners) > 0:
                i, (cx, cy), pts = largest_tag(corners)

                # Red outline + centroid, as required
                cv2.polylines(frame, [pts.astype(int).reshape(-1, 1, 2)], True, RED, 3)
                cv2.circle(frame, (int(cx), int(cy)), 5, RED, -1)
                cv2.putText(frame, f"centroid ({cx:.0f}, {cy:.0f})  id {int(ids[i])}",
                            (int(cx) + 10, int(cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)

                err = (cx - half_w) / half_w          # normalized [-1, 1]

                # Filtered derivative of the error
                if prev_err is not None and prev_t is not None and now > prev_t:
                    derr = (err - prev_err) / (now - prev_t)
                    derr_f = DERIV_ALPHA * derr + (1.0 - DERIV_ALPHA) * derr_f
                prev_err, prev_t = err, now

                if abs(err) < DEADBAND:
                    cmd = 0.0                          # centered: hold still
                else:
                    cmd = kp * err + kd * derr_f
                    if 0 < abs(cmd) < MIN_SPEED:       # beat static friction
                        cmd = MIN_SPEED if cmd > 0 else -MIN_SPEED
                cmd = max(-MAX_SPEED, min(MAX_SPEED, cmd)) * DIRECTION

                car.drive(cmd)
                last_cmd = cmd
                last_seen_t = now

                cv2.putText(frame, f"err {err:+.3f} ({cx - half_w:+.0f}px)   speed {cmd:+.0f}%",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
            else:
                prev_err = None                        # don't differentiate across a gap
                if now - last_seen_t < LOST_GRACE_S:
                    car.drive(last_cmd)                # ride out a dropped frame
                    status = f"tag lost - holding {last_cmd:+.0f}% for a moment"
                else:
                    car.stop()                         # fail safe: stop and wait
                    last_cmd = 0.0
                    status = "TAG LOST - STOPPED"
                cv2.putText(frame, status, (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2, cv2.LINE_AA)

            mode = "SPRING (underdamped)" if spring else "DAMPED"
            cv2.putText(frame, f"[{mode}]  Kp={kp:g} Kd={kd:g}   s: toggle spring   q: quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2, cv2.LINE_AA)

            cv2.imshow("AprilTag car", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('s'):
                spring = not spring
                print(f"Spring mode {'ON' if spring else 'OFF'}")
    finally:
        car.disconnect()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
