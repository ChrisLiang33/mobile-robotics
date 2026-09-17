# Day 3 — AprilTag Center-Seeking Car

A LEGO car with a printed AprilTag on a tower drives back and forth in front of
the computer, parallel to the screen. OpenCV finds the tag in the webcam feed
every frame and a PD controller drives the wheels until the tag sits in the
middle of the camera image — fast when it's far from center, slowing down as it
gets close, stopped when centered. The live feed draws a **red outline** around
the tag and prints the **centroid coordinates** next to it.

## Files

| File | What it is |
|------|------------|
| `apriltag_car.py` | Main program: camera + detection + PD controller + motor commands |
| `generate_apriltag.py` | Makes the printable tag PNG (`tag36h11_id0.png`) |
| `tag36h11_id0.png` | The tag — print at 100% scale (70 mm) and tape it to the car's tower |
| `camlib.py`, `lelib.py` | Course helper libraries (camera picker, LEGO wrappers) copied from the class repo |

## Setup

```bash
pip install legoeducation opencv-python
```

(or use the class venv that already has both: `source ~/Desktop/ME193-Robotics/my_env/bin/activate`)

1. `python generate_apriltag.py`, print the PNG at 100% scale, tape it to the
   tower facing the camera. Keep the white margin — the detector needs it.
2. Turn on the Double Motor drive base.
3. `python apriltag_car.py` — pick your camera when prompted. Use
   `--no-robot` to test the vision/controller with no car connected.

Keys in the video window: **s** toggles spring mode, **q**/Esc quits (motors
stop on exit). If the car drives *away* from center, flip the `DIRECTION`
constant at the top of `apriltag_car.py`; wheel-motor mounting is handled by
`LEFT_SIGN` / `RIGHT_SIGN`.

## Q1 — Describe the controller (policy): how does it determine motor speed?

It's a **PD controller on the tag's horizontal pixel position**, evaluated once
per camera frame (~30 Hz):

```
err   = (tag_centroid_x − frame_center_x) / (frame_width / 2)   # normalized to [−1, +1]
speed = Kp·err + Kd·d(err)/dt                                   # clamped to ±MAX_SPEED
```

That one signed `speed` (in percent) is sent to both wheel motors with
`motor_run(..., blocking=False)` — one motor's sign is flipped because the two
motors are mounted mirror-image on the chassis. There is no position-move
command anywhere; the tracking behavior comes entirely from the P and D terms.

- **P term** (`Kp·err`): command is proportional to how far the tag is from the
  centerline. This is exactly the "fancier" behavior asked for — the car is
  automatically **fast when far away and slow when close**, because the error
  itself shrinks as it approaches center. It's a virtual spring pulling the tag
  toward the middle of the image.
- **D term** (`Kd·d(err)/dt`): the damper. It opposes the tag's velocity across
  the image, so the car brakes as it comes into center instead of flying past.
  The derivative is low-pass filtered (`DERIV_ALPHA`) because pixel centroids
  are noisy. Unlike Day 2's hand-turned dial, the setpoint here (frame center)
  never moves, so derivative-on-error and derivative-on-measurement are the
  same thing — no derivative kick.
- **Deadband** (`|err| < 0.035`): within ~3.5% of center the command is forced
  to 0, so the car doesn't twitch forever over one-pixel noise. That's what
  makes it *stop* in the middle rather than hunt.
- **MIN_SPEED floor**: just outside the deadband, tiny P commands get bumped up
  to the smallest speed that actually moves the car, so static friction can't
  strand it just short of center.

## Q2 — What does the code do if no tag is detected?

Two stages, both visible on the HUD:

1. **Grace period** (`LOST_GRACE_S = 0.4 s`): the last command is held, so a
   single dropped frame (motion blur mid-crossing, brief occlusion) doesn't
   make the car stutter.
2. **After that: full stop.** Both motors get `motor_stop()` and the HUD shows
   `TAG LOST — STOPPED`. The car sits still until the tag is re-detected, then
   control resumes seamlessly. The derivative history is also reset so the D
   term doesn't spike across the gap.

The motors are additionally stopped in a `finally:` block, so quitting with
**q**, closing the window, or even a crash never leaves the car driving.

## Q3 — Can you make it overshoot the center and come back (spring-loaded)?

Yes — press **s**. Spring mode swaps the gains from (`Kp=65, Kd=18`) to
(`Kp=160, Kd=0`): a **stiff spring with the damper removed**. A pure P
controller is literally Hooke's law, `command = −k·x`, with `x` the distance
from center. The car has mass (inertia) plus camera/BLE latency, and with no D
term nothing removes energy on the way in — so it's an **underdamped
mass-spring system**: it charges through center, the now-reversed error yanks
it back, it overshoots the other way, and it rings back and forth with
shrinking amplitude until friction and the deadband finally capture it at
center. Press **s** again to restore the damped gains and watch the same
approach glide in with no overshoot — the two modes side by side are the whole
P-vs-D story from Day 2, now on wheels.
