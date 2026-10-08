# Day 6 — Virtual Ping-Pong with a LEGO Hub as the Paddle

*ME193 final project — Chris Liang*

You hold the LEGO Double Motor hub like a paddle. A ball flies across a
virtual court on the laptop screen; your **wrist**, found by a pose network in
the camera, is the paddle's height, and the hub's **gyro** says whether you
really swung. Both have to be right at the moment the ball arrives. The ball
comes back faster at higher levels, the opponent **learns where you miss**,
and the hub **buzzes in your hand** on every hit and miss. Your score — the
record number of continuous hits — is published live as a float to
`ME193/Rogers/ChrisLiang`, and the UNO Q's LED matrix shows the court.

## Everything from the course, in one game

| Class topic | Where it is in the game |
|---|---|
| OpenCV camera + AprilTags (Day 3) | tag 0 starts the game, tags 1/2/3 set the level (ball speed) |
| PD/P control & LEGO motors (Days 2–3) | the hub's motors as **haptic feedback** on hits and misses |
| IMU (Day 2 lelib) | the hub's gyro detects a swing; swing meter on screen |
| PyAudio FFT whistle (Day 4) | `--whistle`: serve on a high whistle; game sounds through PyAudio |
| MQTT (Days 4–5) | score to `ME193/Rogers/ChrisLiang`, live court to `.../live` |
| YOLO neural nets (Day 5) | YOLOv8-pose finds your wrist = paddle position |
| UNO Q + LED matrix (Day 5) | `unoq/pingpong-board` draws the live court and streak |
| Q-learning (class QLearn demo) | the opponent picks where to aim with a Q-table |

**New this time:** haptic feedback through the hub's motors, a learning
opponent, and the LED matrix as a physical mini-court.

## Files

| File | What it is |
|------|------------|
| `pingpong.py` | the game: camera, pose, tags, IMU thread, physics, Q-learning opponent, haptics, sounds, MQTT, display |
| `qlearn.py` | the class's Q-table (Bellman update, ε-greedy) — used unchanged |
| `whistle.py`, `sounds.py` | Day 4's FFT whistle detector; PyAudio game sounds |
| `generate_tags.py`, `tags_sheet.png` | the four AprilTag cards (print at 100%) |
| `unoq/pingpong-board/` | UNO Q App Lab project: LED-matrix court; `unoq/unoq.sh` deploys it |
| `lelib.py`, `mqttlib.py` | course libraries |

## Setup and running

```bash
pip install ultralytics opencv-python paho-mqtt numpy pyaudio legoeducation
python generate_tags.py          # print tags_sheet.png, cut out the four cards
python pingpong.py               # the real thing: hub + camera + MQTT
python pingpong.py --no-motor    # no hub: SPACE = swing (for testing)
python pingpong.py --whistle     # serve on a high whistle
```

Hold the hub in one hand, in view of the camera, and show **tag 0** to
start. Move your hand up and down to meet the ball, and **swing** as it
arrives. Show **tag 1, 2 or 3** at any time to change the level. Keys: `q`
quit, `r` reset, `1/2/3` level, `s` start (if you lose the tag).

UNO Q scoreboard: `cd unoq && UNOQ_HOST=<board-ip> ./unoq.sh deploy`.

## Q1 — Describe the policy: how does it make decisions?

There are two decision-makers.

**The referee** decides hit or miss when the ball reaches the paddle plane,
from two independent sensors:

```
in_reach = |ball_y − wrist_y| ≤ 0.11        # pose: paddle covers the ball's height
swung    = a gyro peak ≥ 300 in the last 0.40 s   # IMU: a real swing
```
Both true → **hit** (streak +1, the ball returns at an angle set by where it
met the paddle). In reach but no swing yet → the ball waits 0.12 s for a late
swing, then **miss, "no swing"**. Swing but not in reach → **miss, "out of
reach"**; wrist not visible → **miss, "hand not visible"**. A miss resets the
streak to 0; the record streak is the published score.

**The opponent** decides where to aim each serve with tabular Q-learning:
- *state* = which third of the court your paddle is in when the ball leaves
  the wall (top / middle / bottom);
- *action* = which third to aim at;
- *reward* = +1 if you then miss, −1 if you hit;
- ε-greedy with ε = 0.25, learning rate 0.4, no discount (each rally is its
  own episode).
After a few rallies the Q-table shows where you're weak (green cells on
screen) and the opponent aims there most of the time, with 25% random shots
to keep exploring. The level tag sets how fast the ball crosses (2.6 / 1.7 /
1.1 s). The game only starts when the START tag has been seen.

## Q2 — What are the potential limitations of your game?

- **It's one-dimensional.** Pose gives a 2-D wrist position, but the court
  only uses height; left/right motion of your hand does nothing.
- **The swing is "any fast motion".** The gyro threshold can't tell a
  forehand from a backhand, or a swing from shaking the hub; a cheat is to
  wiggle it continuously. Direction-aware swing detection (sign of gyro Z)
  would fix that.
- **Latency.** Camera → pose → decision is ~50 ms, hub IMU over Bluetooth
  adds more, and MQTT to the public broker adds ~100 ms before the UNO Q
  shows anything. At level 3 the ball crosses in 1.1 s, so reaction time
  matters and the matrix lags the screen.
- **One player, one hand.** YOLO picks the closest person and the more
  confident wrist; a second person in frame, or holding the hub in the other
  hand, confuses it unless `--hand` is set.
- **Lighting and distance.** Pose confidence drops when the wrist is near
  the frame edge, far away, or in dim light; then the paddle disappears
  ("hand not visible" = automatic miss).
- **Haptics need the motors attached** and spinning freely; a loaded motor
  (wheels on the floor) turns the buzz into a lurch.
- **A shared public broker**: anyone can publish to the score topic.

## Q3 — What AI/ML algorithms did you use, and how do they work?

- **YOLOv8-pose (convolutional neural network).** One pass of a CNN over the
  frame predicts, for each person, a box and 17 body keypoints with
  confidences; we take the wrist keypoint of the largest person. It was
  trained on the COCO keypoint dataset; we use it as-is.
- **Tabular Q-learning (reinforcement learning), from the class's
  `qlearn.py`.** The opponent keeps a 3×3 table of expected reward for (your
  zone, aim zone), updates the chosen cell toward the reward it got with
  `Q ← Q + α·(r − Q)`, and picks the best cell except for an ε fraction of
  random tries.
- **AprilTag detection** (OpenCV ArUco) and the **FFT whistle detector** are
  classical computer vision and signal processing, not learned models.

## The usual questions

*(Draft — edit in your own words.)*

**What I'm proud of.** The whole chain works end to end: a neural net, an
IMU over Bluetooth, AprilTags, MQTT, a learning opponent and a physical LED
scoreboard, with the hub buzzing in your hand on every hit. And that the
opponent visibly figures out where you're weak within a dozen rallies.

**What was hard.** Deciding what "a hit" means with two noisy sensors —
the pose jitters and the gyro spikes — and tuning the window so a real swing
counts without letting a nervous wiggle count. Also keeping the 8 GB laptop
happy with a pose network, a camera and Bluetooth at once.

**What I'd do next.** Direction-aware swings (forehand vs backhand aim the
return), a two-player mode over MQTT where each player's hub is a paddle,
and running the pose model on the UNO Q itself.
