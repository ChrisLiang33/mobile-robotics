# Day 7 — Sumo Wrestler (behavior tree + KNN, shared control)

*ME193 — Chris Liang*

A LEGO Double Motor base with a plow, a Color Sensor looking at the floor,
and the hub's own IMU, driven by a **behavior tree** that runs on the **UNO Q
riding on the car**. The UNO Q talks Bluetooth to the LEGO hub and sensor,
publishes the robot's state over MQTT, and announces over MQTT when it has
lost. The human steers from the laptop with the keyboard — the keys travel
over MQTT — but the tree has the final say: it will not let you drive onto
the tape, and it shoves at full power the moment it feels contact. First one
pushed out of the blue-tape ring loses, and the laptop plays the sad trombone
when the "lost" message arrives.

```
 laptop: sumo_remote.py  --keys-->  MQTT  --keys-->  UNO Q: sumo.py --headless  --BLE-->  Double Motor + Color Sensor
         (HUD, sad song) <--state--       <--state, "ChrisLiang lost"--  (behavior tree, KNN)
```
(`sumo.py` without `--headless` also runs the whole thing on the laptop, as a fallback.)

## Files

| File | What it is |
|------|------------|
| `sumo.py` | the brain: sensors → blackboard → behavior tree → motors; `--headless` on the UNO Q (teleop over MQTT), or with a window on the laptop |
| `sumo_remote.py` | laptop side: sends key presses to the robot over MQTT, shows its state, plays the sad song on "lost" |
| `unoq_setup.sh` | one-time setup on the UNO Q: Python environment with the LEGO Bluetooth library, Bluetooth check |
| `bt.py` | the behavior-tree library (Selector, Sequence, Condition, Action, RUNNING) |
| `knn.py` | K-nearest-neighbors classifier: Color Sensor reading → "floor" or "tape" |
| `knn_samples.json` | the readings you taught it (created by pressing `f`/`t` in the HUD) |
| `sounds.py` | the sad song (PyAudio) |
| `lelib.py`, `mqttlib.py` | course libraries |

## Build and setup

- Double Motor base, plow on the front, **Color Sensor on the front with its
  window facing straight down**, about 1 cm off the floor, ahead of the wheels.
- Under 1 kg and inside a 30 cm cube — the hub, sensor and a LEGO plow are
  well within that.

```bash
# on the laptop
pip install legoeducation opencv-python numpy pyaudio paho-mqtt
python sumo.py --sim          # no hardware: watch the tree react to fake sensors (e = edge, c = contact)

# on the UNO Q (once): copy the folder up and run the setup script
scp -r day7 arduino@<board-ip>:~/sumo && ssh arduino@<board-ip> 'bash ~/sumo/unoq_setup.sh'
# game day: brain on the board, remote on the laptop
ssh arduino@<board-ip> '~/sumo/.venv/bin/python ~/sumo/sumo.py --headless'     # (--any-motor to skip my card)
python sumo_remote.py
```

MQTT topics: `ME193/ChrisLiang/sumo/teleop` (laptop → robot, key presses),
`ME193/ChrisLiang/sumo/state` (robot → laptop, 5×/s), `ME193/ChrisLiang/sumo/result`
(robot → everyone: `ChrisLiang lost: <reason>`).

**Teach it your ring first:** with the robot on plain floor press `f` a few
times, on the tape press `t` a few times. The HUD shows the KNN's verdict
and confidence live. Until it's taught, it falls back to the sensor's own
"Blue" color name.

Keys: `W/S` forward/back, `A/D` turn, `SPACE` stop, `x` referee declares us
out, `r` reset, `c` fake a contact, `q` quit.

## Q1 — Describe the policy. Which automated decision-making algorithm?

**A behavior tree**, ticked 20 times a second. Every tick walks the tree from
the top; the first branch whose condition is true acts, so the order *is* the
priority:

```
Selector: sumo
  Sequence: out      pushed out?          -> stop, play the sad song
  Sequence: edge     tape under sensor?   -> retreat: back 0.55 s, then turn 0.65 s (overrides the human)
  Sequence: attack   contact?             -> shove at 100%, human may bias the angle ±25
  Sequence: teleop   fresh human command? -> obey it, capped at 80%, EXCEPT forward within 0.4 s of tape
  Action:   search                        -> slow spin 0.9 s / creep forward 0.7 s, repeat
```

Two details make it a real tree rather than a list of ifs: the **retreat is a
RUNNING action** — once started it finishes its back-and-turn even if the tape
is already behind it, then steps aside so lower branches can act in the same
tick; and the **"out" rule is derived from sensor history**: on the tape for
1.5 s straight, or three edge retreats inside 6 s, means we're being pushed
out and the tree stops the motors itself (the referee key `x` does the same).

The second algorithm from the week's list, **K-nearest-neighbors**, decides
the "tape under sensor?" condition: each Color Sensor reading becomes a
4-number feature vector (normalized R, G, B and reflection), the 5 stored
samples nearest to it vote, and the majority label wins. We teach it with a
handful of readings of the actual floor and tape on game day, so lighting and
floor color don't matter.

## Q2 — Which sensors does the build use autonomously?

1. **The Color Sensor**, pointed at the floor, read ~20×/s. It feeds the KNN
   and triggers the retreat and the "out" detection. It acts with no human
   input and overrides the human.
2. **The hub's IMU** (accelerometer + its built-in gesture detector). A jolt
   of ≥ 350 milli-g away from gravity, or the hub's own COLLISION gesture,
   means we just hit or got hit by the opponent; "contact" stays true for
   0.6 s and triggers the full-power shove.

The UNO Q is the computer that reads both of these over Bluetooth, runs the
tree, and reports over MQTT. (A bump switch wired to the UNO Q and published
on `ME193/ChrisLiang/sumo/bump` would count as contact too; the code accepts
it, but this build doesn't carry one.)

## Q3 — What information can the human give via teleoperation / shared control?

The keyboard gives the wrestler **intent, not control**:

- `W/S/A/D` on the laptop set a wanted speed and turn; each press travels to
  the robot over MQTT and counts for 0.45 s, so holding a key keeps it alive
  and letting go stops. This is the lowest priority branch: the robot only
  follows it when nothing more urgent is true.
- **Vetoed near the edge**: if the human asks for forward within 0.4 s of the
  sensor seeing tape, the speed is zeroed and the HUD shows "VETOED: tape
  ahead". Turning is still allowed.
- **Biasing the shove**: during contact the robot goes to 100% on its own,
  but a held `A` or `D` tilts the push by 25% on one wheel — the human picks
  the angle, the robot picks the power.
- **Steering the retreat**: the direction the robot turns after backing off
  the tape is whichever way the human was last steering; with no input it
  picks at random.
- `x` is the referee: declare us out, stop, sad song.

That's the sensor fusion the rubric asks for: the same tick combines the
floor sensor, the IMU and the human's keys, and the tree decides whose
opinion wins.

## Q4 — Strengths and weaknesses

**Logic — strengths**
- Safety first: the edge rule outranks everything, including the driver, and
  the retreat always completes.
- Reacts in 50 ms to the floor sensor and ~20 ms to an IMU hit; the human
  can't be that fast.
- The KNN is taught on the actual floor and tape in the room, so it isn't
  fooled by a different shade of blue or a glossy floor.

**Logic — weaknesses**
- It only knows about the opponent when touching it. There's no distance
  sensor, so "search" is a blind spin-and-creep; a fast opponent can hit us
  from the side while we're searching.
- One floor sensor means it knows "tape" but not which side the tape is on;
  the retreat turns a random (or human-chosen) way.
- "Out" is inferred, not measured: being pushed out slowly across the tape
  looks like a long retreat sequence. Three retreats in six seconds or 1.5 s
  on the tape is a heuristic and can fire early in a scrappy match.
- The IMU can't tell our own hard braking from a hit, so a violent retreat
  can register as contact (contact doesn't outrank the retreat, so this only
  costs a pointless shove afterwards).

**Physical — strengths**
- Low and compact, with the hub's mass over the drive wheels; the LEGO
  motors have plenty of torque for pushing at 100%.

**Physical — weaknesses**
- Plastic wheels on a smooth floor: traction, not power, limits the push.
- The Color Sensor needs to ride at a fixed height; a plow that lifts the
  front wheels also lifts the sensor and blinds it.
- Two radio links in the loop: WiFi/MQTT for the human's keys (a broker
  round trip is ~100 ms, so teleop feels laggy) and Bluetooth for the motors.
  If Bluetooth drops, the hub keeps its last command — the retreat timers are
  short on purpose.
