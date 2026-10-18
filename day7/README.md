# Day 7 — Sumo Wrestler (behavior tree + KNN, shared control)

*ME193 — Chris Liang*

A LEGO Double Motor base with a plow. On the plow, an **HC-SR04 ultrasonic
sensor read by a UNO Q**, which publishes the distance to whatever is ahead
over MQTT. At the back, a **Color Sensor looking at the floor** for the blue
tape. The laptop runs a **behavior tree**: it holds the Bluetooth links to the
LEGO hub and Color Sensor, listens to the UNO Q over MQTT, and takes the
human's keyboard input — then decides. The tree has the final say: it charges
at full power when the ultrasonic sees the opponent in range, and the moment
the rear sensor finds tape under the car it declares the match lost: stops,
plays the sad trombone and announces it over MQTT.

```
 UNO Q + HC-SR04 on the plow  --MQTT "cm"-->  laptop: sumo.py (behavior tree, KNN, keyboard)  --BLE-->  Double Motor
                              <--MQTT state--                                                 --BLE-->  Color Sensor (floor)
                                                 --MQTT-->  ME193/ChrisLiang/sumo/result: "ChrisLiang lost: ..."
```

## Files

| File | What it is |
|------|------------|
| `sumo.py` | the brain on the laptop: sensors → blackboard → behavior tree → motors, keyboard teleop, HUD, MQTT |
| `unoq/sumo-ranger/` | UNO Q App Lab project: reads the HC-SR04, publishes range over MQTT, LED matrix range bar / sad face |
| `sumo_remote.py`, `unoq_setup.sh` | alternative layout (brain on the UNO Q over Bluetooth, laptop as remote) — not used on game day |
| `bt.py` | the behavior-tree library (Selector, Sequence, Condition, Action, RUNNING) |
| `knn.py` | K-nearest-neighbors classifier: Color Sensor reading → "floor" or "tape" |
| `knn_samples.json` | the readings you taught it (created by pressing `f`/`t` in the HUD) |
| `sounds.py` | the sad song (PyAudio) |
| `lelib.py`, `mqttlib.py` | course libraries |

## Build and setup

- Double Motor base, plow on the front, **Color Sensor at the back with its
  window facing straight down**, about 1 cm off the floor. (`SENSOR_AT =
  "back"` in `sumo.py`: when the tape appears under it the escape is to drive
  forward. Mount it at the front instead and set `"front"`: then it backs up
  and turns.)
- **UNO Q on the car** with the HC-SR04 on the plow facing forward: `VCC →
  5V`, `GND → GND`, `TRIG → D7`, `ECHO → D8` through a 1 kΩ / 2 kΩ divider
  (echo is 5 V, the UNO Q's pins are 3.3 V). Power from a USB power bank.
- Under 1 kg and inside a 30 cm cube — the hub, sensor, UNO Q and a LEGO plow
  are well within that.

```bash
pip install legoeducation opencv-python numpy pyaudio paho-mqtt
python sumo.py --sim          # no hardware: fake the sensors (e = edge, o = opponent in range, c = contact)

cd unoq && UNOQ_HOST=<board-ip> ./unoq.sh deploy     # put the ranger app on the UNO Q (once)
python sumo.py                # game day: hub + Color Sensor over Bluetooth, UNO Q over MQTT
```

MQTT topics: `ME193/ChrisLiang/sumo/range` (UNO Q → laptop, `{"cm": 37.2}`,
10×/s), `ME193/ChrisLiang/sumo/state` (laptop → anyone, 5×/s), and
`ME193/ChrisLiang/sumo/result` (`ChrisLiang lost: <reason>` when it happens).

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
  Sequence: out      tape under the rear sensor for 0.3 s? -> stop, play the sad song, publish "lost"
  Sequence: charge   opponent < 20 cm ahead (ultrasonic)?  -> charge at 100% (the "auto power-up"); A/D angle it ±20
  Sequence: teleop   fresh human command?                  -> drive it, capped at 80%
  Action:   idle                                           -> sit still (SEARCH_WHEN_IDLE=True: spin and creep instead)
```

Why this order: the Color Sensor sits at the **back** of the car, so tape
under it means we've already been shoved to the edge — there's no recovering
from that in sumo, so it's the loss condition itself, not something to
escape from (the referee key `x` declares the same thing). It's debounced
twice: three consecutive sensor readings must agree before the edge flag
flips, and the flag must hold for 0.3 s before the tree calls it; a single
bounce of the car over the tape doesn't end the match. The ultrasonic branch
is the "auto power-up": the human drives at up to 80%, but the moment the
UNO Q reports the opponent inside 45 cm, the tree takes over and goes to
100%; a stale reading (no message for 0.5 s) counts as "don't know", never as
"opponent". The tree only ever moves the car on its own because of the
ultrasonic; with nothing in range and no key held, it sits still.

The second algorithm from the week's list, **K-nearest-neighbors**, decides
the "tape under sensor?" condition: each Color Sensor reading becomes a
4-number feature vector (normalized R, G, B and reflection), the 5 stored
samples nearest to it vote, and the majority label wins. We teach it with a
handful of readings of the actual floor and tape on game day, so lighting and
floor color don't matter.

## Q2 — Which sensors does the build use autonomously?

1. **The HC-SR04 ultrasonic on the UNO Q**, facing forward, measured ~16×/s
   on the microcontroller and published over MQTT 10×/s. Under 45 cm means
   the opponent is in front of the plow: the tree charges at full power on
   its own. The UNO Q is the sensor the assignment asks for, talking over MQTT.
2. **The Color Sensor**, pointed at the floor at the back, read ~20×/s over
   Bluetooth. It feeds the KNN and is the "out" detector. It acts with no
   human input and overrides the human. The sensor never labels the tape
   "Blue" itself (it's too dim a target for its color classifier), but at a
   fixed height the floor reflects bright and the tape dark, and the KNN
   learns that from the readings we teach it.

(The hub's IMU is read too — a jolt means contact — and a full-power "shove"
branch can be switched on with `USE_IMU_CONTACT`; it's off for the match
because the car's own bumps triggered it.)

## Q3 — What information can the human give via teleoperation / shared control?

The keyboard gives the wrestler **intent, not control**:

- `W/S/A/D` set a wanted speed and turn. A key press counts for 0.45 s, so
  holding a key keeps it alive and letting go stops. This is the lowest
  priority branch: the robot only follows it when nothing more urgent is true.
- **Angling the charge**: when the ultrasonic puts the robot at 100% on its
  own, a held `A` or `D` tilts the push by 20% on one wheel — the human picks
  the angle, the robot picks the power. The human cannot cancel a charge.
- **Manual mode** (`m`): before the match the tree only *shows* what it
  would do while the keys drive, so each sensor can be checked on its own.
- `x` is the referee: declare us out, stop, sad song.

That's the sensor fusion the rubric asks for: the same tick combines the
ultrasonic range, the floor sensor, the IMU and the human's keys, and the
tree decides whose opinion wins.

## Q4 — Strengths and weaknesses

**Logic — strengths**
- Safety first: the edge rule outranks everything, including the driver, and
  the retreat always completes.
- Reacts in 50 ms to the floor sensor and ~20 ms to an IMU hit; the human
  can't be that fast.
- The KNN is taught on the actual floor and tape in the room, so it isn't
  fooled by a different shade of blue or a glossy floor.

**Logic — weaknesses**
- The ultrasonic only looks straight ahead in a ~15° cone, so an opponent
  beside us is invisible until we're hit (then the IMU takes over). "Search"
  is a blind spin-and-creep.
- Ultrasonic readings off a sloped or narrow LEGO opponent can be noisy; one
  reading under 45 cm is enough to charge, so a false echo wastes a charge.
- One floor sensor at the back means it detects being pushed out backwards
  well, but if the human drives forward over the edge the front of the car
  is across the line before the sensor sees tape — and then it's "out" by
  its own rule. Driving is the human's responsibility near the front edge.
- "Out" is a judgment from one sensor: 0.3 s of tape under the back. A dark
  patch on the floor, or the sensor lifting off the floor, looks the same.
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
- The Color Sensor needs to ride at a fixed height; a tilted car blinds it.
- Two radio links in the loop: Bluetooth for the motors and sensor, and
  WiFi/MQTT through a public broker for the ultrasonic (~100–200 ms). The
  charge decision therefore lags the real distance; the UNO Q, its power
  bank and the HC-SR04 also add weight on the front.
