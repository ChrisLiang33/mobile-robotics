# Day 5 — YOLO Green-Minifig Tracker + Arduino UNO Q Car

Two parts, one pipeline:

1. **Find the green LEGO minifig** in the laptop camera with a YOLO object
   detector **we trained ourselves**, publish its position over MQTT, and show
   it as a **blue dot at the scaled location on the UNO Q's 8×13 LED matrix**.
2. **Center it**: the minifig rides on an Arduino UNO Q robot with DC motors.
   The UNO Q receives the position and drives forward/backward until the
   minifig sits in the **middle of the computer screen**, then stops — the
   AprilTag exercise from Day 3, now with a neural net and an Arduino.

## Files

| File | What it is |
|------|------------|
| `capture.py` | Step 1 — grab training frames from the webcam into `data/raw/` |
| `label.py` | Step 2 — semi-automatic labeler (green-blob proposal, human accepts/redraws/rejects every box) → `data/labels/` |
| `train.py` | Step 3 — split, fine-tune YOLOv8n with ultralytics, save `models/minifig.pt`, print precision/recall/mAP |
| `track_minifig.py` | Step 4 — live detection, blue-dot overlay, LED-matrix preview, MQTT publishing |
| `common.py`, `mqttlib.py` | shared paths/camera helper; course MQTT wrapper |
| `unoq/mqtt-minifig-car/` | the UNO Q App Lab project: `python/main.py` (display + drive policy) and `sketch/sketch.ino` (LED matrix + motor PWM) |
| `unoq/professor_MQTT_Minifig_Monitor.zip` | Prof. Rogers' original display-only project this is built on |
| `models/minifig.pt`, `models/metrics.txt` | our trained weights and their validation scores |

## Setup

```bash
pip install ultralytics opencv-python paho-mqtt numpy
pip install pyobjc-framework-AVFoundation   # macOS only, optional: see below
```
(ultralytics brings PyTorch; on an Apple-silicon Mac training uses the GPU via `mps`.)

**Which camera?** The scripts use the laptop's built-in camera. On a Mac, an
iPhone that is nearby can show up as a second camera (Continuity Camera) and
OpenCV may number it first; with the optional package above the scripts
identify cameras by name and skip the phone. `python common.py` lists them,
and `--camera N` on `capture.py` / `track_minifig.py` forces a specific one.

## The pipeline

**1. Capture** — `python capture.py`. SPACE saves a frame, `a` toggles
auto-save every 0.5 s while you walk the minifig around. Get 150–300 frames:
every part of the image, near and far, tilted, different backgrounds and
lighting, on the robot and off. Also capture frames with **no** minifig and
with **other-colored** minifigs — those become negatives.

**2. Label** — `python label.py`. For each image it proposes a box around
the biggest green blob using an HSV threshold (classical OpenCV, no cloud
tools). You press SPACE to accept, drag to redraw, or `x` to say "no green
minifig here" (writes an empty label → negative example). Nothing is saved
without a human keypress. Output is standard YOLO `0 cx cy w h` text files.

**3. Train** — `python train.py --epochs 60` (YOLOv8n, 640 px). Splits
train/val by runs of consecutive frames, fine-tunes from COCO-pretrained
weights, copies the best checkpoint to `models/minifig.pt`, then validates and
prints the metrics (also saved to `models/metrics.txt`; curves and confusion
matrix in `runs/minifig_val/`). It uses an NVIDIA GPU, an Apple GPU, or the
CPU, whichever it finds. Ours ran on a lab machine with an RTX 5090: 60
epochs in 23 seconds. On our 8 GB MacBook the same run pushed the machine deep
into swap (about 30 s per batch), so train somewhere with memory to spare and
copy `models/minifig.pt` back; running the model afterwards is light.

**4. Track** — `python track_minifig.py`. Runs the model on the webcam,
draws a green box and a **blue dot** on the centroid (same color as the UNO Q
LEDs), shows the normalized error and what the UNO Q should do, and a mini
13×8 **LED-matrix preview** in the corner so you can debug without the board.
Publishes to `ME193/minifig/chris` on `test.mosquitto.org` at up to 15 Hz. (Our
own topic rather than the shared class default `ME193/minifig`: with several
teams in one room, a shared topic lets everyone's detections drive everyone's
car. It is one constant, `MQTT_TOPIC`, in `common.py` and in the UNO Q's
`main.py` — they must match.)

```json
{"x": 731, "y": 402, "w": 1280, "h": 720, "bw": 58, "bh": 131, "conf": 0.93, "found": true}
{"found": false, "w": 1280, "h": 720}        ← when nothing is detected
```

Plumbing test before you have a trained model:
`python track_minifig.py --model yolov8n.pt --class-name person` tracks *you*
with the stock COCO model through the exact same code path.

**5. UNO Q** — run `./unoq/unoq.sh deploy` with the board plugged into the
laptop by USB-C, or `UNOQ_HOST=<board-ip> ./unoq/unoq.sh deploy` over WiFi
(copies `unoq/mqtt-minifig-car/` to the board and starts it;
`./unoq/unoq.sh logs` shows its output). The board must be on WiFi
to reach the broker. It subscribes to the feed, draws the marker, and drives.
Wiring, SSH and App Lab alternatives are in `unoq/mqtt-minifig-car/README.md`.

**Debugging**: open Prof. Rogers' `Public stuff/Debugging/index.html` MQTT
console in a browser, subscribe to `ME193/minifig/chris` to watch the laptop's
feed and `ME193/minifig/chris/drive` to watch the UNO Q's decisions (`{"err": 0.31,
"speed": 30}`), or publish a hand-written position to move the LED dot and the
car without the camera at all.

## Q1 — Describe the policy: how does it make decisions?

The decision is made **on the UNO Q** (`python/main.py`, `compute_speed()`),
20 times a second, from the latest position message:

```
err   = (x − w/2) / (w/2)        # −1 = left edge of the camera, 0 = center, +1 = right edge
speed = Kp · err                 # Kp = 70 → proportional control
```
- `|err| < 0.08` → **speed 0**: the minifig is in the middle band, the car parks.
- otherwise the command is **proportional to the distance from center** —
  fast when far away, slowing as it approaches — floored at `MIN_SPEED = 30`
  so static friction can't strand it just short of center, clamped at ±80,
  and multiplied by `DIRECTION` (±1, set once from a drive test).
- The signed speed goes over the Bridge to the sketch, which PWMs the two
  motors (one sign-flipped for mirrored mounting). Forward/backward along the
  robot's axis is left/right on screen because the minifig rides on its side.

It's a P-controller on pixel error, the same idea as the Day 3 AprilTag car,
but split across two computers: the laptop measures (YOLO), MQTT carries the
measurement, the Arduino decides and acts. The sketch also has a 1-second
watchdog: if the Linux side stops sending commands, the motors stop.

## Q2 — What does your code do if no minifigure is detected?

Three layers, so the car can never keep driving on stale information:

1. **The laptop says so explicitly.** With no detection above the confidence
   threshold, `track_minifig.py` publishes `{"found": false}` (instead of
   going quiet) and shows *NO MINIFIG DETECTED* on screen. The UNO Q marks the
   target lost and commands speed 0 on its next tick (≤ 50 ms).
2. **Silence also means stop.** If the laptop crashes or WiFi drops, the UNO Q
   stops when no position has arrived for 0.5 s (`DRIVE_STALE`), and the
   sketch's own watchdog stops the motors if the Python side dies.
3. **The display remembers.** The LED marker shrinks from a 3×3 block to a
   single pixel at the *last known* position (Prof. Rogers' fresh/stale
   design), so you can see where the minifig was last seen while the car
   waits. Driving resumes the instant a new detection arrives.

## Q3 — How good is your model? Can you confuse it?

**The data.** 195 photos from the laptop camera, 640×360, in two rooms: 150
with the minifig (on the car, near and far, tilted, partly out of frame) and
44 with none (the car alone with its green jumper wires, hands, empty scene).

**The test.** The photos were auto-captured half a second apart, so
neighbours are near-duplicates; a random train/test split would leak the test
photos into training. `train.py` instead holds out whole runs of 10
consecutive frames: 154 photos to train on, 40 the model never saw.

**The numbers** (YOLOv8n fine-tuned for 60 epochs, on those 40 unseen photos;
`models/metrics.txt`, plots in `models/plots/`):

| metric | value |
|---|---|
| precision | 1.000 |
| recall | 0.999 |
| mAP50 | 0.995 |
| mAP50-95 | 0.832 |

Run the way the live tracker uses it (confidence ≥ 0.5): it found the minifig
in **32 of 33** unseen photos that contain one, with **no false alarms** on
the 7 empty ones and no extra boxes. Confidence was 0.94 at the median and
0.62 at worst; boxes overlap the hand-checked labels by 94% (median IoU). On
the laptop it runs at about 50 frames/s on the Apple GPU and 24 on the CPU.

These scores are honest for *our setup* and optimistic for anything else:
the unseen photos still come from the same two rooms, the same car, and the
same afternoon.

**What confuses it (measured):**
- **A tilted minifig.** The one miss (photo 110) is the car held at an angle
  so the figure lies sideways. Almost every training photo has it upright.
- **A hand on the minifig.** In six photos fingers cover part of the figure
  and we labeled those "no minifig". The model learned exactly that: it
  reports nothing on them even at a confidence threshold of 0.10, including
  one where the figure is almost fully visible. A labeling choice became a
  model behavior.
- **Green jumper wires — the confusion we designed out.** Our first labeling
  tool boxed the car's green wires as if they were the minifig. We fixed the
  labels and kept 44 "nothing here" photos of the car without the figure;
  the trained model raises no false alarm on any of them.

**Still to try live** (run `track_minifig.py` and watch the confidence):
- **A different-colored minifig** (red, yellow, blue): a model trained only
  on green positives with other colors as *negatives* should ignore them.
  If you skipped negatives, it will often fire on *any* minifig-shaped thing
  at lower confidence — color is a weaker cue than shape for a convnet.
- **Green things that aren't minifigs** (a green marker, a plant, a green
  shirt): the dataset-limited risk in the other direction.
- **Distance / angle / lighting** outside what was captured: confidence drops
  first, then detections drop out — the model only knows what it has seen.
- **Partial occlusion** (hand over the head): usually survives if the dataset
  had some.

Raising `--conf` trades missed detections for fewer false alarms; the
negatives in the dataset are what move both curves at once. Record what
actually confused it and at what confidence — that's the answer.
