# Day 4 — Whistle-Controlled Car + MQTT World Cup

Whistle at your laptop and the LEGO car obeys: a **high** whistle speeds it up,
a **low** whistle stops it, and the two **middle** bands turn it left or right.
A live window shows the microphone waveform, the FFT spectrum with the command
bands shaded, and the decision being acted on. On top of that sits the World
Cup game: MQTT start signal, light-sensor tagging, score whistle, and
victory/death songs.

## Files

| File | What it is |
|------|------------|
| `whistle_car.py` | Main assignment: PyAudio stream → FFT pitch policy → motors, with live signal/decision display |
| `worldcup.py` | Game mode (ball or goalie): MQTT start, light-sensor tag-out, score whistle, songs |
| `mqtt_driver.py` | Bonus: two people whistle at their own laptops, MQTT merges both streams onto one robot |
| `songs.py` | Victory / death tone sequences played through PyAudio output |
| `lelib.py`, `mqttlib.py` | Course libraries (LEGO wrappers, MQTT wrapper) |

## Setup

```bash
brew install portaudio
pip install pyaudio numpy matplotlib paho-mqtt legoeducation
```

(or use the class venv, which already has everything:
`source ~/Desktop/ME193-Robotics/my_env/bin/activate`)

macOS will ask for microphone permission the first time — if the stream is
silent, check System Settings → Privacy & Security → Microphone for your
terminal app, then restart the terminal.

## Running

```bash
python whistle_car.py               # drive the car by whistling
python whistle_car.py --no-robot    # audio + policy only, no robot needed
python worldcup.py --role ball      # game day: you are the striker
python worldcup.py --role goalie    # game day: you are the keeper
```

Whistle commands (hold the whistle until the car reacts, ~0.15 s):

| Whistle pitch | Band | Command |
|---|---|---|
| Low (500–900 Hz) | red | **STOP**; whistle low *again* while stopped → **REVERSE** (backs up while held) |
| Mid-low (900–1400 Hz) | blue | **turn LEFT** (while held) |
| Mid-high (1400–2000 Hz) | green | **turn RIGHT** (while held) |
| High (2000–3500 Hz) | orange | **SPEED UP** (accelerates while held) |

The spectrum plot shows exactly which band your whistle lands in — if your
"low" whistle reads as mid, just adjust the `BANDS` table to your own
whistling range.

## Q1 — Describe the policy: how does it make decisions?

Every ~46 ms audio chunk (2048 samples at 44.1 kHz) goes through the same
pipeline:

1. **FFT** the Hann-windowed chunk and find the strongest frequency peak
   inside the whistle band (500–3500 Hz).
2. **Gate it** (see Q3): the peak must be loud enough, pure enough, and
   persist across 3 consecutive chunks to count as a whistle.
3. **Classify** the surviving pitch into one of four bands → STOP / LEFT /
   RIGHT / FASTER.
4. **Integrate into car state.** The policy keeps two state variables:
   `speed` and `turn`. FASTER adds 4% per chunk while held (so a long high
   whistle accelerates smoothly toward 80%); STOP zeroes the speed — and a
   *second* low whistle starting while the car is already stopped drives it in
   reverse (down to −40%) for as long as it's held, so low-low is the back-up
   sequence; LEFT/RIGHT
   set a steering offset that lasts only while the whistle is held. Wheel
   commands are `left = speed + turn·25`, `right = speed − turn·25` — a moving
   car arcs, a stopped car spins in place.
5. A control thread sends the wheel speeds to the two motors over BLE at
   20 Hz (skipping sends when nothing changed).

So unlike Day 3's continuous PD controller, this is a **discrete
event policy with integrated state**: whistles are commands that edit the
car's state, not a signal the car proportionally tracks.

## Q2 — What does your code do if no whistle is detected?

Silence is *not* a command — the car keeps doing whatever it was last told,
which is what makes "whistle high for a bit, then let it cruise" work. Three
exceptions keep that safe:

1. **Turns end immediately**: steering is only applied while a mid whistle is
   actually sounding, so letting go straightens the car out.
2. **Fail-safe timeout**: after 10 s with no valid whistle at all, a moving
   car stops on its own.
3. In the World Cup, the car is also forced to zero before "start" arrives
   and after the game ends.

## Q3 — How did you try to mask out unwanted noise?

Three layers, each catching a different kind of noise:

1. **Band-limiting.** Only 500–3500 Hz can ever trigger a command. That
   discards low-frequency room rumble, fans, motor noise, and the fundamental
   of almost all speech (~85–255 Hz) before classification even starts.
2. **Loudness + purity gates.** The peak must clear an absolute magnitude
   threshold (quiet background tones can't trigger) *and* stand at least 8×
   above the average in-band level. A whistle is a near-pure sine — one tall
   spike on a flat spectrum — while claps, speech, and music spread energy
   across many bins and fail the purity test even when loud.
3. **Persistence.** The same band must win 3 chunks in a row (~140 ms) before
   the command fires, so momentary chirps, squeaky chairs, and FFT flicker
   between adjacent bins get ignored.

The dashed threshold line and the live spectrum in the display make all three
visible — you can watch a clap fail the purity test and a whistle pass it.

## World Cup game (`worldcup.py`)

- Both roles idle (motors frozen) until the message **"start"** arrives on the
  MQTT topic `ME193/Rogers`, then whistle-driving goes live.
- **Ball**: the color/light sensor rides open and facing forward. If the
  goalie holds it over `LIGHT_THRESH` reflection for 0.3 s, the car shuts
  down, publishes `ball:failed` on `ME193/worldcup/chris`, and plays the
  death song. Scoring: hold a **high whistle for 2 s straight** (long enough
  that ordinary speed-up whistles can't fake it) → publishes `ball:scored`
  and plays the victory song.
- **Goalie**: subscribed to the same topic — `ball:failed` triggers the
  victory song on the goalie's laptop, `ball:scored` the death song.
- The topic and the two message strings at the top of `worldcup.py` are the
  "agreed messages" — the opponent's code must use the same three constants.

## Bonus — two mics, one robot (`mqtt_driver.py`)

Both group members stream audio on their **own** computers and control one
robot over MQTT, split by function:

```bash
# person A's laptop (throttle):   python whistle_car.py --no-robot --publish ME193/chris/cmd
# person B's laptop (steering):   python whistle_car.py --no-robot --publish ME193/tasha/cmd
# the laptop near the robot:      python mqtt_driver.py --throttle ME193/chris/cmd --steer ME193/tasha/cmd
```

`whistle_car.py --publish` sends every decision change over MQTT;
`mqtt_driver.py` merges the two streams — A's whistles own speed, B's own
steering — into the same policy/wheel math the single-player car uses.
