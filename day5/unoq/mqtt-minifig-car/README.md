# 🚗 MQTT Minifig Car (UNO Q App Lab project)

Load this folder as an app in **Arduino App Lab** on the UNO Q (or import it
as a zip). Based on Prof. Rogers' *MQTT Minifig Monitor*; adds DC-motor drive.

- `python/main.py` — subscribes to `ME193/minifig`, draws the marker on the
  LED matrix, runs the centering policy, calls `drive(speed)` on the sketch,
  echoes decisions on `ME193/minifig/drive`.
- `sketch/sketch.ino` — LED matrix + H-bridge motor PWM + 1 s watchdog.
  **Set the motor pins / signs at the top to match the robot.**

Before running: set `DEVICE_ID` in `main.py` to this board's App Lab name,
and make sure the board is on WiFi (it must reach `test.mosquitto.org`).

## Wiring the motors

The sketch drives each motor with **two PWM pins** (A/B): `A=PWM, B=0` is
forward, `A=0, B=PWM` is backward, `0/0` is stop. Defaults:

| sketch constant | UNO Q pin | goes to |
|---|---|---|
| `M_LEFT_A`  | D5  | left motor input 1 |
| `M_LEFT_B`  | D6  | left motor input 2 |
| `M_RIGHT_A` | D9  | right motor input 1 |
| `M_RIGHT_B` | D10 | right motor input 2 |

All four must be **PWM-capable pins** (the ones marked `~` on the header;
D3/D5/D6/D9/D10/D11 on the UNO layout). Change the constants if you move them.

**Rules that apply to every driver board:**
- **Motor power is separate.** Battery pack (4–6× AA, or a 2S LiPo through
  the driver's rating) → driver `VM` / `+12V` / `VIN`. Never run motors off
  the UNO Q's 5 V or 3.3 V pins — they can't source the current and the
  brown-outs reset the board.
- **Common ground.** Driver `GND` ↔ UNO Q `GND` ↔ battery `−`. Without this
  the PWM signals have no reference and nothing works.
- The UNO Q's GPIO is **3.3 V logic**. All the boards below accept that as
  "high". Don't feed 5 V into a UNO Q pin.
- Each motor: its two leads go to one driver output pair (`OUT1/OUT2` or
  `AO1/AO2`). Polarity only sets direction — fix a backwards wheel in
  software with `LEFT_SIGN`/`RIGHT_SIGN`, no need to rewire.

**Per driver:**

- **L298N (red module)** — `IN1,IN2` ← D5,D6; `IN3,IN4` ← D9,D10. Leave the
  `ENA`/`ENB` **jumpers on** (enable tied high) so the IN pins' PWM controls
  speed. Keep the `5V-EN` jumper on; do *not* connect its 5 V output to the
  UNO Q. Works down to ~6 V motor supply; drops ~2 V across itself, so small
  motors feel weak — fine for this.
- **TB6612FNG** — `AIN1,AIN2` ← D5,D6; `BIN1,BIN2` ← D9,D10; `PWMA`,`PWMB`
  tied to 3.3 V (or wire them to D3/D11 and set them HIGH in `setup()`);
  `STBY` → 3.3 V; `VCC` → 3.3 V (logic); `VM` → battery.
- **DRV8833** — `AIN1,AIN2` ← D5,D6; `BIN1,BIN2` ← D9,D10; `VCC`/`VM` →
  battery (3–10 V); `nSLEEP` high if the board exposes it. Simplest of all.
- **Arduino Motor Shield Rev3** — different scheme (DIR + PWM + BRAKE per
  channel on fixed pins 12/3/9 and 13/11/8). Say so and the sketch's
  `setMotor()` gets a 5-line rewrite.

**First power-up test (no camera, no model):** run the app, open the MQTT
debugger (`Public stuff/Debugging/index.html`), and publish to
`ME193/minifig/test` the payload `{"speed": 50}`. Both wheels should spin
*forward* for 2 s, then stop. `{"speed": -50}` = backward.
- One wheel backwards → flip that side's `LEFT_SIGN` / `RIGHT_SIGN`.
- Nothing moves → check common ground and that `ENA/ENB`/`STBY` are high.
- Board resets when motors kick → motors are drawing from the UNO Q; use a
  separate battery.
Then run the real thing: if the car drives *away* from the center, flip
`DIRECTION` in `main.py`.
