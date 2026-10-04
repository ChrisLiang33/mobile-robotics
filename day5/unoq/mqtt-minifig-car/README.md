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

## Wiring: Seeed Studio / Cytron Maker Drive (MX1508)

The Maker Drive takes **two PWM inputs per motor** — exactly what the sketch
sends: `A: PWM, B: 0` = forward, `A: 0, B: PWM` = backward, `0/0` = brake.
Logic-high is anything ≥ 1.7 V, so the UNO Q's 3.3 V pins drive it directly.

**Signal header (6 pins on the Maker Drive) ↔ UNO Q:**

| Maker Drive pin | UNO Q pin | sketch constant |
|---|---|---|
| `M1A` | D5  | `M_LEFT_A`  |
| `M1B` | D6  | `M_LEFT_B`  |
| `M2A` | D9  | `M_RIGHT_A` |
| `M2B` | D10 | `M_RIGHT_B` |
| `GND` | GND | — (required: common ground) |
| `5VO` | **nothing** | 5 V @ 200 mA out — far too little for a UNO Q; leave it unconnected |

**Motor terminals:** left motor's two leads → `M1A`/`M1B` screw terminal,
right motor → `M2A`/`M2B`. Polarity only sets direction; fix a backwards
wheel with `LEFT_SIGN`/`RIGHT_SIGN` in the sketch, don't rewire.

**Power — two separate supplies:**
- **UNO Q**: the USB-C power bank, via its USB-C port. (The UNO Q runs
  Linux and needs a real 5 V supply; never try to feed it from the Maker
  Drive's `5VO`.)
- **Maker Drive `VB+`/`VB−` (green terminal)**: its own **2.5–9.5 V** battery
  — a 4×AA holder (6 V) is ideal; the board is reverse-polarity protected.
  Tie this battery's `−` to the same ground as the UNO Q (the `GND` header
  pin does that once it's wired).
- *No second battery?* Fallback: UNO Q `5V` header pin → `VB+`, UNO Q `GND`
  → `VB−`. Small gearmotors run fine on 5 V, but motor start-up spikes can
  brown out the Linux side and reboot the board mid-drive — if it resets
  when the wheels kick, you need the separate battery.

**UNO Q PWM gotchas (ArduinoCore-zephyr):**
- `pinMode()` before `analogWrite()` silently kills PWM on that pin — the
  sketch deliberately has no `pinMode()` calls for the motor pins.
- On cores before **0.55.2**, PWM on D3 disabled D6/D8 and PWM on D11
  disabled D5/D12/D13. The pins above avoid D3/D11; update the core in App
  Lab (Boards manager) if it's older.
- Default `analogWrite` frequency (~1 kHz) is within the driver's DC–20 kHz.

**Test it without any code:** power the Maker Drive and press its onboard
**M1A/M1B/M2A/M2B test buttons** — each spins the motor at full speed in
that direction and lights the status LED. That proves the motors and
battery before the UNO Q is involved.

**Then from the UNO Q:** run the app, open the MQTT debugger
(`Public stuff/Debugging/index.html`), and publish `{"speed": 50}` to
`ME193/minifig/test`. Both wheels should spin *forward* for 2 s, then stop.
`{"speed": -50}` = backward.
- One wheel backwards → flip that side's `LEFT_SIGN` / `RIGHT_SIGN`.
- Nothing moves but the test buttons work → check `GND` header pin and
  that the sketch has no `pinMode()` on the motor pins.
- Board reboots when motors kick → separate motor battery.
Then run the real thing: if the car drives *away* from the center, flip
`DIRECTION` in `main.py`.
