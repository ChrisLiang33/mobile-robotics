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
