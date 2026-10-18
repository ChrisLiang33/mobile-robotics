# 📡 Sumo Ranger (UNO Q App Lab project)

The wrestler's eyes: an HC-SR04 ultrasonic sensor on the plow, read by the
sketch, published over MQTT on `ME193/ChrisLiang/sumo/range` as `{"cm": 37.2}`
ten times a second (`-1` = nothing within range). The LED matrix shows a
range bar, a full flash while the brain is charging, and a sad face when it
has lost.

Wiring: `VCC → 5V`, `GND → GND`, `TRIG → D7`, `ECHO → D8` **through a
1 kΩ / 2 kΩ divider** (the echo pin is 5 V, the UNO Q's pins are 3.3 V).

Deploy: `../unoq.sh deploy` over USB or `UNOQ_HOST=<ip> ../unoq.sh deploy`.
