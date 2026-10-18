"""Print the Color Sensor's raw readings for a while -- hold it over floor and
tape and see what it actually returns.   python sensor_probe.py [seconds]"""
import sys, time
import legoeducation as le
from lelib import colorSensor
from sumo import COLOR_CARD_NAME, COLOR_CARD_SERIAL
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 30
cs = colorSensor()
print("connecting to the Color Sensor...")
cs.connect(card_serial=COLOR_CARD_SERIAL, card_color=getattr(le, f"LEGO_COLOR_{COLOR_CARD_NAME}"))
print("connected; move it between floor and tape")
t0 = time.time(); last = None; n = 0
while time.time() - t0 < secs:
    r = cs.raw_reading(); rgb = cs.raw_rgb(); name = cs.detect_color(); refl = cs.reflection()
    cur = (rgb, refl, name)
    n += 1
    if cur != last:
        print(f"{time.time()-t0:5.1f}s  rgb {rgb}  refl {refl:3d}  color {name:9s}  hue {r['hue']:.2f} sat {r['saturation']:.2f} val {r['value']:.2f}", flush=True)
        last = cur
    time.sleep(0.1)
print(f"done: {n} polls")
cs.disconnect()
