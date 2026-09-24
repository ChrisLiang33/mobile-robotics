"""
Day 4 bonus -- one robot, two whistlers, MQTT in between.

Each group member runs whistle_car.py on their OWN computer with their
OWN microphone, publishing decisions instead of (or as well as) driving:

    person A:  python whistle_car.py --no-robot --publish ME193/chris/cmd
    person B:  python whistle_car.py --no-robot --publish ME193/tasha/cmd

Then ONE computer near the robot runs this driver, which merges the two
command streams onto one car -- A's whistles own the throttle
(FASTER/STOP), B's whistles own the steering (LEFT/RIGHT):

    python mqtt_driver.py --throttle ME193/chris/cmd --steer ME193/tasha/cmd

Same dead-man behavior as whistle_car: the car only moves while the
throttle person is actually whistling, and stops when both topics go
quiet. Ctrl-C stops the motors.
"""

import argparse
import threading
import time

import whistle_car as wc
from mqttlib import MQTTClient


def main():
    ap = argparse.ArgumentParser(description="Drive one robot from two MQTT whistle streams")
    ap.add_argument("--throttle", required=True, metavar="TOPIC",
                    help="topic whose FASTER/STOP commands control speed")
    ap.add_argument("--steer", required=True, metavar="TOPIC",
                    help="topic whose LEFT/RIGHT commands control steering")
    ap.add_argument("--no-robot", action="store_true")
    args = ap.parse_args()

    car = wc.Car(enabled=not args.no_robot)
    car.connect()

    policy = wc.Policy()

    # Publishers only send decision CHANGES, but the policy's dead-man
    # throttle needs a continuous feed -- so remember the latest payload
    # per topic and re-apply it every control tick.
    latest = {"throttle": "NONE", "steer": "NONE"}

    def on_throttle(topic, payload):
        latest["throttle"] = payload

    def on_steer(topic, payload):
        latest["steer"] = payload

    mqtt = MQTTClient()
    mqtt.connect()
    mqtt.subscribe(args.throttle, on_throttle)
    mqtt.subscribe(args.steer, on_steer)
    print(f"throttle <- {args.throttle}    steering <- {args.steer}")

    running = [True]

    def control_loop():
        while running[0]:
            th, st = latest["throttle"], latest["steer"]
            if th in ("FASTER", "STOP", "REVERSE"):
                policy.update(th)
            if st in ("LEFT", "RIGHT"):
                policy.update(st)
            if th not in ("FASTER", "STOP", "REVERSE") and st not in ("LEFT", "RIGHT"):
                policy.update(None)        # both silent -> dead-man stop
            l, r, speed, turn, last = policy.wheels()
            car.drive(l, r)
            time.sleep(1.0 / wc.CMD_HZ)

    threading.Thread(target=control_loop, daemon=True).start()

    try:
        while True:
            l, r, speed, turn, last = policy.wheels()
            turn_s = {0: "straight", -1: "LEFT", 1: "RIGHT"}[turn]
            print(f"\rlast: {last:7s} speed {speed:3.0f}% {turn_s:9s} wheels L{l:+4.0f} R{r:+4.0f}",
                  end="", flush=True)
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nstopping.")
    finally:
        running[0] = False
        time.sleep(0.1)
        car.disconnect()
        mqtt.disconnect()


if __name__ == "__main__":
    main()
