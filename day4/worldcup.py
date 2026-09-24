"""
Day 4 -- World Cup game mode, built on whistle_car.py.

    python worldcup.py --role ball            # you are the striker
    python worldcup.py --role goalie          # you are the keeper
    add --no-robot to rehearse without the car

Both roles: the car is frozen until the message "start" arrives over
MQTT on ME193/Rogers. After that, normal whistle driving (same policy
and live display as whistle_car.py, with a game-status line added).

BALL role:
  * The color/light sensor rides OPEN and FACING FORWARD on the car.
    If the goalie gets close enough to it (reflection over
    LIGHT_THRESH for ~0.3 s), you are tagged out: motors shut down,
    MSG_FAIL is published on GAME_TOPIC, and the death song plays.
  * If you make it into the goal, blow the SCORE whistle -- a HIGH
    whistle held for SCORE_HOLD_S seconds straight (long enough that
    normal speed-up whistles don't trigger it). That publishes
    MSG_SCORE and plays the victory song.

GOALIE role:
  * Subscribed to GAME_TOPIC: when MSG_FAIL arrives (you tagged the
    ball) your computer sings the victory song; when MSG_SCORE arrives
    (they scored) it plays the death song.

The message strings below are the "agreed messages" -- make sure your
opponent's code uses the same GAME_TOPIC / MSG_FAIL / MSG_SCORE.
"""

import argparse
import threading
import time

import whistle_car as wc
from mqttlib import MQTTClient
from songs import play_death, play_victory

START_TOPIC = "ME193/Rogers"          # Rogers publishes "start" here
GAME_TOPIC = "ME193/worldcup/chris"   # agree on this with your opponent
MSG_FAIL = "ball:failed"              # goalie tagged the ball's light sensor
MSG_SCORE = "ball:scored"             # ball made it into the goal

LIGHT_THRESH = 60      # reflection (0-100) that counts as "goalie is here"
LIGHT_HOLD_S = 0.3     # must stay above threshold this long (no false tags)
SCORE_HOLD_S = 2.0     # continuous HIGH whistle that declares a goal

LIGHT_CARD_SERIAL = wc.CARD_SERIAL    # light sensor pairs with the same card
LIGHT_CARD_COLOR_NAME = wc.CARD_COLOR_NAME


class Game:
    def __init__(self, role, car, policy, mqtt):
        self.role = role
        self.car = car
        self.policy = policy
        self.mqtt = mqtt
        self.lock = threading.Lock()
        self.started = False
        self.over = False
        self.result = ""
        self.high_since = None        # for the score whistle

    # ---- MQTT ----
    def on_start(self, topic, payload):
        if "start" in payload.lower() and not self.started:
            with self.lock:
                self.started = True
            print(f"START received on {topic}: game on!")

    def on_game_msg(self, topic, payload):
        if self.role != "goalie" or self.over:
            return
        if payload == MSG_FAIL:
            self.finish("YOU TAGGED THE BALL -- victory!", play_victory)
        elif payload == MSG_SCORE:
            self.finish("They scored on you...", play_death)

    # ---- ball-role events ----
    def tagged_out(self):
        self.mqtt.publish(GAME_TOPIC, MSG_FAIL)
        self.finish("TAGGED BY THE GOALIE -- shutting down", play_death)

    def scored(self):
        self.mqtt.publish(GAME_TOPIC, MSG_SCORE)
        self.finish("GOOOAL!", play_victory)

    def finish(self, message, song):
        with self.lock:
            if self.over:
                return
            self.over = True
            self.result = message
        self.policy.speed = 0.0
        self.policy.turn = 0
        self.car.stop()
        print(message)
        threading.Thread(target=song, daemon=True).start()

    # ---- hooks into whistle_car ----
    def gate_decision(self, decision):
        """Runs on every audio chunk: freezes the car outside the game
        window and watches for the long-HIGH score whistle."""
        with self.lock:
            active = self.started and not self.over
        if not active:
            self.policy.speed = 0.0
            self.policy.turn = 0
            return
        if self.role == "ball":
            now = time.monotonic()
            if decision == "FASTER":
                if self.high_since is None:
                    self.high_since = now
                elif now - self.high_since >= SCORE_HOLD_S:
                    self.scored()
            else:
                self.high_since = None

    def status(self):
        with self.lock:
            if self.over:
                return f"[{self.role.upper()}] GAME OVER: {self.result}"
            if not self.started:
                return f"[{self.role.upper()}] waiting for 'start' on {START_TOPIC}..."
            if self.role == "ball" and self.high_since is not None:
                held = time.monotonic() - self.high_since
                return f"[BALL] PLAYING -- score whistle {held:.1f}/{SCORE_HOLD_S:.0f}s"
            return f"[{self.role.upper()}] PLAYING -- whistle to drive"


def watch_light_sensor(game, enabled, running):
    """Ball role: poll the front light sensor; sustained high reflection
    means the goalie is on top of us."""
    if not enabled:
        print("--no-robot: light sensor disabled (press 't' logic not simulated).")
        return
    import legoeducation as le
    from lelib import colorSensor
    color = (getattr(le, f"LEGO_COLOR_{LIGHT_CARD_COLOR_NAME}")
             if LIGHT_CARD_COLOR_NAME else None)
    cs = colorSensor()
    print("Connecting to Color Sensor...")
    cs.connect(card_serial=LIGHT_CARD_SERIAL, card_color=color)
    print("Color Sensor connected.")
    above_since = None
    while running[0] and not game.over:
        r = cs.reflection()
        now = time.monotonic()
        if game.started and r >= LIGHT_THRESH:
            if above_since is None:
                above_since = now
            elif now - above_since >= LIGHT_HOLD_S:
                game.tagged_out()
                break
        else:
            above_since = None
        time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser(description="ME193 World Cup")
    ap.add_argument("--role", choices=["ball", "goalie"], required=True)
    ap.add_argument("--no-robot", action="store_true")
    args = ap.parse_args()

    car = wc.Car(enabled=not args.no_robot)
    car.connect()

    mqtt = MQTTClient()
    mqtt.connect()

    detector = wc.WhistleDetector()
    policy = wc.Policy()
    game = Game(args.role, car, policy, mqtt)

    mqtt.subscribe(START_TOPIC, game.on_start)
    if args.role == "goalie":
        mqtt.subscribe(GAME_TOPIC, game.on_game_msg)

    running = [True]
    if args.role == "ball":
        threading.Thread(target=watch_light_sensor,
                         args=(game, not args.no_robot, running),
                         daemon=True).start()

    import queue
    viz_q = queue.Queue(maxsize=8)
    pa, stream = wc.start_audio(detector, policy, viz_q,
                                on_decision=game.gate_decision)

    def control_loop():
        while running[0]:
            l, r, *_ = policy.wheels()
            car.drive(l, r)
            time.sleep(1.0 / wc.CMD_HZ)

    threading.Thread(target=control_loop, daemon=True).start()

    try:
        wc.run_ui(policy, viz_q, status_fn=game.status)
    finally:
        running[0] = False
        time.sleep(0.1)
        stream.stop_stream()
        stream.close()
        pa.terminate()
        car.disconnect()
        mqtt.disconnect()


if __name__ == "__main__":
    main()
