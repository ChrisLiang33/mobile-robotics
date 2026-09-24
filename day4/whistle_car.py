"""
Day 4 -- Whistle-controlled car (PyAudio).

The microphone stream is chopped into ~46 ms chunks; each chunk is FFT'd
and the dominant pitch inside the whistle band (500-3500 Hz) is
classified into one of four commands:

    LOW      500- 900 Hz   STOP (instant)
    MID-LOW  900-1400 Hz   TURN LEFT  (while the whistle is held)
    MID-HIGH 1400-2000 Hz  TURN RIGHT (while the whistle is held)
    HIGH    2000-2700 Hz   FORWARD    (dead-man: only while whistling,
                           speed ramps up the longer it's held)
    TOP     2700-3500 Hz   REVERSE    (dead-man, same idea backwards --
                           parked at the top of the spectrum, farthest
                           from background noise and other whistlers)

A live matplotlib window shows the raw waveform, the spectrum with the
command bands shaded, the detection threshold, the detected peak, and
the decision being acted on -- the "show the signal and the resulting
decision" requirement.

Noise masking (three layers, see README):
  1. band-limit: only 500-3500 Hz can ever trigger (kills rumble, fans,
     most speech energy)
  2. loudness + purity gates: the peak must beat an absolute threshold
     AND stand PURITY_MIN times above the average in-band level --
     whistles are near-pure tones, claps/speech are broadband and fail
  3. persistence: the same band must win HOLD_FRAMES chunks in a row
     (~140 ms) before the command fires

The throttle is a DEAD-MAN switch: the car only moves while a whistle
is actually sounding. Stop whistling and it stops within RELEASE_GRACE_S
(a short grace so one blurry FFT chunk mid-whistle doesn't stutter it).
A low whistle is still the instant stop.

Run:
    python whistle_car.py               # drive the car
    python whistle_car.py --no-robot    # audio + policy only, no BLE
    python whistle_car.py --publish ME193/chris/cmd
                                        # ALSO publish each decision over
                                        # MQTT (see mqtt_driver.py: two
                                        # people, two mics, one robot)
"""

import argparse
import queue
import threading
import time

import numpy as np
import pyaudio

# --- Audio -----------------------------------------------------------------
RATE = 44100
CHUNK = 2048                  # ~46 ms per chunk, ~21.5 Hz FFT resolution

# --- Whistle bands (Hz) -> commands ---------------------------------------
BANDS = [
    ("STOP",  500,  900),
    ("LEFT",  900, 1400),
    ("RIGHT", 1400, 2000),
    ("FASTER", 2000, 2700),
    ("REVERSE", 2700, 3500),
]
BAND_LO, BAND_HI = 500.0, 3500.0

# --- Noise masking ---------------------------------------------------------
MAG_THRESH = 4.0              # absolute FFT peak magnitude gate
PURITY_MIN = 8.0              # peak must be this many x the in-band average
HOLD_FRAMES = 3               # consecutive agreeing chunks before acting

# --- Driving ---------------------------------------------------------------
MAX_SPEED = 80                # percent
MAX_REVERSE = 40              # reverse speed cap (percent)
ACCEL_STEP = 4                # speed added per FORWARD-whistle chunk (~21/s)
REVERSE_STEP = 3              # reverse speed added per REVERSE-whistle chunk
FORWARD_KICK = 15             # instant speed on whistle start (beats friction)
REVERSE_KICK = 18
TURN_SPEED = 25               # wheel differential while turning
RELEASE_GRACE_S = 0.3         # whistle gone this long -> dead-man stop
CMD_HZ = 20                   # motor command rate

# --- Bluetooth card (same pattern as Day 3) --------------------------------
CARD_COLOR_NAME = "ORANGE"    # None = first Double Motor found
CARD_SERIAL = 7572
LEFT_SIGN, RIGHT_SIGN = +1, -1  # motors are mounted mirror-image
TURN_SIGN = -1                  # -1: chassis turned opposite the labels in testing


class WhistleDetector:
    """FFT pitch detector with band/loudness/purity/persistence masking."""

    def __init__(self):
        self.window = np.hanning(CHUNK)
        self.freqs = np.fft.rfftfreq(CHUNK, 1.0 / RATE)
        self.in_band = (self.freqs >= BAND_LO) & (self.freqs <= BAND_HI)
        self._last_band = None
        self._streak = 0

    def process(self, samples):
        """Return (decision, peak_freq, peak_mag, spectrum).
        decision is a band name once masking is satisfied, else None."""
        mag = np.abs(np.fft.rfft(samples * self.window))
        band_mag = np.where(self.in_band, mag, 0.0)
        i = int(np.argmax(band_mag))
        peak_f, peak_m = float(self.freqs[i]), float(mag[i])
        purity = peak_m / (float(mag[self.in_band].mean()) + 1e-9)

        candidate = None
        if peak_m > MAG_THRESH and purity > PURITY_MIN:
            for name, lo, hi in BANDS:
                if lo <= peak_f < hi:
                    candidate = name
                    break

        if candidate is not None and candidate == self._last_band:
            self._streak += 1
        else:
            self._streak = 1 if candidate is not None else 0
        self._last_band = candidate

        decision = candidate if (candidate and self._streak >= HOLD_FRAMES) else None
        return decision, peak_f, peak_m, mag


class Car:
    """Differential-drive wrapper; inert with --no-robot."""

    def __init__(self, enabled):
        self.enabled = enabled
        self.dm = None
        self._le = None
        self._last = (None, None)

    def connect(self):
        if not self.enabled:
            print("--no-robot: not connecting to the car.")
            return
        import legoeducation as le
        from lelib import doubleMotor
        self._le = le
        color = getattr(le, f"LEGO_COLOR_{CARD_COLOR_NAME}") if CARD_COLOR_NAME else None
        self.dm = doubleMotor()
        print("Connecting to Double Motor...")
        self.dm.connect(card_serial=CARD_SERIAL, card_color=color)
        print("Connected.")

    def drive(self, left, right):
        l = int(round(max(-100, min(100, left))))
        r = int(round(max(-100, min(100, right))))
        if (l, r) == self._last:
            return                      # don't spam identical BLE commands
        self._last = (l, r)
        if not self.enabled:
            return
        le = self._le
        if l == 0:
            self.dm.motor_stop(motor=le.MOTOR_LEFT)
        else:
            self.dm.motor_run(motor=le.MOTOR_LEFT, speed=LEFT_SIGN * l, blocking=False)
        if r == 0:
            self.dm.motor_stop(motor=le.MOTOR_RIGHT)
        else:
            self.dm.motor_run(motor=le.MOTOR_RIGHT, speed=RIGHT_SIGN * r, blocking=False)

    def stop(self):
        self.drive(0, 0)

    def disconnect(self):
        if self.enabled and self.dm is not None:
            self.stop()
            self.dm.disconnect()


class Policy:
    """Turns the stream of decisions into wheel speeds (thread-safe)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.speed = 0.0
        self.turn = 0            # -1 left, 0 straight, +1 right
        self.last_decision = "-"
        self.last_heard = time.monotonic()

    def update(self, decision):
        now = time.monotonic()
        with self.lock:
            if decision == "STOP":
                self.speed = 0.0
                self.turn = 0
            elif decision == "FASTER":
                # Dead-man throttle: moves only while the whistle sounds.
                # Kick past static friction, then ramp toward MAX_SPEED.
                self.speed = FORWARD_KICK if self.speed <= 0 else \
                    min(MAX_SPEED, self.speed + ACCEL_STEP)
                self.turn = 0
            elif decision == "REVERSE":
                self.speed = -REVERSE_KICK if self.speed >= 0 else \
                    max(-MAX_REVERSE, self.speed - REVERSE_STEP)
                self.turn = 0
            elif decision == "LEFT":
                self.turn = -1
            elif decision == "RIGHT":
                self.turn = +1
            else:                          # no whistle this chunk
                self.turn = 0              # turns only last while whistling
                if now - self.last_heard > RELEASE_GRACE_S:
                    self.speed = 0.0       # dead-man: silence = stop
                return
            self.last_decision = decision
            self.last_heard = now

    def wheels(self):
        with self.lock:
            l = self.speed + TURN_SIGN * self.turn * TURN_SPEED
            r = self.speed - TURN_SIGN * self.turn * TURN_SPEED
            return l, r, self.speed, self.turn, self.last_decision


def start_audio(detector, policy, viz_q, on_decision=None):
    """Open the PyAudio input stream; classification happens in its callback.
    on_decision(decision) is called for every chunk (worldcup.py hooks this)."""
    pa = pyaudio.PyAudio()

    def callback(data, frame_count, time_info, status):
        samples = np.frombuffer(data, dtype=np.float32)
        decision, peak_f, peak_m, mag = detector.process(samples)
        policy.update(decision)
        if on_decision is not None:
            on_decision(decision)
        try:
            viz_q.put_nowait((samples, mag, decision, peak_f, peak_m))
        except queue.Full:
            pass
        return (None, pyaudio.paContinue)

    stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=RATE,
                     input=True, frames_per_buffer=CHUNK,
                     stream_callback=callback)
    stream.start_stream()
    return pa, stream


def run_ui(policy, viz_q, status_fn=None, on_close=None):
    """Live waveform + spectrum + decision display (blocks until closed).
    status_fn() may return an extra status line (worldcup uses this)."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    freqs = np.fft.rfftfreq(CHUNK, 1.0 / RATE)
    fmask = freqs <= 4000

    fig, (ax_w, ax_s) = plt.subplots(2, 1, figsize=(10, 7))
    fig.canvas.manager.set_window_title("Whistle car")
    fig.subplots_adjust(hspace=0.45, top=0.86)

    wave_line, = ax_w.plot(np.zeros(CHUNK), linewidth=0.8)
    ax_w.set_ylim(-0.5, 0.5)
    ax_w.set_xlim(0, CHUNK)
    ax_w.set_title("Waveform (latest chunk)")
    ax_w.set_xlabel("sample")

    spec_line, = ax_s.plot(freqs[fmask], np.zeros(fmask.sum()), linewidth=1.0)
    peak_dot, = ax_s.plot([], [], "ro", markersize=8)
    band_colors = {"STOP": "#d62728", "LEFT": "#1f77b4",
                   "RIGHT": "#2ca02c", "FASTER": "#ff7f0e",
                   "REVERSE": "#9467bd"}
    for name, lo, hi in BANDS:
        ax_s.axvspan(lo, hi, alpha=0.12, color=band_colors[name])
        ax_s.text((lo + hi) / 2, 0.93, name, transform=ax_s.get_xaxis_transform(),
                  ha="center", fontsize=9, color=band_colors[name])
    ax_s.axhline(MAG_THRESH, color="gray", linestyle="--", linewidth=1)
    ax_s.set_ylim(0, 40)
    ax_s.set_xlim(0, 4000)
    ax_s.set_title("Spectrum -- shaded = command bands, dashed = threshold")
    ax_s.set_xlabel("frequency (Hz)")

    title = fig.suptitle("listening...", fontsize=14)

    def update(_):
        latest = None
        while True:                       # drain to the newest chunk
            try:
                latest = viz_q.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            samples, mag, decision, peak_f, peak_m = latest
            wave_line.set_ydata(samples)
            spec_line.set_ydata(mag[fmask])
            if decision:
                peak_dot.set_data([peak_f], [min(peak_m, 39)])
            else:
                peak_dot.set_data([], [])
        l, r, speed, turn, last = policy.wheels()
        turn_s = {0: "straight", -1: "LEFT", 1: "RIGHT"}[turn]
        line = f"decision: {last}    speed: {speed:.0f}%   {turn_s}    wheels L{l:+.0f} R{r:+.0f}"
        if status_fn is not None:
            line += "\n" + status_fn()
        title.set_text(line)
        return wave_line, spec_line, peak_dot, title

    ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    if on_close is not None:
        fig.canvas.mpl_connect("close_event", lambda e: on_close())
    plt.show()
    return ani


def main():
    ap = argparse.ArgumentParser(description="Whistle-controlled car")
    ap.add_argument("--no-robot", action="store_true",
                    help="run audio + policy without connecting to the car")
    ap.add_argument("--publish", metavar="TOPIC", default=None,
                    help="also publish every decision to this MQTT topic")
    args = ap.parse_args()

    car = Car(enabled=not args.no_robot)
    car.connect()

    mqtt_client = None
    on_decision = None
    if args.publish:
        from mqttlib import MQTTClient
        mqtt_client = MQTTClient()
        mqtt_client.connect()
        last_sent = [None]

        def on_decision(decision):     # only publish changes, not every chunk
            if decision != last_sent[0]:
                last_sent[0] = decision
                mqtt_client.publish(args.publish, decision or "NONE")
        print(f"Publishing decisions to MQTT topic '{args.publish}'")

    detector = WhistleDetector()
    policy = Policy()
    viz_q = queue.Queue(maxsize=8)

    running = [True]

    def control_loop():
        while running[0]:
            l, r, *_ = policy.wheels()
            car.drive(l, r)
            time.sleep(1.0 / CMD_HZ)

    pa, stream = start_audio(detector, policy, viz_q, on_decision)
    threading.Thread(target=control_loop, daemon=True).start()

    try:
        run_ui(policy, viz_q)
    finally:
        running[0] = False
        time.sleep(0.1)
        stream.stop_stream()
        stream.close()
        pa.terminate()
        car.disconnect()
        if mqtt_client is not None:
            mqtt_client.disconnect()


if __name__ == "__main__":
    main()
