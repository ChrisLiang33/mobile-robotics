"""Short game sounds through PyAudio output: ping, pong, buzz, fanfare."""

import threading

import numpy as np
import pyaudio

RATE = 44100
_lock = threading.Lock()


def _render(notes, volume=0.35):
    parts = []
    for f, d in notes:
        n = int(RATE * d)
        t = np.arange(n) / RATE
        tone = np.sin(2 * np.pi * f * t) if f > 0 else np.zeros(n)
        env = np.minimum(1.0, np.minimum(np.arange(n), np.arange(n)[::-1]) / (0.005 * RATE))
        parts.append((volume * tone * env).astype(np.float32))
    return np.concatenate(parts)


def _play(notes):
    with _lock:                       # one sound at a time, no device fights
        pa = pyaudio.PyAudio()
        try:
            s = pa.open(format=pyaudio.paFloat32, channels=1, rate=RATE, output=True)
            s.write(_render(notes).tobytes())
            s.stop_stream()
            s.close()
        finally:
            pa.terminate()


def play(name):
    """Fire-and-forget: 'ping' (your hit), 'pong' (wall return), 'miss', 'record', 'serve'."""
    notes = {
        "ping":   [(880, 0.06)],
        "pong":   [(440, 0.06)],
        "miss":   [(160, 0.25)],
        "serve":  [(660, 0.05), (0, 0.03), (660, 0.05)],
        "record": [(523, 0.1), (659, 0.1), (784, 0.1), (1047, 0.3)],
    }[name]
    threading.Thread(target=_play, args=(notes,), daemon=True).start()


if __name__ == "__main__":
    import time
    for n in ("serve", "ping", "pong", "miss", "record"):
        print(n); play(n); time.sleep(0.7)
