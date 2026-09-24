"""
Tiny tone-sequence player for the World Cup win/lose songs (PyAudio out).

    from songs import play_victory, play_death
"""

import numpy as np
import pyaudio

RATE = 44100

# (frequency Hz, duration s); 0 Hz = rest
VICTORY = [(523, 0.15), (659, 0.15), (784, 0.15), (1047, 0.4), (784, 0.12), (1047, 0.6)]
DEATH = [(330, 0.35), (311, 0.35), (294, 0.35), (277, 1.0)]  # sad trombone slide


def _render(notes):
    parts = []
    for f, d in notes:
        n = int(RATE * d)
        t = np.arange(n) / RATE
        tone = 0.4 * np.sin(2 * np.pi * f * t) if f > 0 else np.zeros(n)
        env = np.minimum(1.0, np.minimum(np.arange(n), np.arange(n)[::-1]) / (0.02 * RATE))
        parts.append((tone * env).astype(np.float32))  # fade in/out kills clicks
    return np.concatenate(parts)


def _play(notes):
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=RATE, output=True)
    stream.write(_render(notes).tobytes())
    stream.stop_stream()
    stream.close()
    pa.terminate()


def play_victory():
    _play(VICTORY)


def play_death():
    _play(DEATH)


if __name__ == "__main__":
    print("victory:")
    play_victory()
    print("death:")
    play_death()
