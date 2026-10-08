"""
FFT whistle detector from Day 4, trimmed to what the game needs: a HIGH
whistle (2000-3500 Hz, loud, pure, held ~140 ms) is the serve signal.
"""

import threading

import numpy as np
import pyaudio

RATE, CHUNK = 44100, 2048
BAND_LO, BAND_HI = 2000.0, 3500.0
MAG_THRESH, PURITY_MIN, HOLD_FRAMES = 4.0, 8.0, 3


class WhistleListener:
    """Runs PyAudio in the background; .pop() returns True once per whistle."""

    def __init__(self):
        self.window = np.hanning(CHUNK)
        self.freqs = np.fft.rfftfreq(CHUNK, 1.0 / RATE)
        self.in_band = (self.freqs >= 500) & (self.freqs <= BAND_HI)
        self.target = (self.freqs >= BAND_LO) & (self.freqs <= BAND_HI)
        self._streak = 0
        self._fired = False
        self._lock = threading.Lock()
        self.level = 0.0            # latest in-band peak, for the HUD
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True,
                                   frames_per_buffer=CHUNK, stream_callback=self._cb)
        self.stream.start_stream()

    def _cb(self, data, n, t, status):
        x = np.frombuffer(data, dtype=np.float32)
        mag = np.abs(np.fft.rfft(x * self.window))
        i = int(np.argmax(np.where(self.in_band, mag, 0.0)))
        peak_m = float(mag[i])
        purity = peak_m / (float(mag[self.in_band].mean()) + 1e-9)
        ok = peak_m > MAG_THRESH and purity > PURITY_MIN and bool(self.target[i])
        with self._lock:
            self.level = peak_m
            self._streak = self._streak + 1 if ok else 0
            if self._streak == HOLD_FRAMES:
                self._fired = True
        return (None, pyaudio.paContinue)

    def pop(self):
        with self._lock:
            f, self._fired = self._fired, False
        return f

    def close(self):
        self.stream.stop_stream(); self.stream.close(); self.pa.terminate()
