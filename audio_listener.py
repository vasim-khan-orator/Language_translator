import pyaudio
import numpy as np
import time

# -----------------------------
# AUDIO SETTINGS
# -----------------------------
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
# 20ms frames @ 16kHz = 320 samples
CHUNK = 320

# -----------------------------
# START AUDIO STREAM
# -----------------------------
def start_audio_stream(audio_queue):

    def _start(window_sec=3.0, sample_rate=RATE, emit_interval_ms=250):

        p = pyaudio.PyAudio()

        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=sample_rate,
            input=True,
            frames_per_buffer=CHUNK
        )

        window_samples = int(sample_rate * window_sec)

        # Continuous ring buffer keeping the last `window_sec` seconds.
        ring_buffer = np.zeros(window_samples, dtype=np.int16)

        emit_interval = emit_interval_ms / 1000.0
        last_emit = time.time()

        print("[Audio] Listening (ring buffer)...")

        while True:

            audio_data = stream.read(
                CHUNK,
                exception_on_overflow=False
            )

            audio_np = np.frombuffer(
                audio_data,
                dtype=np.int16
            )

            # roll left by CHUNK and append new samples at the end
            if CHUNK >= window_samples:
                # chunk larger than window — keep last window_samples of this chunk
                ring_buffer[:] = audio_np[-window_samples:]
            else:
                ring_buffer = np.roll(ring_buffer, -CHUNK)
                ring_buffer[-CHUNK:] = audio_np

            # Only emit to consumers on the configured interval (e.g. every 250ms).
            now = time.time()
            if now - last_emit >= emit_interval:
                # Push a copy of the current full window into the queue so
                # consumers always receive a fixed-size, up-to-date buffer.
                audio_queue.put(ring_buffer.copy())
                last_emit = now

    # Backwards-compatible single-arg call
    _start()