import pyaudio
import numpy as np

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

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("[Audio] Listening...")

    while True:

        audio_data = stream.read(
            CHUNK,
            exception_on_overflow=False
        )

        audio_np = np.frombuffer(
            audio_data,
            dtype=np.int16
        )

        audio_queue.put(audio_np)