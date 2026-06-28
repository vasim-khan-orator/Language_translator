import pyaudio
import numpy as np
import time
import queue

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
def start_audio_stream(audio_queue, shutdown_event=None, flush_event=None):
    """
    Continuously streams raw 20ms audio chunks (int16 numpy arrays)
    into `audio_queue`. The consumer thread accumulates chunks and
    performs VAD / STT.
    """

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("[Audio] Listening (streaming raw chunks)...")

    try:
        while not (shutdown_event and shutdown_event.is_set()):
            if flush_event and flush_event.is_set():
                while not audio_queue.empty():
                    try:
                        audio_queue.get_nowait()
                    except queue.Empty:
                        break
                flush_event.clear()

            audio_data = stream.read(
                CHUNK,
                exception_on_overflow=False
            )

            audio_np = np.frombuffer(
                audio_data,
                dtype=np.int16
            )

            # Push chunk into queue. If full, drop oldest chunk.
            try:
                audio_queue.put_nowait(audio_np)
            except queue.Full:
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    audio_queue.put_nowait(audio_np)
                except queue.Full:
                    pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("[Audio] Stream closed cleanly.")