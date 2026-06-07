import torch
from silero_vad import load_silero_vad
from silero_vad import get_speech_timestamps

# -----------------------------
# LOAD MODEL
# -----------------------------
model = load_silero_vad()

# -----------------------------
# DETECT SPEECH
# -----------------------------
def detect_speech(audio_chunk):

    audio_tensor = torch.tensor(
        audio_chunk,
        dtype=torch.float32
    ) / 32768.0

    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=16000
    )

    return len(speech_timestamps) > 0