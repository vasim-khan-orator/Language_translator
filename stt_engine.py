from faster_whisper import WhisperModel
import numpy as np

# -----------------------------
# LOAD MODEL
# -----------------------------
model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

# -----------------------------
# TRANSCRIBE AUDIO
# -----------------------------
def transcribe_audio(audio_chunk):

    audio_float = audio_chunk.astype(np.float32) / 32768.0

    segments, info = model.transcribe(
        audio_float,
        language="hi",
        beam_size=1
    )

    final_text = ""

    for segment in segments:
        final_text += segment.text.strip() + " "

    return final_text.strip()