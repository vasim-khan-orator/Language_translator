import torch
import numpy as np
import sys
from silero_vad import load_silero_vad
from silero_vad import get_speech_timestamps


# -----------------------------
# LOAD MODEL
# -----------------------------
try:
    model = load_silero_vad()
except Exception as e:
    print(f"[VAD] Failed to load Silero VAD model: {e}", file=sys.stderr)
    print("[VAD] Check internet connection or PyTorch installation.", file=sys.stderr)
    sys.exit(1)


# =====================================================
# THRESHOLDS
# =====================================================

# Silero speech probability threshold.
# Default is 0.5 — raised to 0.60 to reduce false
# positives from background noise, TV audio, and
# ambient sound that was triggering STT in V1.
_SPEECH_THRESHOLD = 0.60

# Ignore speech bursts shorter than this.
# Real spoken words are rarely under 150–200 ms.
# Filters clicks, coughs, and noise spikes that
# Silero briefly marks as speech.
_MIN_SPEECH_DURATION_MS = 100

# Don't split a speech region on silences shorter
# than this. Prevents micro-pauses between syllables
# from fragmenting a single word into multiple
# speech timestamps.
_MIN_SILENCE_DURATION_MS = 300

# Minimum fraction of the audio window that must
# be speech before we pass it to STT.
_MIN_SPEECH_RATIO = 0.10


# =====================================================
# DETECT SPEECH
# =====================================================

def detect_speech(audio_chunk):
    """
    Returns True only when the audio chunk contains
    a meaningful amount of real speech — not just a
    noise spike or brief ambient sound.

    Two-stage gate:
      1. Silero VAD with raised threshold + duration
         filters (rejects short noise bursts).
      2. Speech-ratio check (rejects windows where
         speech is present but tiny compared to the
         total duration — likely a false positive).
    """

    if isinstance(audio_chunk, np.ndarray) and audio_chunk.dtype == np.int16:
        audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32) / 32768.0
    else:
        audio_tensor = torch.as_tensor(audio_chunk, dtype=torch.float32)

    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=16000,

        # Raised from default 0.5 — less sensitive
        # to low-probability noise events.
        threshold=_SPEECH_THRESHOLD,

        # Drop speech regions shorter than 200 ms —
        # real phonemes don't come in shorter bursts.
        min_speech_duration_ms=_MIN_SPEECH_DURATION_MS,

        # Don't break continuous speech on pauses
        # shorter than 300 ms (breath, micro-pause).
        min_silence_duration_ms=_MIN_SILENCE_DURATION_MS,
    )

    # Stage 1: no speech timestamps at all
    if not speech_timestamps:
        return False

    # -------------------------------------------------
    # SPEECH RATIO GATE
    # -------------------------------------------------
    # Even with timestamps present, check that speech
    # occupies enough of the window to be worth sending
    # to Whisper. Avoids wasting STT compute on windows
    # where a brief noise spike produced one timestamp.

    total_samples = audio_tensor.shape[0]

    speech_samples = sum(
        ts["end"] - ts["start"]
        for ts in speech_timestamps
    )

    speech_ratio = speech_samples / total_samples

    # Stage 2: speech must cover >= 15% of the window
    return speech_ratio >= _MIN_SPEECH_RATIO
