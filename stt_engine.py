from faster_whisper import WhisperModel
import numpy as np
import re
import sys


import torch

# -----------------------------
# LOAD MODEL
# -----------------------------
try:
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _compute_type = "float16" if _device == "cuda" else "int8"
    print(f"[STT] Initializing Whisper ('small') on device: {_device.upper()} ({_compute_type})...")
    model = WhisperModel(
        "small",
        device=_device,
        compute_type=_compute_type
    )
except Exception as e:
    print(f"[STT] Failed to load Whisper model: {e}", file=sys.stderr)
    print("[STT] Check internet connection or model cache.", file=sys.stderr)
    sys.exit(1)


# =====================================================
# THRESHOLDS
# =====================================================

# Minimum confidence Whisper must have that the
# audio is Hindi. Below this it's likely noise or
# a different language — discard the transcription.
_MIN_LANGUAGE_PROB = 0.70

# Per-segment: discard if no_speech_prob exceeds
# this value. Filters silence, breath, and ambient
# noise frames that pass VAD.
_MAX_NO_SPEECH_PROB = 0.50

# Minimum fraction of characters in the transcription
# that must be Devanagari. Catches hallucinations
# where Whisper outputs Hebrew, Latin, or mixed
# script garbage (e.g. "מהड़ א" seen in V1 run).
_MIN_DEVANAGARI_RATIO = 0.40

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


# =====================================================
# SCRIPT VALIDATION
# =====================================================

def _is_valid_hindi(text):
    """
    Returns True only if the transcribed text
    contains enough Devanagari characters to be
    genuine Hindi output.

    Pure Latin / Hebrew / symbol hallucinations
    will fail this check and be discarded before
    they reach the translation pipeline.
    """
    if not text:
        return False

    # Ignore whitespace when measuring ratio
    chars = [c for c in text if not c.isspace()]

    if not chars:
        return False

    devanagari_count = len(_DEVANAGARI_RE.findall(text))
    ratio = devanagari_count / len(chars)

    return ratio >= _MIN_DEVANAGARI_RATIO


# =====================================================
# TRANSCRIBE AUDIO
# =====================================================

def transcribe_audio(audio_chunk):
    """
    Transcribes a 16 kHz int16 audio chunk to
    Hindi text (Devanagari).

    Returns None (instead of garbage) when:
      - Whisper's language confidence is below
        _MIN_LANGUAGE_PROB  (likely noise or wrong
        language)
      - Every segment is flagged as non-speech
      - The output contains no Devanagari script
        (hallucination — Hebrew, Latin, etc.)
    """

    if isinstance(audio_chunk, np.ndarray) and audio_chunk.dtype == np.int16:
        audio_float = audio_chunk.astype(np.float32) / 32768.0
    else:
        audio_float = np.asarray(audio_chunk, dtype=np.float32)

    segments, info = model.transcribe(
        audio_float,
        language="hi",
        beam_size=5,

        max_new_tokens=128,

        condition_on_previous_text=False,

        no_speech_threshold=_MAX_NO_SPEECH_PROB,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,

        # Anti-hallucination: penalize repeated tokens so
        # Whisper doesn't get stuck in loops like
        # "एक लिएक लिएक लिए..."
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,

        # Suppress hallucinated text during silence gaps
        # within the audio. If a segment's audio is mostly
        # silence but Whisper still generates text, skip it.
        hallucination_silence_threshold=0.5,
    )

    # -------------------------------------------------
    # LANGUAGE CONFIDENCE GATE
    # -------------------------------------------------
    # info is computed eagerly by faster-whisper before
    # any segment is generated, so this check is safe
    # to do before iterating the segments generator.
    if info.language_probability < _MIN_LANGUAGE_PROB:
        return None

    # -------------------------------------------------
    # COLLECT VALID SEGMENTS
    # -------------------------------------------------
    final_text = ""

    for segment in segments:

        # Drop individual segments Whisper marks as
        # non-speech even if the overall language
        # confidence was acceptable.
        if segment.no_speech_prob > _MAX_NO_SPEECH_PROB:
            continue

        final_text += segment.text.strip() + " "

    final_text = final_text.strip()

    # -------------------------------------------------
    # REPETITION COLLAPSE
    # -------------------------------------------------
    # Catch any remaining Whisper loops that survived
    # the generation-level repetition_penalty.
    # e.g. "एक लिए एक लिए एक लिए" -> "एक लिए"
    final_text = _collapse_repeats(final_text)

    # -------------------------------------------------
    # SCRIPT VALIDATION
    # -------------------------------------------------
    if not _is_valid_hindi(final_text):
        return None

    if final_text:
        return {
            "text": final_text,
            "confidence": float(info.language_probability),
            "is_final": False,
        }

    return None


def _collapse_repeats(text):
    """
    Detect and collapse repeated word sequences in text.
    Checks for repeated n-grams (1 to 5 words) and
    removes duplicate occurrences.
    """
    if not text:
        return text

    words = text.split()
    if len(words) < 4:
        return text

    # Check for repeated n-grams of length 1 to 5
    for n in range(1, 6):
        if len(words) < n * 2:
            continue

        cleaned = []
        i = 0
        while i < len(words):
            # Check if the next n words repeat the current n words
            if i + n * 2 <= len(words):
                current = words[i:i + n]
                next_group = words[i + n:i + n * 2]
                if current == next_group:
                    # Found a repeat; keep current, skip duplicates
                    cleaned.extend(current)
                    i += n
                    # Skip all consecutive repetitions
                    while i + n <= len(words) and words[i:i + n] == current:
                        i += n
                    continue
            cleaned.append(words[i])
            i += 1
        words = cleaned

    return " ".join(words)
