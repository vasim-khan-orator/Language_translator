import re


# -----------------------------
# FILLER WORDS (Devanagari)
# -----------------------------
# Whisper transcribes Hindi in Devanagari script,
# so fillers must match Devanagari spellings.
# Roman spellings ("hmm", "matlab") will NEVER
# match Whisper output and silently break detection.
# -----------------------------
FILLERS = [
    "हम्म",     # hmm
    "हम",       # hm (short)
    "उम्म",     # umm
    "उम",       # um (short)
    "मतलब",    # matlab — "meaning", used as filler
    "यानी",     # yaani  — "i.e.", used as filler
    "आ",        # aa — hesitation sound
    "आं",       # aaN — nasal hesitation
    "अं",       # aN  — short hesitation
    "अच्छा",    # achha — "okay", used as filler
    "ठीक",      # theek — "okay/fine", used as filler
    "बस",       # bas — "just/done", used as filler
    "वो",       # vo  — "that/um", used as filler mid-sentence
    "हाँ",      # haan — "yes", used as filler
]


# -----------------------------
# PUNCTUATION PATTERN
# -----------------------------
# Whisper often attaches punctuation directly to
# the word boundary, e.g. "है।" or "गया," — strip
# these before any matching or translation step.
# Includes Hindi danda (।), double danda (॥),
# and standard ASCII punctuation.
# -----------------------------
_PUNCT = re.compile(r"^[।॥\.,!?;:'\"()\-\s]+|[।॥\.,!?;:'\"()\-\s]+$")


# -----------------------------
# CLEAN WORD
# -----------------------------
def clean_word(word):
    """
    Strip leading/trailing punctuation that
    Whisper may attach to word boundaries.
    Does NOT lowercase — Devanagari has no case,
    and lowercasing corrupts Devanagari characters.
    """
    return _PUNCT.sub("", word).strip()


# -----------------------------
# EXTRACT WORDS
# -----------------------------
def extract_words(text):
    """
    Split transcript text into cleaned words,
    removing empty strings and filler words.
    """
    raw_words = text.strip().split()

    cleaned_words = []

    for word in raw_words:

        word = clean_word(word)

        if not word:
            continue

        cleaned_words.append(word)

    return cleaned_words


# -----------------------------
# CHECK FILLER
# -----------------------------
def is_filler(word):
    """
    Returns True if the word is a known filler
    after stripping attached punctuation.
    """
    return clean_word(word) in FILLERS


