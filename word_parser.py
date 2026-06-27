import re
import warnings


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

        if word in FILLERS:
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


# -----------------------------
# INCREMENTAL WORD DIFF
# -----------------------------
def diff_new_words(prev_words, new_words):
    """
    Given the previous Whisper hypothesis (prev_words)
    and the new hypothesis (new_words) from the next
    rolling window, return ONLY the words that are
    genuinely new — i.e. not already seen in the
    overlap region.

    This prevents re-processing the same words that
    appear in the overlapping audio between rolling
    windows, which was the root cause of duplicate
    token fragments in V1.

    Example:
        prev = ["मैं", "बाज़ार"]
        new  = ["मैं", "बाज़ार", "गया", "था"]
        → returns ["गया", "था"]

    If no overlap is found, the full new_words list
    is returned (conservative: better to duplicate
    one word than to drop real new content).
    """
    warnings.warn(
        "diff_new_words() is deprecated — use HypothesisManager in hypothesis_manager.py",
        DeprecationWarning,
        stacklevel=2,
    )
    if not prev_words:
        # No previous hypothesis — nothing to remove, all words are new
        return 0, new_words

    if not new_words:
        # No new hypothesis — remove entire previous suffix
        return len(prev_words), []

    # Find the longest suffix of prev_words that matches
    # a prefix of new_words (the overlap). We'll remove the
    # non-overlapping tail of prev_words and replace it with
    # the non-overlapping tail of new_words.
    max_check = min(len(prev_words), len(new_words))

    overlap_len = 0

    for i in range(max_check, 0, -1):
        if prev_words[-i:] == new_words[:i]:
            overlap_len = i
            break

    remove_count = len(prev_words) - overlap_len
    replacement_words = new_words[overlap_len:]

    return remove_count, replacement_words
