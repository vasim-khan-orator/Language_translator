FILLERS = [
    "hmm",
    "umm",
    "matlab",
    "aaa",
    "aaaa",
    "uh"
]

# -----------------------------
# EXTRACT WORDS
# -----------------------------
def extract_words(text):

    words = text.lower().strip().split()

    cleaned_words = []

    for word in words:

        if word in FILLERS:
            continue

        cleaned_words.append(word)

    return cleaned_words


# -----------------------------
# CHECK FILLER
# -----------------------------
def is_filler(word):

    return word.lower().strip() in FILLERS