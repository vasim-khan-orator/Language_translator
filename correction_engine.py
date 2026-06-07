# -----------------------------
# RECONSTRUCT SENTENCE
# -----------------------------
def reconstruct_sentence(tokens):

    english_words = []

    for token in tokens:

        english_words.append(
            token["english"]
        )

    raw_sentence = " ".join(english_words)

    corrected_sentence = apply_rules(
        raw_sentence
    )

    return corrected_sentence


# -----------------------------
# RULE ENGINE
# -----------------------------
def apply_rules(sentence):

    words = sentence.lower().split()

    # -----------------------------
    # SIMPLE TIME CORRECTION
    # -----------------------------
    if "tomorrow" in words and "went" in words:

        words.remove("tomorrow")

        words.insert(0, "yesterday")

    # -----------------------------
    # MARKET FIX
    # -----------------------------
    if "market" in words:

        index = words.index("market")

        words.insert(index, "the")

        if "to" not in words:
            words.insert(index, "to")

    # -----------------------------
    # VERB REORDERING
    # -----------------------------
    if "went" in words:

        words.remove("went")

        if "i" in words:

            i_index = words.index("i")

            words.insert(i_index + 1, "went")

    # -----------------------------
    # CAPITALIZATION
    # -----------------------------
    final_sentence = " ".join(words)

    final_sentence = final_sentence.capitalize()

    return final_sentence