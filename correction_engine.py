from translator_engine import translate_phrase


# =====================================================
# RECONSTRUCT SENTENCE
# =====================================================
#
# V1 approach (removed):
#   - Join per-word English translations (already wrong)
#   - Run 3 hardcoded keyword rules to patch SOV order
#   Result: only worked for one demo sentence; produced
#   "Alive kill dunga" / "In tax four khar go i am" for
#   everything else.
#
# V2 approach (this file):
#   - Join all HINDI tokens from the buffer into one
#     full sentence string
#   - Pass that one sentence to translate_phrase()
#   - IndicTrans2 now has full grammatical context:
#       • SOV → SVO reordering done by the model
#       • Verb conjugations decoded correctly
#       • Articles (to / the / a) inserted automatically
#   - apply_rules() is gone — no manual patches needed
# =====================================================

def reconstruct_sentence(tokens):
    """
    Translate the full buffered Hindi sentence at once.

    Args:
        tokens : list of dicts with keys "hindi" and "english"
                 (the "english" key is ignored here — translation
                  is now done on the joined Hindi string, not
                  on the per-word English values)

    Returns:
        Grammatically correct English sentence string,
        or empty string if tokens is empty.
    """

    if not tokens:
        return ""

    # Join all buffered Hindi words into one sentence string.
    # IndicTrans2 needs the complete sentence to produce
    # correct word order — feeding it word-by-word is what
    # caused the SOV salad in V1.
    hindi_sentence = " ".join(
        token["hindi"] for token in tokens
    )

    # Single phrase-level translation call.
    # translate_phrase() handles the hin_Deva eng_Latn
    # prefix and max_new_tokens — see translator_engine.py.
    translated = translate_phrase(hindi_sentence)

    if not translated:
        # Fallback: raw English join (same as V1 output)
        # so the display never goes blank on a model error.
        fallback = " ".join(
            token["english"] for token in tokens
        )
        return fallback.capitalize()

    # Capitalize first letter as a safety step in case
    # the model returns a lowercase-first string.
    return translated[0].upper() + translated[1:]
