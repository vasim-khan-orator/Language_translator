from translator_engine import translate_phrase
from context_manager import update_current, finalize_current


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
    Accepts raw Hindi string or list of tokens/strings.
    """

    if not tokens:
        return ""

    if isinstance(tokens, str):
        hindi_sentence = tokens.strip()
        fallback = hindi_sentence
    else:
        hindi_sentence = " ".join(
            t["hindi"] if isinstance(t, dict) else str(t) for t in tokens
        ).strip()
        fallback = " ".join(
            t.get("english", t.get("hindi", "")) if isinstance(t, dict) else str(t) for t in tokens
        ).strip()

    if not hindi_sentence:
        return ""

    update_current(hindi_sentence)

    translated = translate_phrase(hindi_sentence)

    if not translated:
        res = fallback.capitalize() if fallback else ""
        finalize_current(res)
        return res

    # Capitalize first letter as a safety step in case
    # the model returns a lowercase-first string.
    res = translated[0].upper() + translated[1:]
    finalize_current(res)
    return res
