from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

# -----------------------------
# MODEL NAME
# -----------------------------
MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"

# -----------------------------
# LOAD TOKENIZER
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

# -----------------------------
# CACHE
# -----------------------------
translation_cache = {}

# -----------------------------
# TRANSLATE WORD
# -----------------------------
def translate_word(word):

    if word in translation_cache:
        return translation_cache[word]

    formatted_text = f"hin_Deva eng_Latn {word}"

    inputs = tokenizer(
        formatted_text,
        return_tensors="pt",
        padding=True
    )

    outputs = model.generate(**inputs)

    translated_text = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )[0]

    translation_cache[word] = translated_text

    return translated_text