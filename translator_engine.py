from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import queue
import threading
import collections
from typing import Optional, Callable
import time
import sys


# =====================================================
# MODEL
# =====================================================

MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"

import torch
_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Translator] Initializing IndicTrans2 (200M) on device: {_device.upper()}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    ).to(_device)
except Exception as e:
    print(f"[Translator] Failed to load IndicTrans2 model: {e}", file=sys.stderr)
    print("[Translator] Check internet connection or HuggingFace cache.", file=sys.stderr)
    sys.exit(1)

_model_lock = threading.Lock()


# =====================================================
# GENERATION SETTINGS
# =====================================================

# Max output tokens for all generation calls.
# V1 left this unset, which silently capped output
# at 20 tokens (the model-agnostic default) and
# caused the UserWarning visible in every run.
# 128 covers any realistic sentence length while
# preventing runaway generation.
_MAX_NEW_TOKENS = 128

# IndicTrans2 tag prefix format.
# All input strings must start with this pair —
# it tells the model which script to translate
# from (hin_Deva = Hindi Devanagari) and to
# (eng_Latn = English Latin script).
_PREFIX = "hin_Deva eng_Latn"


# =====================================================
# CACHES
# =====================================================

# Caches are instantiated below after LRUCache definition.


# Simple thread-safe LRU cache for phrase translations.
class LRUCache:
    def __init__(self, maxsize=200):
        self.maxsize = maxsize
        self.lock = threading.Lock()
        self.data = collections.OrderedDict()

    def get(self, key):
        with self.lock:
            try:
                val = self.data.pop(key)
                # move to end (most-recent)
                self.data[key] = val
                return val
            except KeyError:
                return None

    def set(self, key, value):
        with self.lock:
            if key in self.data:
                self.data.pop(key)
            self.data[key] = value
            if len(self.data) > self.maxsize:
                # pop least-recently-used
                self.data.popitem(last=False)

    def contains(self, key):
        with self.lock:
            return key in self.data


# phrase-level cache using LRU
_phrase_cache = LRUCache(maxsize=512)



# =====================================================
# ASYNC PHRASE QUEUE
# =====================================================

# Items are tuples: (hindi_sentence, optional callback)
_phrase_queue = queue.Queue()


def enqueue_phrase(hindi_sentence: str, callback: Optional[Callable[[Optional[str]], None]] = None):
    """Enqueue a phrase for asynchronous translation. Callback is called
    with the translation (or None on failure) once available."""
    if not hindi_sentence or not hindi_sentence.strip():
        if callback:
            callback(None)
        return

    key = hindi_sentence.strip()

    # If already cached, call back immediately
    cached = _phrase_cache.get(key)
    if cached is not None:
        if callback:
            callback(cached)
        return

    _phrase_queue.put((key, callback))


def _phrase_worker():
    while True:
        try:
            key, callback = _phrase_queue.get()
        except Exception:
            time.sleep(0.1)
            continue

        if not key:
            if callback:
                callback(None)
            _phrase_queue.task_done()
            continue

        # Double-check cache to avoid duplicated work
        cached = _phrase_cache.get(key)
        if cached is not None:
            if callback:
                callback(cached)
            _phrase_queue.task_done()
            continue

        formatted = f"{_PREFIX} {key}"

        try:
            inputs = tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(_device)

            with _model_lock:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=_MAX_NEW_TOKENS,
                    num_beams=4,
                    early_stopping="never",
                    use_cache=False,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                )

            translated = tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )[0].strip()

            if translated:
                _phrase_cache.set(key, translated)
                if callback:
                    callback(translated)
            else:
                if callback:
                    callback(None)

        except Exception:
            if callback:
                callback(None)

        _phrase_queue.task_done()


# Start background worker thread
_worker_thread = threading.Thread(target=_phrase_worker, daemon=True)
_worker_thread.start()


# =====================================================
# TRANSLATE PHRASE  ← primary output-quality path
# =====================================================

def translate_phrase(hindi_sentence):
    """
    Translate a complete Hindi sentence to English.

    Called once at finalization time by
    correction_engine.reconstruct_sentence() after
    silence or a filler word ends the utterance.

    Passing the FULL sentence gives IndicTrans2 the
    grammatical context it needs to:
      • Reorder SOV → SVO  (Hindi verb-final → English verb-medial)
      • Correctly decode conjugated verb forms
        (e.g. "dunga" = "will do" only makes sense with a subject)
      • Insert articles and prepositions (to / the / a)
        that Hindi does not use but English requires

    Args:
        hindi_sentence : str — joined Hindi words from token buffer,
                         e.g. "मैं घर जाता हूँ"

    Returns:
        Translated English string, or None on error / empty input.
    """

    if not hindi_sentence or not hindi_sentence.strip():
        return None

    key = hindi_sentence.strip()

    cached = _phrase_cache.get(key)
    if cached is not None:
        return cached

    formatted = f"{_PREFIX} {key}"

    try:
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,        # prevent tokenizer error on very long sentences
            max_length=256,         # IndicTrans2 context window
        ).to(_device)

        with _model_lock:
            outputs = model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,    # Fix: was unset → 20-token silent cap
                num_beams=4,                        # beam search: better quality than greedy
                early_stopping="never",
                use_cache=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )

        translated = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0].strip()

        if translated:
            _phrase_cache.set(key, translated)
            return translated

        return None

    except Exception:
        return None
