from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import queue
import threading
import collections
from typing import Optional, Callable
import time
import sys
import re


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
# 128 covers any realistic sentence length while
# preventing runaway generation.
_MAX_NEW_TOKENS = 128

# IndicTrans2 tag prefix format.
_PREFIX = "hin_Deva eng_Latn"


# =====================================================
# CACHES
# =====================================================

class LRUCache:
    def __init__(self, maxsize=200):
        self.maxsize = maxsize
        self.lock = threading.Lock()
        self.data = collections.OrderedDict()

    def get(self, key):
        with self.lock:
            try:
                val = self.data.pop(key)
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
                self.data.popitem(last=False)

    def contains(self, key):
        with self.lock:
            return key in self.data


_phrase_cache = LRUCache(maxsize=512)


# =====================================================
# ASYNC PHRASE QUEUE
# =====================================================

_phrase_queue = queue.Queue()


def enqueue_phrase(hindi_sentence: str, callback: Optional[Callable[[Optional[str]], None]] = None):
    """Enqueue a phrase for asynchronous translation."""
    if not hindi_sentence or not hindi_sentence.strip():
        if callback:
            callback(None)
        return

    key = hindi_sentence.strip()

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


_worker_thread = threading.Thread(target=_phrase_worker, daemon=True)
_worker_thread.start()


# =====================================================
# SENTENCE CHUNKING
# =====================================================
# IndicTrans2 200M has a 256 SentencePiece token context
# window.  A single Hindi word averages ~3-5 subword
# tokens, so ~12 Hindi words is a safe chunk size that
# stays well within the limit with room for the prefix
# tags and generation overhead.

_CHUNK_WORD_LIMIT = 12


def _split_into_chunks(text, limit=_CHUNK_WORD_LIMIT):
    """
    Split Hindi text into chunks of at most `limit` words.
    Tries to split at natural punctuation boundaries first
    (Devanagari danda). Falls back to word-limit splitting.
    """
    sentences = re.split(r'[।|?!.\n]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= limit:
            chunks.append(sentence)
        else:
            for i in range(0, len(words), limit):
                chunk = " ".join(words[i:i + limit])
                if chunk.strip():
                    chunks.append(chunk.strip())

    return chunks


# =====================================================
# TRANSLATE SINGLE CHUNK (internal)
# =====================================================

def _translate_single_chunk(hindi_text):
    """
    Translate a single chunk (guaranteed to be within
    the 256-token context window) through IndicTrans2.
    """
    formatted = f"{_PREFIX} {hindi_text}"

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

        return translated if translated else None

    except Exception:
        return None


# =====================================================
# TRANSLATE PHRASE  -- primary output-quality path
# =====================================================

def translate_phrase(hindi_sentence):
    """
    Translate a complete Hindi sentence to English.
    For long input, automatically chunks into segments
    that fit within IndicTrans2's 256-token context
    window and joins the translated results.
    """

    if not hindi_sentence or not hindi_sentence.strip():
        return None

    key = hindi_sentence.strip()

    cached = _phrase_cache.get(key)
    if cached is not None:
        return cached

    words = key.split()

    # Short sentence: translate directly (fast path)
    if len(words) <= _CHUNK_WORD_LIMIT:
        result = _translate_single_chunk(key)
        if result:
            _phrase_cache.set(key, result)
        return result

    # Long sentence: chunk, translate each, join
    chunks = _split_into_chunks(key)
    translated_parts = []

    for chunk in chunks:
        chunk_cached = _phrase_cache.get(chunk)
        if chunk_cached is not None:
            translated_parts.append(chunk_cached)
            continue

        result = _translate_single_chunk(chunk)
        if result:
            _phrase_cache.set(chunk, result)
            translated_parts.append(result)
        else:
            translated_parts.append(chunk)

    if translated_parts:
        joined = " ".join(translated_parts)
        _phrase_cache.set(key, joined)
        return joined

    return None
