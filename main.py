import queue
import signal
import sys
import threading
import time

import numpy as np

from audio_listener import start_audio_stream
from vad_engine import detect_speech
from stt_engine import transcribe_audio

from word_parser import (
    extract_words,
    is_filler,
    # diff_new_words replaced by HypothesisManager
)

from hypothesis_manager import HypothesisManager

from translator_engine import translate_word

from token_buffer import TokenBuffer

from silence_manager import SilenceManager

from correction_engine import reconstruct_sentence

from output_renderer import (
    update_live,            # replaces display_output(mode="LIVE", ...)
    finalize                # replaces the raw print() block
)


# =====================================================
# GLOBAL QUEUES
# =====================================================

audio_queue = queue.Queue()

text_queue = queue.Queue()


# =====================================================
# GRACEFUL SHUTDOWN FLAG
# =====================================================

_shutdown = threading.Event()


# =====================================================
# TOKEN BUFFER
# =====================================================

token_buffer = TokenBuffer()


# =====================================================
# SILENCE MANAGER
# =====================================================

silence_manager = SilenceManager(
    silence_threshold=2.0   # raised from 1.0 — stops single-word flushes
)


# =====================================================
# STREAMING WINDOW SETTINGS
# =====================================================

SAMPLE_RATE = 16000
ROLLING_WINDOW_SEC = 3.0
OVERLAP_SEC = 1.0

ROLLING_WINDOW_SAMPLES = int(
    SAMPLE_RATE * ROLLING_WINDOW_SEC
)
OVERLAP_SAMPLES = int(
    SAMPLE_RATE * OVERLAP_SEC
)


# =====================================================
# AUDIO THREAD
# =====================================================

def audio_capture_worker():
    """
    Continuously captures microphone audio
    and pushes chunks into audio queue.
    Signals shutdown on fatal mic error.
    """
    try:
        start_audio_stream(audio_queue)
    except Exception as e:
        print(f"\n[Audio] Fatal error: {e}", file=sys.stderr)
        _shutdown.set()


# =====================================================
# SPEECH RECOGNITION THREAD
# =====================================================

def speech_recognition_worker():
    """
    Audio -> Hindi text

    Changes from V1:
    - Blocking queue.get(timeout) replaces busy-wait polling
    - Buffer is always trimmed (even on no-speech) to prevent
      unbounded growth
    - try/except around every inference call so a model error
      doesn't silently kill the thread
    """

    while not _shutdown.is_set():

        # BLOCKING PULL — audio_listener now provides fixed-size windows
        try:
            window = audio_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        try:
            # DETECT SPEECH
            speech_active = detect_speech(window)

            if not speech_active:
                continue

            # SPEECH TO TEXT
            hindi_text = transcribe_audio(window)

            if hindi_text:
                text_queue.put(hindi_text["text"])
                silence_manager.update_activity()

        except Exception as e:
            print(f"[STT] Error in recognition cycle: {e}", file=sys.stderr)
            continue


# =====================================================
# WORD PROCESSING THREAD
# =====================================================

def word_processing_worker():
    """
    Hindi text
    -> diff new words (replaces single last_word check)
    -> filler detection
    -> translate
    -> token buffer
    -> live subtitle update (full accumulated lists)
    -> silence / filler triggered finalization

    Changes from V1:
    - Blocking queue.get(timeout) replaces busy-wait polling
    - diff_new_words() replaces `if word == last_word: continue`
    - Accumulated word lists passed to update_live() on every token
      so the renderer always has full sentence context
    - Finalization uses should_finalize(token_count) which enforces
      the 2 s silence threshold AND the min 3-token guard together
    - finalize() replaces the raw print() block
    - try/except around every stage so one bad token doesn't crash
      the whole pipeline
    """

    # Hypothesis manager handles incremental diffs from Whisper
    manager = HypothesisManager()

    # Running sentence state fed to the renderer on every new token
    live_hindi_words = []
    live_english_words = []

    while not _shutdown.is_set():

        filler_detected = False

        # -------------------------------------------------------
        # PULL NEXT TRANSCRIPTION — non-blocking with timeout
        # The `else` block only runs when we actually got an item
        # -------------------------------------------------------

        try:
            hindi_text = text_queue.get(timeout=0.05)
        except queue.Empty:
            pass    # no new text this cycle — still check finalization
        else:
            try:
                all_words = extract_words(hindi_text)

                # -----------------------------------------------
                # INCREMENTAL DIFF — only process genuinely new
                # words vs the previous overlapping window.
                # This replaces the V1 `if word == last_word`
                # single-word memory with a proper hypothesis diff.
                # -----------------------------------------------

                remove_count, replacement_words = manager.update_and_diff(
                    all_words
                )

                # Build preview Hindi hypothesis and render it BEFORE translating.
                start_index = len(token_buffer) - remove_count if remove_count else len(token_buffer)

                # Renderer step: show Hindi hypothesis immediately.
                live_hindi_words = (
                    live_hindi_words[:start_index]
                    + replacement_words
                )
                # Note: live_english_words remains the previous state until
                # translations are available.
                update_live(live_hindi_words, live_english_words, stable_count=start_index)

                # TRANSLATOR STEP: translate replacement words and apply
                # them to the token buffer once translations complete.
                replacement_tokens = []
                for word in replacement_words:
                    if is_filler(word):
                        filler_detected = True
                        break

                    english_word = translate_word(word)

                    replacement_tokens.append({
                        "hindi": word,
                        "english": english_word,
                    })

                if replacement_tokens or remove_count:
                    token_buffer.replace_tokens(start_index, replacement_tokens)

                    # Update live lists to reflect the buffer state (now with translations)
                    live_hindi_words = (
                        live_hindi_words[:start_index]
                        + [t["hindi"] for t in replacement_tokens]
                    )
                    live_english_words = (
                        live_english_words[:start_index]
                        + [t["english"] for t in replacement_tokens]
                    )

                    # Reflect the translated result immediately
                    update_live(live_hindi_words, live_english_words, stable_count=start_index)

            except Exception as e:
                print(f"[Processing] Word error: {e}", file=sys.stderr)

        # =================================================
        # FINALIZATION LOGIC
        #
        # should_finalize() handles all three conditions:
        #   1. silence_duration >= 2.0 s
        #   2. token_count >= min_tokens (3)   ← guards against
        #      single-word flushes even if silence is detected
        #   3. hard_timeout (45 s) for continuous speech
        # =================================================

        semantic_tokens = token_buffer.get_tokens()

        if (
            filler_detected
            or silence_manager.should_finalize(semantic_tokens)
        ):

            if semantic_tokens:

                # ---------------------------------
                # RECONSTRUCT FINAL SENTENCE
                # ---------------------------------

                try:
                    final_sentence = reconstruct_sentence(
                        semantic_tokens
                    )
                except Exception as e:
                    # Correction engine failure — fall back to raw join
                    final_sentence = " ".join(
                        t["english"] for t in semantic_tokens
                    )
                    print(
                        f"[Correction] Error, using raw join: {e}",
                        file=sys.stderr
                    )

                raw_hindi = " ".join(
                    t["hindi"] for t in semantic_tokens
                )
                raw_english = " ".join(
                    t["english"] for t in semantic_tokens
                )

                # ---------------------------------
                # FINAL SUBTITLE OUTPUT
                # Replaces the raw print() block —
                # renderer moves sentence to history
                # and rolls the display
                # ---------------------------------

                finalize(raw_hindi, raw_english, final_sentence)

                # ---------------------------------
                # RESET ALL STATE
                # ---------------------------------

                token_buffer.clear()
                manager.prev_words = []
                live_hindi_words = []
                live_english_words = []

            silence_manager.reset()


# =====================================================
# SIGNAL HANDLER — clean Ctrl-C / SIGTERM exit
# =====================================================

def _handle_exit(sig, frame):
    print("\n[System] Shutting down...", file=sys.stderr)
    _shutdown.set()
    sys.exit(0)


# =====================================================
# MAIN
# =====================================================

def main():

    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

    print("\n====================================")
    print("Realtime Hindi Translator V1")
    print("====================================\n")

    # ---------------------------------
    # THREADS
    # ---------------------------------

    audio_thread = threading.Thread(
        target=audio_capture_worker,
        daemon=True
    )

    stt_thread = threading.Thread(
        target=speech_recognition_worker,
        daemon=True
    )

    processing_thread = threading.Thread(
        target=word_processing_worker,
        daemon=True
    )

    # ---------------------------------
    # START THREADS
    # ---------------------------------

    audio_thread.start()
    stt_thread.start()
    processing_thread.start()

    # ---------------------------------
    # KEEP ALIVE UNTIL SHUTDOWN EVENT
    # Replaced infinite time.sleep(1) with
    # a flag-aware loop so SIGTERM/SIGINT
    # can break cleanly without blocking
    # ---------------------------------

    while not _shutdown.is_set():
        time.sleep(0.5)


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()
