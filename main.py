import sys

from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)

    window = MainWindow()
    window.showFullScreen()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
import queue
import threading
import time

import numpy as np

from audio_listener import start_audio_stream
from vad_engine import detect_speech
from stt_engine import transcribe_audio

from word_parser import (
    extract_words,
    is_filler
)

from translator_engine import translate_word

from token_buffer import TokenBuffer

from silence_manager import SilenceManager

from correction_engine import reconstruct_sentence

from output_renderer import display_output


# =====================================================
# GLOBAL QUEUES
# =====================================================

audio_queue = queue.Queue()

text_queue = queue.Queue()


# =====================================================
# TOKEN BUFFER
# =====================================================

token_buffer = TokenBuffer()


# =====================================================
# SILENCE MANAGER
# =====================================================

silence_manager = SilenceManager(
    silence_threshold=1.0
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
    """

    start_audio_stream(audio_queue)


# =====================================================
# SPEECH RECOGNITION THREAD
# =====================================================

def speech_recognition_worker():

    """
    Audio -> Hindi text
    """

    rolling_buffer = np.zeros(
        0,
        dtype=np.int16
    )

    while True:

        if not audio_queue.empty():

            audio_chunk = audio_queue.get()

            rolling_buffer = np.concatenate(
                [rolling_buffer, audio_chunk]
            )

            if (
                rolling_buffer.size
                < ROLLING_WINDOW_SAMPLES
            ):
                continue

            window = rolling_buffer[
                -ROLLING_WINDOW_SAMPLES:
            ]

            # ---------------------------------
            # DETECT SPEECH
            # ---------------------------------

            speech_active = detect_speech(
                window
            )

            if not speech_active:
                continue

            # ---------------------------------
            # SPEECH TO TEXT
            # ---------------------------------

            hindi_text = transcribe_audio(
                window
            )

            if hindi_text:

                text_queue.put(hindi_text)

                silence_manager.update_activity()

            rolling_buffer = window[
                -OVERLAP_SAMPLES:
            ]

        time.sleep(0.01)


# =====================================================
# WORD PROCESSING THREAD
# =====================================================

def word_processing_worker():

    """
    Hindi text
    -> words
    -> semantic translation
    -> token buffering
    -> sentence reconstruction
    """

    last_word = None

    while True:

        filler_detected = False

        # ---------------------------------
        # PROCESS STREAMING TEXT
        # ---------------------------------

        if not text_queue.empty():

            hindi_text = text_queue.get()

            hindi_words = extract_words(
                hindi_text
            )

            for word in hindi_words:

                # ---------------------------------
                # FILLER DETECTION
                # ---------------------------------

                if is_filler(word):

                    filler_detected = True
                    break

                # ---------------------------------
                # DUPLICATE FILTERING
                # ---------------------------------

                if word == last_word:
                    continue

                last_word = word

                # ---------------------------------
                # TRANSLATE SINGLE TOKEN
                # ---------------------------------

                english_word = translate_word(
                    word
                )

                # ---------------------------------
                # STORE SEMANTIC TOKEN
                # ---------------------------------

                token_buffer.add_token(
                    hindi_word=word,
                    english_word=english_word
                )

                # ---------------------------------
                # LIVE OUTPUT
                # ---------------------------------

                display_output(
                    mode="LIVE",
                    hindi=word,
                    english=english_word
                )

        # =================================================
        # FINALIZATION LOGIC
        # =================================================

        if (
            silence_manager.is_silence_detected()
            or filler_detected
        ):

            semantic_tokens = (
                token_buffer.get_tokens()
            )

            if semantic_tokens:

                # ---------------------------------
                # RECONSTRUCT FINAL SENTENCE
                # ---------------------------------

                final_sentence = reconstruct_sentence(
                    semantic_tokens
                )

                # ---------------------------------
                # BUILD RAW ENGLISH
                # ---------------------------------

                raw_english = " ".join([
                    token["english"]
                    for token in semantic_tokens
                ])

                raw_hindi = " ".join([
                    token["hindi"]
                    for token in semantic_tokens
                ])

                # ---------------------------------
                # FINAL OUTPUT
                # ---------------------------------

                print("\n================================")
                print("FINAL TRANSLATION")
                print("================================")

                print(f"Hindi      : {raw_hindi}")

                print(f"English    : {raw_english}")

                print(f"Translated : {final_sentence}")

                print("================================\n")

                # ---------------------------------
                # CLEAR BUFFERS
                # ---------------------------------

                token_buffer.clear()

                last_word = None

            silence_manager.reset()

        time.sleep(0.05)


# =====================================================
# MAIN
# =====================================================

def main():

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
    # KEEP PROGRAM RUNNING
    # ---------------------------------

    while True:
        time.sleep(1)


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()