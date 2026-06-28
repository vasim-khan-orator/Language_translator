import queue
import signal
import sys
import threading
import time

import numpy as np

from audio_listener import start_audio_stream
from vad_engine import detect_speech
from stt_engine import transcribe_audio

from word_parser import extract_words, is_filler
from translator_engine import translate_phrase
from silence_manager import SilenceManager
from correction_engine import reconstruct_sentence
from output_renderer import update_live, finalize


# =====================================================
# GLOBAL QUEUES & FLAGS
# =====================================================

# Increased queue size to store 10 seconds of 20ms audio chunks (500 chunks)
audio_queue = queue.Queue(maxsize=500)
text_queue = queue.Queue()

_shutdown = threading.Event()
_flush_audio = threading.Event()

silence_manager = SilenceManager(
    silence_threshold=1.5   # 1.5s pause marks the end of a spoken sentence
)


# =====================================================
# AUDIO THREAD
# =====================================================

def audio_capture_worker():
    """
    Continuously captures raw 20ms microphone audio chunks
    and pushes them into audio_queue.
    """
    try:
        start_audio_stream(audio_queue, shutdown_event=_shutdown, flush_event=_flush_audio)
    except Exception as e:
        print(f"\n[Audio] Fatal error: {e}", file=sys.stderr)
        _shutdown.set()


# =====================================================
# SPEECH RECOGNITION (VAD ACCUMULATOR) THREAD
# =====================================================

def speech_recognition_worker():
    """
    State machine: IDLE -> ACCUMULATING -> FINALIZE.
    Accumulates raw audio chunks while speaking and runs Whisper
    on the complete accumulated audio, eliminating rolling-window clipping.
    """
    state = "IDLE"
    accumulated_chunks = []
    idle_window = []  # rolling 300ms window (15 chunks of 20ms) to detect speech start
    last_live_transcription_time = 0
    last_speech_time = time.time()

    while not _shutdown.is_set():
        try:
            chunk = audio_queue.get(timeout=0.05)
            got_chunk = True
        except queue.Empty:
            got_chunk = False

        now = time.time()

        if state == "IDLE":
            if got_chunk:
                idle_window.append(chunk)
                if len(idle_window) > 15:  # keep last 300ms
                    idle_window.pop(0)

                if len(idle_window) == 15:
                    window_audio = np.concatenate(idle_window)
                    if detect_speech(window_audio):
                        state = "ACCUMULATING"
                        accumulated_chunks = list(idle_window)
                        last_speech_time = now
                        last_live_transcription_time = now
                        silence_manager.reset()
                        idle_window.clear()
            continue

        elif state == "ACCUMULATING":
            if got_chunk:
                accumulated_chunks.append(chunk)

                # Check last 300ms for speech activity to keep silence timer updated
                if len(accumulated_chunks) >= 15:
                    recent_audio = np.concatenate(accumulated_chunks[-15:])
                    if detect_speech(recent_audio):
                        last_speech_time = now
                        silence_manager.update_activity()

            # Live preview update every ~750ms while speaking
            if now - last_live_transcription_time >= 0.75 and len(accumulated_chunks) >= 25:
                full_audio = np.concatenate(accumulated_chunks)
                res = transcribe_audio(full_audio)
                if res and res.get("text"):
                    text_queue.put({"text": res["text"], "is_final": False})
                last_live_transcription_time = now

            # Check if silence threshold reached or hard timeout triggered
            silence_duration = now - last_speech_time
            if silence_duration >= silence_manager.silence_threshold or silence_manager.is_hard_timeout():
                # FINALIZE UTTERANCE
                if accumulated_chunks:
                    full_audio = np.concatenate(accumulated_chunks)
                    res = transcribe_audio(full_audio)
                    if res and res.get("text"):
                        text_queue.put({"text": res["text"], "is_final": True})

                accumulated_chunks.clear()
                idle_window.clear()
                state = "IDLE"
                silence_manager.reset()


# =====================================================
# WORD PROCESSING THREAD
# =====================================================

def word_processing_worker():
    """
    Receives complete sentence transcriptions (live or final),
    translates them at the phrase level, and renders subtitles.
    """
    while not _shutdown.is_set():
        try:
            item = text_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        try:
            hindi_text = item["text"].strip()
            is_final = item["is_final"]

            if not hindi_text:
                continue

            if not is_final:
                # Live accumulating preview (phrase translation)
                english_trans = translate_phrase(hindi_text)
                update_live(hindi_text, english_trans if english_trans else hindi_text)
            else:
                # Final completed sentence
                final_english = reconstruct_sentence(hindi_text)
                finalize(hindi_text, final_english, final_english)
                
                # Drain remaining outdated live previews in queue and trigger audio buffer clean
                while not text_queue.empty():
                    try:
                        text_queue.get_nowait()
                    except queue.Empty:
                        break
                _flush_audio.set()

        except Exception as e:
            print(f"[Processing] Error: {e}", file=sys.stderr)


# =====================================================
# SIGNAL HANDLER
# =====================================================

def _handle_exit(sig, frame):
    print("\n[System] Shutting down...", file=sys.stderr)
    _shutdown.set()
    sys.exit(0)


# =====================================================
# MAIN ENTRY POINT
# =====================================================

def main():
    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

    print("\n====================================")
    print("Realtime Hindi Translator V2 (VAD Accumulator)")
    print("====================================\n")

    audio_thread = threading.Thread(target=audio_capture_worker, daemon=True)
    stt_thread = threading.Thread(target=speech_recognition_worker, daemon=True)
    processing_thread = threading.Thread(target=word_processing_worker, daemon=True)

    audio_thread.start()
    stt_thread.start()
    processing_thread.start()

    while not _shutdown.is_set():
        time.sleep(0.5)


if __name__ == "__main__":
    main()
