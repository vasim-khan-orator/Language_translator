"""
Context manager for translations.

Stores the current accumulating sentence, the previous finalized
sentence, and a bounded conversation history. Provides helpers to
produce a contextualized Hindi string that can be passed to the
translation model so translations can use prior-sentence context.

API (module-level helpers):
    update_current(hindi_str)
    get_current()
    finalize_current(translated=None)
    get_context_for_translation(hindi_sentence, max_prev=2)
    get_conversation()
    clear()
"""
from typing import List, Tuple, Optional
import threading


class ContextManager:
    def __init__(self, max_history: int = 50):
        self._lock = threading.Lock()
        self._current: str = ""            # accumulating current Hindi sentence
        self._previous: Optional[str] = None
        self._conversation: List[Tuple[str, Optional[str]]] = []  # list of (hindi, translated)
        self._max_history = max_history

    def update_current(self, hindi_text: str):
        """Replace the current accumulating sentence."""
        with self._lock:
            self._current = hindi_text or ""

    def append_to_current(self, hindi_text: str):
        """Append words to the current accumulating sentence."""
        if not hindi_text:
            return
        with self._lock:
            if self._current:
                self._current = f"{self._current} {hindi_text}"
            else:
                self._current = hindi_text

    def get_current(self) -> str:
        with self._lock:
            return self._current

    def finalize_current(self, translated: Optional[str] = None):
        """
        Move the current sentence into conversation history and
        mark it as previous. Clears the current accumulator.
        """
        with self._lock:
            hindi = self._current.strip()
            if hindi:
                self._conversation.append((hindi, translated))
                if len(self._conversation) > self._max_history:
                    # drop oldest
                    self._conversation.pop(0)
                self._previous = hindi

            # clear current
            self._current = ""

    def get_previous(self) -> Optional[str]:
        with self._lock:
            return self._previous

    def get_conversation(self) -> List[Tuple[str, Optional[str]]]:
        with self._lock:
            return list(self._conversation)

    def clear(self):
        with self._lock:
            self._current = ""
            self._previous = None
            self._conversation = []

    def get_context_for_translation(self, hindi_sentence: str, max_prev: int = 2) -> str:
        """
        Produce a contextualized Hindi string to pass to the sentence-level
        translator by prepending up to `max_prev` previous finalized
        sentences. The order is oldest → newest → target sentence so the
        model sees prior context first.
        """
        with self._lock:
            parts: List[str] = []
            if self._conversation:
                # take up to max_prev most recent finalized sentences
                prev_items = self._conversation[-max_prev:]
                parts.extend(h for h, _ in prev_items if h)

            if hindi_sentence and hindi_sentence.strip():
                parts.append(hindi_sentence.strip())

            return " \n ".join(parts)


# Module-level singleton for easy import/use
_manager = ContextManager()


def update_current(hindi_str: str):
    _manager.update_current(hindi_str)


def append_to_current(hindi_str: str):
    _manager.append_to_current(hindi_str)


def get_current() -> str:
    return _manager.get_current()


def finalize_current(translated: Optional[str] = None):
    _manager.finalize_current(translated)


def get_previous() -> Optional[str]:
    return _manager.get_previous()


def get_conversation() -> List[Tuple[str, Optional[str]]]:
    return _manager.get_conversation()


def clear():
    _manager.clear()


def get_context_for_translation(hindi_sentence: str, max_prev: int = 2) -> str:
    return _manager.get_context_for_translation(hindi_sentence, max_prev=max_prev)


__all__ = [
    "update_current",
    "append_to_current",
    "get_current",
    "finalize_current",
    "get_previous",
    "get_conversation",
    "clear",
    "get_context_for_translation",
]
