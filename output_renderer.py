import sys
import shutil


# =====================================================
# ANSI ESCAPE CODES
# =====================================================
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_CYAN   = "\033[96m"     # live Hindi text
_YELLOW = "\033[93m"     # live English text
_WHITE  = "\033[97m"     # finalized Hindi (history)
_GREEN  = "\033[92m"     # finalized translated (history)
_GRAY   = "\033[90m"     # separators, labels


# =====================================================
# SUBTITLE RENDERER
# =====================================================

class SubtitleRenderer:
    """
    In-place terminal subtitle display.

    Instead of printing a new block per token (the V1
    "packet" style that produced endless scroll), this
    maintains a fixed 7-line display area and redraws
    it on every update using ANSI cursor control —
    exactly like a real-time caption/subtitle feed.

    Layout (7 fixed lines, always redrawn in place):

      ══════════════════════════════════════════════
       [history 0 — older finalized sentence, dim ]
       [history 1 — recent finalized sentence     ]
      ──────────────────────────────────────────────
       ▶  [accumulating Hindi text                ]
          [accumulating English text              ]
      ══════════════════════════════════════════════

    When a sentence is finalized it moves into the
    history rows and the live area clears — the
    older history row scrolls off to make room,
    giving the rolling-subtitle feel shown in the
    V2 architecture PDF.
    """

    _DISPLAY_LINES = 7     # total fixed-height lines
    _MAX_HISTORY   = 2     # finalized sentences to keep visible


    def __init__(self):
        self._history      = []   # list of (hindi_str, translated_str)
        self._live_hindi   = ""
        self._live_english = ""
        self._initialized  = False

        # Check once — if stdout is not a real TTY
        # (e.g. piped to a file) fall back to simple
        # line-by-line prints so ANSI codes don't
        # corrupt the output.
        self._is_tty = sys.stdout.isatty()


    # --------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------

    def _width(self):
        return shutil.get_terminal_size((80, 24)).columns

    def _fit(self, text, max_len):
        """
        Truncate text with a trailing ellipsis if it
        exceeds max_len characters. Prevents long
        sentences from wrapping and breaking the fixed
        line count assumption.
        """
        if len(text) <= max_len:
            return text
        return text[:max_len - 1] + "…"


    # --------------------------------------------------
    # CORE RENDER
    # --------------------------------------------------

    def _render(self):

        if not self._is_tty:
            self._render_fallback()
            return

        w     = self._width()
        inner = w - 4           # usable content width
        half  = inner // 2 - 4  # width per side in history row

        heavy = _GRAY + "═" * w + _RESET
        light = _GRAY + "─" * w + _RESET

        lines = []

        # ---- Line 0 : top border ----
        lines.append(heavy)

        # ---- Lines 1-2 : history slots ----
        # Always emit exactly _MAX_HISTORY lines so the
        # fixed line count stays constant regardless of
        # how many sentences have been finalized so far.
        for slot in range(self._MAX_HISTORY):

            # Map slot to the correct history entry.
            # Slot 0 = oldest (dimmer), slot 1 = newest.
            offset = len(self._history) - self._MAX_HISTORY + slot

            if 0 <= offset < len(self._history):

                h_text, t_text = self._history[offset]

                # Oldest slot is dimmer to give a
                # visual "fading out" effect.
                dim = _DIM if slot == 0 else ""

                h_fit = self._fit(h_text, half)
                t_fit = self._fit(t_text, half)

                line = (
                    f"  {dim}{_WHITE}{h_fit}{_RESET}"
                    f"  {_GRAY}→{_RESET}  "
                    f"{dim}{_GREEN}{t_fit}{_RESET}"
                )

            else:
                # Empty slot — pad with blank so line
                # count stays fixed.
                line = ""

            lines.append(line)

        # ---- Line 3 : light divider ----
        lines.append(light)

        # ---- Line 4 : live Hindi ----
        if self._live_hindi:
            lines.append(
                f"  {_BOLD}{_CYAN}▶  "
                f"{self._fit(self._live_hindi, inner - 4)}"
                f"{_RESET}"
            )
        else:
            lines.append(
                f"  {_DIM}{_GRAY}Listening...{_RESET}"
            )

        # ---- Line 5 : live English ----
        if self._live_english:
            lines.append(
                f"     {_YELLOW}"
                f"{self._fit(self._live_english, inner - 5)}"
                f"{_RESET}"
            )
        else:
            # Keep the line present (but empty) so the
            # total count is always _DISPLAY_LINES.
            lines.append("")

        # ---- Line 6 : bottom border ----
        lines.append(heavy)

        # --------------------------------------------------
        # DRAW: move cursor back to top of display area
        # then overwrite each line in place.
        # --------------------------------------------------

        if self._initialized:
            # Move cursor up by exactly _DISPLAY_LINES rows
            # to return to the first line of our display area.
            sys.stdout.write(f"\033[{self._DISPLAY_LINES}A")

        for line in lines:
            # \r  → go to column 1 of this line
            # \033[2K → erase entire line (prevents leftover
            #           characters from a wider previous draw)
            sys.stdout.write(f"\r\033[2K{line}\n")

        sys.stdout.flush()
        self._initialized = True


    def _render_fallback(self):
        """
        Simple line-by-line fallback for non-TTY contexts
        (e.g. output piped to a file or another process).
        Mirrors the information shown in the ANSI display.
        """
        if self._live_hindi:
            print(f"[LIVE] {self._live_hindi}")
            print(f"       {self._live_english}")

        elif self._history:
            h, t = self._history[-1]
            print(f"[FINAL] {h}  →  {t}")


    # =====================================================
    # PUBLIC API
    # =====================================================

    def update_live(self, hindi_words, english_words):
        """
        Call on every new token, passing the FULL
        accumulated word lists (not just the new word).
        The renderer replaces the live area entirely
        each call — no appending needed from the caller.

        Args:
            hindi_words   : list of Hindi word strings
                            accumulated since last reset
            english_words : corresponding English
                            translations (same length)
        """
        self._live_hindi   = " ".join(hindi_words)
        self._live_english = " ".join(english_words)
        self._render()


    def finalize(self, hindi, english, translated):
        """
        Call when a sentence boundary is detected
        (silence, filler, or hard timeout).

        Moves the current sentence into the history
        area, clears the live area, and redraws.
        The oldest history row rolls off if more than
        _MAX_HISTORY sentences have been finalized.

        Args:
            hindi      : raw accumulated Hindi string
            english    : raw word-for-word English
                         (kept for logging/debug)
            translated : corrected/reconstructed English
                         sentence shown in the display
        """
        if hindi.strip():
            self._history.append((hindi, translated))

            if len(self._history) > self._MAX_HISTORY:
                self._history.pop(0)

        self._live_hindi   = ""
        self._live_english = ""
        self._render()


    def clear(self):
        """
        Reset the renderer completely.
        Call on startup or when restarting a session.
        """
        self._history      = []
        self._live_hindi   = ""
        self._live_english = ""
        self._initialized  = False


# =====================================================
# MODULE-LEVEL SINGLETON
# =====================================================
# Exposes a single shared renderer instance so any
# module can call the functions below without
# managing the object lifetime themselves.
# =====================================================

_renderer = SubtitleRenderer()


def update_live(hindi_words, english_words):
    """
    Update the live accumulating subtitle area.
    Pass the full accumulated word lists, not just
    the latest word.
    """
    _renderer.update_live(hindi_words, english_words)


def finalize(hindi, english, translated):
    """
    Finalize the current sentence — move it to the
    history display and clear the live area.
    """
    _renderer.finalize(hindi, english, translated)


def clear():
    """Reset the display entirely."""
    _renderer.clear()


# --------------------------------------------------
# BACKWARD COMPATIBILITY SHIM
# --------------------------------------------------
# Keeps the old display_output() signature working
# during the transition period while main.py is
# being updated (fix #5).
# Remove once main.py is fully updated.
# --------------------------------------------------
def display_output(mode, hindi, english):
    if mode == "LIVE":
        _renderer.update_live([hindi], [english])
    elif mode == "FINAL":
        _renderer.finalize(hindi, english, english)
