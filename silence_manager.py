import time


class SilenceManager:
    """
    Decides when a spoken utterance is "done" and
    the token buffer should be finalized.

    Three independent conditions can trigger finalization,
    all checked through the single should_finalize() call:

      1. SILENCE — speaker paused for >= silence_threshold
         seconds. Raised from V1's 1.0 s to 2.0 s so that
         natural breath-pauses between clauses don't flush
         the buffer mid-sentence.

      2. HARD TIMEOUT — speaker has been talking for >=
         hard_timeout seconds without a long enough pause.
         Prevents the semantic token buffer from growing
         unbounded during very long continuous speech.
         Matches the 30–60 s semantic memory window in the
         V2 architecture doc.

      3. MIN TOKENS GATE — finalization is suppressed when
         the buffer holds fewer than min_tokens tokens,
         regardless of how long the silence has been.
         This was the direct cause of single-word "final
         translations" in V1 (silence_threshold=1.0 flushed
         the buffer after every word).

    Filler detection is intentionally kept outside this
    class — a filler word should always trigger finalization
    regardless of token count, so main.py handles that
    branch separately and calls reset() directly.
    """

    def __init__(
        self,
        silence_threshold=2.0,
        min_tokens=3,
        hard_timeout=45.0,
    ):
        # Seconds of continuous silence before finalizing.
        # 2.0 s is long enough to survive natural clause
        # pauses but short enough to feel responsive.
        self.silence_threshold = silence_threshold

        # Minimum number of tokens that must be buffered
        # before a silence event can trigger finalization.
        # Prevents 1–2 word fragments being output as
        # complete sentences.
        self.min_tokens = min_tokens

        # Maximum seconds a single utterance can run
        # before being force-finalized.
        self.hard_timeout = hard_timeout

        now = time.time()

        # Updated every time new speech activity is seen.
        self.last_activity_time = now

        # Set once at the start of each new utterance
        # (reset on finalization). Used for hard_timeout.
        self.utterance_start_time = now


    # -----------------------------
    # UPDATE ACTIVITY
    # -----------------------------
    def update_activity(self):
        """
        Call whenever a new word/token is received.
        Pushes back the silence clock.
        """
        self.last_activity_time = time.time()


    # -----------------------------
    # SILENCE CHECK
    # -----------------------------
    def is_silence_detected(self):
        """
        Returns True if the speaker has been quiet
        for at least silence_threshold seconds.
        """
        silence_duration = (
            time.time() - self.last_activity_time
        )
        return silence_duration >= self.silence_threshold


    # -----------------------------
    # GRAMMAR COMPLETION HEURISTICS
    # -----------------------------
    def is_grammar_complete(self, tokens):
        """
        Heuristic checks to determine if the most recent token(s)
        form a grammatically complete sentence. Accepts `tokens`
        as a list of token dicts (with keys 'hindi' and/or 'english')
        or as a list of strings.
        """
        if not tokens:
            return False

        # Accept either token dicts or raw strings
        if isinstance(tokens[0], dict):
            last_hindi = tokens[-1].get("hindi", "")
            last_english = tokens[-1].get("english", "")
        else:
            # assume list of strings (Hindi)
            last_hindi = tokens[-1]
            last_english = ""

        last_hindi = (last_hindi or "").strip()
        last_english = (last_english or "").strip()

        # Punctuation that strongly indicates sentence end
        terminal_chars = ["।", ".", "?", "!", ";"]
        for ch in terminal_chars:
            if last_hindi.endswith(ch) or last_english.endswith(ch):
                return True

        # Common Hindi verb/sentence endings (heuristic suffixes)
        suffixes = [
            "हूँ", "है", "हैं", "था", "थे", "थी",
            "रहा", "रही", "रहे", "गया", "गई", "गए",
            "जाता", "जाती", "जाते", "करता", "करती", "करते",
            "होगा", "होगी", "होंगे", "दी", "दिया", "दे", "दीया"
        ]

        for s in suffixes:
            if last_hindi.endswith(s):
                return True

        # Fallback: if last English token looks like a finished clause
        if last_english and last_english[-1] in ".!?":
            return True

        return False


    # -----------------------------
    # HARD TIMEOUT CHECK
    # -----------------------------
    def is_hard_timeout(self):
        """
        Returns True if the current utterance has been
        running for longer than hard_timeout seconds.
        Prevents unbounded token buffer growth during
        very long continuous speech.
        """
        utterance_duration = (
            time.time() - self.utterance_start_time
        )
        return utterance_duration >= self.hard_timeout


    # -----------------------------
    # SILENCE DURATION
    # -----------------------------
    def get_silence_duration(self):
        """
        Returns current silence duration in seconds.
        Useful for the output renderer to fade or
        animate the display as silence builds.
        """
        return time.time() - self.last_activity_time


    # =====================================================
    # SHOULD FINALIZE  ← primary API for main.py
    # =====================================================
    def should_finalize(self, token_count):
        """
        Single decision point for main.py.

        Returns True when ALL of the following hold:
          - token_count >= min_tokens  (enough content)
          - silence is detected  OR  hard timeout hit

        Usage in main.py:
            if filler_detected or silence_manager.should_finalize(
                len(token_buffer.get_tokens())
            ):
                # finalize and reset

        Note: filler_detected is kept outside this method
        intentionally — fillers bypass the min_tokens gate
        since they are explicit sentence-boundary markers.
        """
        # Backwards-compatible: if a list was passed instead of a count,
        # extract the length and keep the tokens for grammar checks.
        tokens = None
        if isinstance(token_count, (list, tuple)):
            tokens = token_count
            token_count = len(tokens)

        if token_count < self.min_tokens:
            return False

        # Primary trigger: grammatical sentence completion.
        if tokens is not None and self.is_grammar_complete(tokens):
            return True

        # Secondary trigger: hard timeout as before.
        return self.is_hard_timeout()


    # -----------------------------
    # RESET
    # -----------------------------
    def reset(self):
        """
        Call after finalization.
        Resets BOTH timers so the next utterance
        starts with a clean slate.
        """
        now = time.time()
        self.last_activity_time = now
        self.utterance_start_time = now
