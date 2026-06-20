class TokenBuffer:
    """
    Stores semantic tokens for the current utterance.

    A sliding window is used so that if the speaker talks
    continuously for a very long time (until hard_timeout),
    the buffer never grows without bound.
    """

    # Maximum tokens to retain
    MAX_TOKENS = 60

    def __init__(self):
        self.tokens = []

    # -----------------------------
    # ADD TOKEN
    # -----------------------------
    def add_token(
        self,
        hindi_word,
        english_word
    ):
        token = {
            "hindi": hindi_word,
            "english": english_word
        }

        self.tokens.append(token)

        # Sliding window:
        # keep only the newest MAX_TOKENS tokens
        if len(self.tokens) > self.MAX_TOKENS:
            self.tokens.pop(0)

    # -----------------------------
    # GET TOKENS
    # -----------------------------
    def get_tokens(self):
        return self.tokens

    # -----------------------------
    # TOKEN COUNT
    # -----------------------------
    def __len__(self):
        return len(self.tokens)

    # -----------------------------
    # CLEAR TOKENS
    # -----------------------------
    def clear(self):
        self.tokens.clear()

    # -----------------------------
    # REPLACE TOKENS
    # -----------------------------
    def replace_tokens(self, start_index, new_tokens):
        """
        Replace tokens starting at `start_index` with `new_tokens`.

        `new_tokens` is a list of token dicts: {"hindi": ..., "english": ...}.

        Behavior:
        - If `start_index` is negative or greater than current length,
          it is clamped to valid bounds.
        - After replacement, enforce the sliding window `MAX_TOKENS`.
        """

        if start_index < 0:
            start_index = 0

        if start_index > len(self.tokens):
            start_index = len(self.tokens)

        # Perform replacement
        self.tokens = self.tokens[:start_index] + list(new_tokens)

        # Enforce sliding window of newest MAX_TOKENS tokens
        if len(self.tokens) > self.MAX_TOKENS:
            self.tokens = self.tokens[-self.MAX_TOKENS:]
