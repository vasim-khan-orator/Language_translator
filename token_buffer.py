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
        # Return a shallow copy to prevent external mutation
        return list(self.tokens)

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

    # -----------------------------
    # INSERT TOKENS (no replacement)
    # -----------------------------
    def insert_tokens(self, start_index, new_tokens):
        """
        Insert `new_tokens` at `start_index` without removing existing tokens.
        """
        if start_index < 0:
            start_index = 0

        if start_index > len(self.tokens):
            start_index = len(self.tokens)

        self.tokens = self.tokens[:start_index] + list(new_tokens) + self.tokens[start_index:]

        if len(self.tokens) > self.MAX_TOKENS:
            self.tokens = self.tokens[-self.MAX_TOKENS:]

    # -----------------------------
    # APPEND TOKENS
    # -----------------------------
    def append_tokens(self, new_tokens):
        """Append a list of token dicts to the buffer."""
        self.tokens.extend(list(new_tokens))

        if len(self.tokens) > self.MAX_TOKENS:
            self.tokens = self.tokens[-self.MAX_TOKENS:]

    # -----------------------------
    # UPDATE SINGLE TOKEN
    # -----------------------------
    def update_token(self, index, hindi=None, english=None):
        """Update fields of the token at `index`. Raises IndexError if out of range."""
        if index < 0:
            index = 0
        if index >= len(self.tokens):
            raise IndexError("token index out of range")

        token = self.tokens[index]
        if hindi is not None:
            token["hindi"] = hindi
        if english is not None:
            token["english"] = english

    # -----------------------------
    # DELETE RANGE
    # -----------------------------
    def delete_range(self, start_index, count=1):
        """Delete `count` tokens starting at `start_index`."""
        if count <= 0:
            return

        if start_index < 0:
            start_index = 0

        end_index = start_index + count
        # Clamp bounds
        start_index = min(start_index, len(self.tokens))
        end_index = min(end_index, len(self.tokens))

        if start_index < end_index:
            del self.tokens[start_index:end_index]
