class TokenBuffer:

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


    # -----------------------------
    # GET TOKENS
    # -----------------------------
    def get_tokens(self):

        return self.tokens


    # -----------------------------
    # CLEAR TOKENS
    # -----------------------------
    def clear(self):

        self.tokens = []