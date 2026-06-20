"""
hypothesis_manager.py

Maintains a stable streaming transcript from Whisper.

Instead of permanently accepting every new Whisper output,
this module compares the previous hypothesis with the new one
and commits only the genuinely stable words.

Example

Previous:
    मैं घर

Current:
    मैं घर जा

Stable:
    मैं घर

Pending:
    जा

-------------------------

Previous:
    मैं घर जा

Current:
    मैं घर जाता हूँ

Stable:
    मैं घर

Pending:
    जाता हूँ

Only stable words should enter the TokenBuffer.
"""

from typing import List


class HypothesisManager:

    def __init__(self):

        # Previous Whisper hypothesis
        self.previous_words: List[str] = []

        # Words already committed to TokenBuffer
        self.committed_words: List[str] = []


    # --------------------------------------------------
    # Longest Common Prefix
    # --------------------------------------------------

    def _common_prefix_length(
        self,
        a: List[str],
        b: List[str]
    ) -> int:

        i = 0

        while (
            i < len(a)
            and i < len(b)
            and a[i] == b[i]
        ):
            i += 1

        return i


    # --------------------------------------------------
    # Update Hypothesis
    # --------------------------------------------------

    def update(self, current_words: List[str]):

        """
        Parameters
        ----------
        current_words
            Latest Whisper hypothesis.

        Returns
        -------
        newly_stable_words
            Words that became stable since the previous update.
        """

        prefix = self._common_prefix_length(
            self.previous_words,
            current_words
        )

        newly_stable = []

        # Commit only words that are stable
        while len(self.committed_words) < prefix:

            newly_stable.append(
                current_words[len(self.committed_words)]
            )

            self.committed_words.append(
                current_words[len(self.committed_words)]
            )

        self.previous_words = current_words.copy()

        return newly_stable


    # --------------------------------------------------
    # Pending Words
    # --------------------------------------------------

    def get_pending_words(self):

        return self.previous_words[
            len(self.committed_words):
        ]


    # --------------------------------------------------
    # Current Transcript
    # --------------------------------------------------

    def get_transcript(self):

        return self.previous_words.copy()


    # --------------------------------------------------
    # Reset
    # --------------------------------------------------

    def reset(self):

        self.previous_words.clear()
        self.committed_words.clear()