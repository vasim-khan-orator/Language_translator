"""
Manage Whisper hypothesis updates to prefer in-place
replacements for partial word edits while still handling
rolling-window suffix overlaps.

API:
    hm = HypothesisManager()
    remove_count, replacement_words = hm.update_and_diff(new_words)

This returns the number of words to remove from the existing
token buffer and the list of replacement words to append.
"""
from typing import List, Tuple


class HypothesisManager:
    def __init__(self):
        self.prev_words: List[str] = []

    def _suffix_prefix_overlap(self, prev: List[str], new: List[str]) -> int:
        max_check = min(len(prev), len(new))
        overlap_len = 0
        for i in range(max_check, 0, -1):
            if prev[-i:] == new[:i]:
                overlap_len = i
                break
        return overlap_len

    def _common_prefix_len(self, prev: List[str], new: List[str]) -> int:
        max_check = min(len(prev), len(new))
        i = 0
        while i < max_check and prev[i] == new[i]:
            i += 1
        return i

    def update_and_diff(self, new_words: List[str]) -> Tuple[int, List[str]]:
        """
        Compute an incremental diff between the previously-seen
        hypothesis and the incoming `new_words` list.

        Strategy: compute both the trailing-suffix -> leading-prefix
        overlap (best for rolling windows) and the common-prefix
        match (best for partial-word replacements). Choose the
        strategy that minimizes the number of words to remove from
        the previous hypothesis.
        """
        prev = self.prev_words

        if not prev:
            remove_count = 0
            replacement = new_words
            self.prev_words = new_words
            return remove_count, replacement

        if not new_words:
            remove_count = len(prev)
            replacement = []
            self.prev_words = []
            return remove_count, replacement

        suffix_overlap = self._suffix_prefix_overlap(prev, new_words)
        prefix_common = self._common_prefix_len(prev, new_words)

        # Compute remove counts for both strategies
        remove_suffix = len(prev) - suffix_overlap
        remove_prefix = len(prev) - prefix_common

        if remove_suffix <= remove_prefix:
            # Prefer suffix->prefix overlap when it reduces removals
            remove_count = remove_suffix
            replacement = new_words[suffix_overlap:]
        else:
            remove_count = remove_prefix
            replacement = new_words[prefix_common:]

        # Update stored hypothesis
        self.prev_words = new_words

        return remove_count, replacement


__all__ = ["HypothesisManager"]
