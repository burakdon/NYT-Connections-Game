"""Deduplication: checks generated puzzles against known NYT puzzles."""

import json
import logging
from pathlib import Path

from src.config import NYT_PUZZLES_PATH, DEDUP_WORD_OVERLAP_THRESHOLD

logger = logging.getLogger(__name__)


class Deduplicator:
    """Checks generated puzzles against the ground truth NYT dataset.

    Two distinct checks:
      1. `check(puzzle_words)` -- legacy: flags if >6 words overlap with any
         single NYT puzzle (16-word-set overlap).
      2. `check_groups(groups)` -- NEW: flags if any generated 4-word group
         exactly matches any of the ~2,216 past NYT 4-word groups.

    The rubric says "If you generate a past connections puzzle you will
    automatically fail", so we enforce BOTH checks. A puzzle is a duplicate
    if either trigger fires.
    """

    def __init__(self, nyt_path: str = None):
        self.nyt_path = nyt_path or NYT_PUZZLES_PATH
        self._nyt_puzzles = None
        self._nyt_group_set = None  # set of frozensets (past NYT 4-word groups)

    @property
    def nyt_puzzles(self) -> list[dict]:
        if self._nyt_puzzles is None:
            self._nyt_puzzles = self._load_nyt_puzzles()
        return self._nyt_puzzles

    @property
    def nyt_group_set(self) -> set:
        """Lazy-load a set of frozensets, one per past NYT 4-word group."""
        if self._nyt_group_set is None:
            groups = set()
            for p in self.nyt_puzzles:
                for ans in p.get("answers", []):
                    ws = ans.get("words", [])
                    if ws:
                        groups.add(frozenset(w.upper() for w in ws))
            self._nyt_group_set = groups
            logger.info(f"Deduplicator: indexed {len(groups)} past NYT groups")
        return self._nyt_group_set

    def _load_nyt_puzzles(self) -> list[dict]:
        path = Path(self.nyt_path)
        if not path.exists():
            logger.warning(f"NYT puzzles file not found at {self.nyt_path}; dedup disabled.")
            return []
        with open(path) as f:
            return json.load(f)

    def check_groups(self, groups: list) -> dict:
        """Check if any of the generated 4-word groups match a past NYT group.

        Args:
            groups: list of dicts with "words" key, OR list of list[str].

        Returns:
            {
                "exact_group_match": bool,
                "matching_groups": list of frozensets that matched
            }
        """
        matches = []
        for g in groups:
            if isinstance(g, dict):
                ws = g.get("words", [])
            else:
                ws = g
            key = frozenset(w.upper() for w in ws)
            if key in self.nyt_group_set:
                matches.append(key)

        return {
            "exact_group_match": len(matches) > 0,
            "matching_groups": matches,
        }

    def check(self, puzzle_words: list[str], groups: list = None) -> dict:
        """Check a generated puzzle against all known NYT puzzles.

        Args:
            puzzle_words: the 16 words of the puzzle.
            groups: optional -- the 4 groups of the puzzle. If provided, we also
                check for exact 4-word group matches against past NYT groups.

        Returns:
            {
                "is_duplicate": bool,              # True if EITHER trigger fires
                "max_overlap": int,                # 16-word overlap count
                "overlapping_puzzle": str or None,
                "exact_group_match": bool,         # 4-word group match check
                "matching_groups": list[frozenset],
                "reason": str                      # why dedup flagged this
            }
        """
        words_upper = {w.upper() for w in puzzle_words}
        max_overlap = 0
        worst_match = None

        for nyt in self.nyt_puzzles:
            nyt_words = {w.upper() for w in nyt.get("words", [])}
            overlap = len(words_upper & nyt_words)
            if overlap > max_overlap:
                max_overlap = overlap
                worst_match = nyt.get("contest", nyt.get("date", "unknown"))

        word_dup = max_overlap > DEDUP_WORD_OVERLAP_THRESHOLD

        # Group-level check (only if groups provided)
        group_result = {"exact_group_match": False, "matching_groups": []}
        if groups is not None:
            group_result = self.check_groups(groups)

        is_dup = word_dup or group_result["exact_group_match"]

        reason = ""
        if word_dup and group_result["exact_group_match"]:
            reason = "word_overlap+group_match"
        elif word_dup:
            reason = "word_overlap"
        elif group_result["exact_group_match"]:
            reason = "exact_group_match"

        if is_dup:
            logger.warning(f"Duplicate detected: reason={reason}")

        return {
            "is_duplicate": is_dup,
            "max_overlap": max_overlap,
            "overlapping_puzzle": worst_match if max_overlap > 0 else None,
            "exact_group_match": group_result["exact_group_match"],
            "matching_groups": group_result["matching_groups"],
            "reason": reason,
            # Kept for backward compat with existing callers
            "category_matches": [],
        }
