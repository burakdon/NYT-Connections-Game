"""Main orchestrator: generates and validates complete Connections puzzles."""

import json
import random
import logging
from datetime import datetime, timezone

from src.config import (
    GROUPS_PER_PUZZLE, TOTAL_WORDS, RANDOM_SEED,
)
from src.llm_client import LLMClient
from src.generator.group_creator import GroupCreator, EmbeddingSelector
from src.generator.puzzle_editor import PuzzleEditor
from src.generator.difficulty import assign_colors
from src.generator.deduplicator import Deduplicator

logger = logging.getLogger(__name__)


class PuzzlePipeline:
    """End-to-end puzzle generation pipeline."""

    def __init__(self, llm: LLMClient = None, embedding_model=None,
                 word_bank: list[str] = None, nyt_path: str = None):
        self.llm = llm or LLMClient()
        self.embedding_model = embedding_model
        self.selector = EmbeddingSelector(model=embedding_model)
        self.creator = GroupCreator(self.llm, self.selector, word_bank=word_bank)
        self.editor = PuzzleEditor(self.llm)
        self.deduplicator = Deduplicator(nyt_path=nyt_path)
        self.rng = random.Random(RANDOM_SEED)
        self._puzzle_counter = 0

    def generate(self, method: str = "iterative") -> dict:
        """Generate a single puzzle.

        Args:
            method: "iterative" or "false_group"

        Returns:
            Puzzle dict matching the schema in CLAUDE.md.
        """
        logger.info(f"Generating puzzle via {method} method...")

        if method == "false_group":
            groups = self._generate_false_group()
        else:
            groups = self._generate_iterative()

        # Editor pass
        groups = self.editor.review(groups)

        # Assign difficulty colors
        groups = assign_colors(groups, model=self.embedding_model)

        # Collect all words and check for dedup
        all_words = []
        for g in groups:
            all_words.extend(g["words"])

        dedup_result = self.deduplicator.check(all_words)

        # Shuffle words for the puzzle
        shuffled_words = list(all_words)
        self.rng.shuffle(shuffled_words)

        # Build puzzle
        self._puzzle_counter += 1
        puzzle_id = f"INF-{self._puzzle_counter:05d}"

        # Clean up groups for output (remove internal fields)
        clean_groups = []
        for g in groups:
            clean_groups.append({
                "category": g["category"],
                "words": g["words"],
                "color": g.get("color", "yellow"),
                "similarity_score": g.get("similarity_score", 0.0),
            })

        puzzle = {
            "id": puzzle_id,
            "words": shuffled_words,
            "groups": clean_groups,
            "metadata": {
                "generation_method": method,
                "solver_agreement": None,  # filled by roundtable
                "solvers_used": [],
                "solver_results": {},
                "dedup_check": not dedup_result["is_duplicate"],
                "overall_difficulty": sum(
                    g.get("similarity_score", 0) for g in clean_groups
                ) / len(clean_groups) if clean_groups else 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        }

        logger.info(f"Generated puzzle {puzzle_id} with {len(all_words)} words")
        return puzzle

    def _generate_iterative(self) -> list[dict]:
        """Generate 4 groups one at a time."""
        groups = []
        for i in range(GROUPS_PER_PUZZLE):
            group = self.creator.create_group_iterative(previous_groups=groups)
            groups.append(group)
            logger.info(
                f"Group {i+1}/{GROUPS_PER_PUZZLE}: {group['category']} "
                f"({len(group['words'])} words, score={group['similarity_score']:.3f})"
            )
        return groups

    def _generate_false_group(self) -> list[dict]:
        """Generate 4 groups via the false-group method."""
        return self.creator.create_false_group_puzzle()

    def generate_batch(self, n: int, method: str = "iterative") -> list[dict]:
        """Generate n puzzles."""
        puzzles = []
        for i in range(n):
            puzzle = self.generate(method=method)
            puzzles.append(puzzle)
            logger.info(f"Batch progress: {i+1}/{n}")
        return puzzles
