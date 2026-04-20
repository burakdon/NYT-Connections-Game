"""LLM-based solver using Claude with Chain-of-Thought prompting."""

import json
import logging

from src.llm_client import LLMClient
from src.config import LLM_SOLVER_TEMPERATURE
from src.generator.prompts import format_solver

logger = logging.getLogger(__name__)


class LLMSolver:
    """Solves Connections puzzles using Claude with CoT reasoning."""

    def __init__(self, llm: LLMClient = None):
        self.llm = llm or LLMClient()

    def solve(self, words: list[str]) -> list[dict]:
        """Ask Claude to solve the puzzle via chain-of-thought.

        Returns list of {"words": [...], "category": str} dicts.
        """
        words_upper = [w.upper() for w in words]
        system, user = format_solver(words_upper)
        response = self.llm.complete(system, user, temperature=LLM_SOLVER_TEMPERATURE)

        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("LLM solver returned non-JSON; attempting extraction.")
            return self._extract_groups_from_text(response, words_upper)

        groups = []
        for g in result.get("groups", []):
            groups.append({
                "words": [w.upper() for w in g.get("words", [])],
                "category": g.get("category", "UNKNOWN"),
                "score": 0.0,
            })

        logger.info(f"LLM solver found {len(groups)} groups")
        return groups

    @staticmethod
    def _extract_groups_from_text(text: str, all_words: list[str]) -> list[dict]:
        """Fallback: try to extract word groups from free-text response."""
        groups = []
        words_set = set(all_words)
        current_group = []

        for word in all_words:
            if word in words_set:
                current_group.append(word)
                if len(current_group) == 4:
                    groups.append({
                        "words": current_group,
                        "category": "EXTRACTED",
                        "score": 0.0,
                    })
                    current_group = []

        return groups
